"""Fill model: next-session open with slippage and commission."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

from tradingbench.sim.ledger import Ledger
from tradingbench.sim.validate import Order, ValidatedOrder


SLIPPAGE_BPS = {"equity": 10.0, "crypto": 25.0}
COMMISSION_BPS = 5.0
GAP_THRESHOLD = 0.20


@dataclass
class Fill:
    symbol: str
    side: str
    qty: float
    fill_price: float
    notional: float
    fee: float
    flags: list[str]

    def to_dict(self) -> dict:
        return {
            "symbol": self.symbol,
            "side": self.side,
            "qty": self.qty,
            "fill_price": self.fill_price,
            "notional": self.notional,
            "fee": self.fee,
            "flags": self.flags,
        }


def fill_price(
    side: str,
    open_px: float,
    close_prev: float | None,
    asset_class: str,
) -> tuple[float, list[str]]:
    """Compute fill price and any flags (e.g. LARGE_GAP)."""
    flags: list[str] = []
    slip_bps = SLIPPAGE_BPS.get(asset_class, 10.0)
    side_sign = 1.0 if side == "buy" else -1.0
    px = open_px * (1.0 + side_sign * slip_bps / 1e4)

    if close_prev is not None and close_prev > 0:
        gap = abs(open_px / close_prev - 1.0)
        if gap > GAP_THRESHOLD:
            # fill at open (no extra slippage beyond model), flag gap
            px = open_px
            flags.append("LARGE_GAP")

    return px, flags


def execute_orders(
    validated: list[ValidatedOrder],
    ledger: Ledger,
    open_prices: dict[str, float],
    close_prev: dict[str, float],
    asset_class: dict[str, str],
    fill_date: date,
) -> list[Fill]:
    """Execute accepted orders against the ledger. Mutates ledger."""
    fills: list[Fill] = []
    ledger.ts = fill_date

    for vo in validated:
        if not vo.accepted:
            continue
        order = vo.order
        open_px = open_prices[order.symbol]
        ac = asset_class.get(order.symbol, "equity")
        px, flags = fill_price(
            order.side,
            open_px,
            close_prev.get(order.symbol),
            ac,
        )
        fee = order.notional_usd * (COMMISSION_BPS / 1e4)

        if order.side == "buy":
            # re-check cash with actual fee (tiny drift vs validation estimate)
            if order.notional_usd + fee > ledger.cash + 1e-6:
                vo.accepted = False
                # leave as rejected at execution — rare edge
                continue
            qty = order.notional_usd / px
            ledger.apply_buy(order.symbol, qty, px, fee)
        else:
            held = ledger.position_qty(order.symbol)
            if held <= 0:
                continue
            qty = min(held, order.notional_usd / px)
            notional = qty * px
            fee = notional * (COMMISSION_BPS / 1e4)
            ledger.apply_sell(order.symbol, qty, px, fee)
            order = Order(
                symbol=order.symbol,
                side=order.side,
                notional_usd=notional,
                thesis=order.thesis,
                confidence=order.confidence,
                horizon_steps=order.horizon_steps,
            )

        vo.qty = qty
        vo.fill_price = px
        vo.fee = fee
        vo.flags = flags
        fills.append(
            Fill(
                symbol=order.symbol,
                side=order.side,
                qty=qty,
                fill_price=px,
                notional=qty * px,
                fee=fee,
                flags=flags,
            )
        )

    return fills
