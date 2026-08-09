"""Order validation. First failure rejects the order (not the whole decision)."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import date
from typing import Any

from tradingbench.sim.ledger import Ledger


@dataclass(frozen=True)
class Order:
    symbol: str
    side: str  # buy | sell
    notional_usd: float
    thesis: str = ""
    confidence: float | None = None
    horizon_steps: int | None = None

    @staticmethod
    def from_dict(d: dict[str, Any]) -> "Order":
        return Order(
            symbol=str(d["symbol"]).upper(),
            side=str(d["side"]).lower(),
            notional_usd=float(d["notional_usd"]),
            thesis=str(d.get("thesis") or ""),
            confidence=float(d["confidence"]) if d.get("confidence") is not None else None,
            horizon_steps=int(d["horizon_steps"]) if d.get("horizon_steps") is not None else None,
        )

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class Violation:
    code: str
    symbol: str | None
    detail: str
    step: int | None = None

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ValidatedOrder:
    order: Order
    accepted: bool
    violation: Violation | None = None
    # filled later by execution
    qty: float | None = None
    fill_price: float | None = None
    fee: float | None = None
    flags: list[str] | None = None

    def to_dict(self) -> dict:
        return {
            "order": self.order.to_dict(),
            "accepted": self.accepted,
            "violation": self.violation.to_dict() if self.violation else None,
            "qty": self.qty,
            "fill_price": self.fill_price,
            "fee": self.fee,
            "flags": self.flags or [],
        }


MIN_ORDER_USD = 10.0
MAX_POSITION_WEIGHT = 0.25


def validate_orders(
    orders: list[Order],
    ledger: Ledger,
    prices: dict[str, float],  # last close (for position valuation)
    open_prices: dict[str, float | None],  # t+1 open; None => NO_MARKET
    universe_symbols: set[str],
    asset_class: dict[str, str],
    nav: float,
    max_position_weight: float = MAX_POSITION_WEIGHT,
    min_order_usd: float = MIN_ORDER_USD,
    step: int | None = None,
) -> list[ValidatedOrder]:
    """Validate each order sequentially, tracking provisional cash/positions."""
    cash = ledger.cash
    # provisional qty map
    qty_map = {s: p.qty for s, p in ledger.positions.items()}
    # provisional market values using last prices
    results: list[ValidatedOrder] = []

    for order in orders:
        def reject(code: str, detail: str) -> ValidatedOrder:
            return ValidatedOrder(
                order=order,
                accepted=False,
                violation=Violation(code=code, symbol=order.symbol, detail=detail, step=step),
            )

        if order.side not in ("buy", "sell") or order.notional_usd is None:
            results.append(reject("MALFORMED", "side must be buy|sell and notional_usd required"))
            continue

        if order.notional_usd < min_order_usd:
            results.append(
                reject("DUST", f"notional ${order.notional_usd:.2f} < min ${min_order_usd:.2f}")
            )
            continue

        if order.symbol not in universe_symbols:
            results.append(reject("UNKNOWN_SYMBOL", f"{order.symbol} not in universe as of t"))
            continue

        open_px = open_prices.get(order.symbol)
        if open_px is None or open_px <= 0:
            results.append(reject("NO_MARKET", f"no bar on fill date for {order.symbol}"))
            continue

        last_px = prices.get(order.symbol, open_px)

        if order.side == "buy":
            # fees estimated at 5 bps of notional
            est_fee = order.notional_usd * 5e-4
            total_needed = order.notional_usd + est_fee
            if total_needed > cash + 1e-9:
                results.append(
                    reject(
                        "INSUFFICIENT_CASH",
                        f"need ${total_needed:.2f}, have ${cash:.2f}",
                    )
                )
                continue

            # provisional weight check after buy
            buy_qty = order.notional_usd / open_px
            new_qty = qty_map.get(order.symbol, 0.0) + buy_qty
            # approximate NAV after spend
            provisional_nav = max(nav, 1e-9)
            new_weight = (new_qty * open_px) / provisional_nav
            if new_weight > max_position_weight + 1e-9:
                results.append(
                    reject(
                        "POSITION_CAP",
                        f"would reach {new_weight:.1%} of NAV (cap {max_position_weight:.0%})",
                    )
                )
                continue

            cash -= total_needed
            qty_map[order.symbol] = new_qty
            results.append(ValidatedOrder(order=order, accepted=True))

        else:  # sell
            held_qty = qty_map.get(order.symbol, 0.0)
            if held_qty <= 0:
                results.append(reject("NO_SHORTING", f"no long position in {order.symbol}"))
                continue

            held_value = held_qty * last_px
            if order.notional_usd > held_value + 1e-6:
                results.append(
                    reject(
                        "INSUFFICIENT_POSITION",
                        f"sell notional ${order.notional_usd:.2f} > holding MV ${held_value:.2f}",
                    )
                )
                continue

            sell_qty = min(held_qty, order.notional_usd / open_px)
            if sell_qty <= 0:
                results.append(reject("DUST", "sell qty rounds to zero"))
                continue

            est_fee = order.notional_usd * 5e-4
            cash += order.notional_usd - est_fee
            qty_map[order.symbol] = held_qty - sell_qty
            if qty_map[order.symbol] < 1e-12:
                qty_map[order.symbol] = 0.0
            results.append(ValidatedOrder(order=order, accepted=True))

    return results
