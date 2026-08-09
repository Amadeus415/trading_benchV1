"""Portfolio ledger: cash, positions, mark-to-market, corporate actions."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass, field
from datetime import date
from typing import Mapping


@dataclass(frozen=True)
class Position:
    symbol: str
    qty: float
    avg_cost: float  # cost basis per unit, split-adjusted on ex-date

    def market_value(self, price: float) -> float:
        return self.qty * price

    def unrealized_pnl(self, price: float) -> float:
        return self.qty * (price - self.avg_cost)

    def unrealized_pnl_pct(self, price: float) -> float:
        if self.avg_cost == 0:
            return 0.0
        return (price - self.avg_cost) / self.avg_cost


@dataclass(frozen=True)
class LedgerState:
    ts: date
    cash: float
    positions: dict[str, Position]
    nav: float
    realized_pnl: float
    fees_paid: float

    def to_dict(self) -> dict:
        return {
            "ts": self.ts.isoformat(),
            "cash": round(self.cash, 6),
            "positions": {
                s: {"symbol": p.symbol, "qty": p.qty, "avg_cost": p.avg_cost}
                for s, p in self.positions.items()
            },
            "nav": round(self.nav, 6),
            "realized_pnl": round(self.realized_pnl, 6),
            "fees_paid": round(self.fees_paid, 6),
        }


class Ledger:
    """Mutable portfolio accounting. Snapshot via `state()`."""

    def __init__(self, cash: float, as_of: date):
        self.cash = float(cash)
        self.positions: dict[str, Position] = {}
        self.realized_pnl = 0.0
        self.fees_paid = 0.0
        self.ts = as_of
        self._held_steps: dict[str, int] = {}  # symbol -> steps held (decision cadence)

    def state(self, prices: Mapping[str, float], ts: date | None = None) -> LedgerState:
        ts = ts or self.ts
        pos_value = 0.0
        for sym, pos in self.positions.items():
            px = prices.get(sym)
            if px is None:
                # fallback: mark at cost if price missing (should be rare)
                px = pos.avg_cost
            pos_value += pos.market_value(px)
        nav = self.cash + pos_value
        return LedgerState(
            ts=ts,
            cash=self.cash,
            positions=deepcopy(self.positions),
            nav=nav,
            realized_pnl=self.realized_pnl,
            fees_paid=self.fees_paid,
        )

    def apply_buy(self, symbol: str, qty: float, price: float, fee: float) -> None:
        cost = qty * price + fee
        self.cash -= cost
        self.fees_paid += fee
        if symbol in self.positions:
            old = self.positions[symbol]
            new_qty = old.qty + qty
            new_cost = (old.qty * old.avg_cost + qty * price) / new_qty
            self.positions[symbol] = Position(symbol, new_qty, new_cost)
        else:
            self.positions[symbol] = Position(symbol, qty, price)
            self._held_steps[symbol] = 0

    def apply_sell(self, symbol: str, qty: float, price: float, fee: float) -> None:
        pos = self.positions[symbol]
        proceeds = qty * price - fee
        self.cash += proceeds
        self.fees_paid += fee
        realized = qty * (price - pos.avg_cost) - fee
        self.realized_pnl += realized
        remaining = pos.qty - qty
        if remaining < 1e-12:
            del self.positions[symbol]
            self._held_steps.pop(symbol, None)
        else:
            self.positions[symbol] = Position(symbol, remaining, pos.avg_cost)

    def apply_split(self, symbol: str, ratio: float) -> None:
        if symbol not in self.positions or ratio <= 0:
            return
        pos = self.positions[symbol]
        self.positions[symbol] = Position(
            symbol=symbol,
            qty=pos.qty * ratio,
            avg_cost=pos.avg_cost / ratio,
        )

    def apply_dividend(self, symbol: str, amount_per_share: float) -> None:
        if symbol not in self.positions:
            return
        credit = self.positions[symbol].qty * amount_per_share
        self.cash += credit

    def bump_held_steps(self) -> None:
        for sym in list(self._held_steps):
            if sym in self.positions:
                self._held_steps[sym] += 1
            else:
                del self._held_steps[sym]

    def held_steps(self, symbol: str) -> int:
        return self._held_steps.get(symbol, 0)

    def position_qty(self, symbol: str) -> float:
        return self.positions[symbol].qty if symbol in self.positions else 0.0
