"""Episode engine: decision-day step loop with weekly cadence."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Any

import pandas as pd

from tradingbench.data.store import PointInTimeStore, Snapshot
from tradingbench.sim.execution import Fill, execute_orders
from tradingbench.sim.ledger import Ledger, LedgerState
from tradingbench.sim.validate import Order, ValidatedOrder, Violation, validate_orders


@dataclass
class StepResult:
    step: int
    decision_date: date
    fill_date: date
    orders_in: list[Order]
    validated: list[ValidatedOrder]
    fills: list[Fill]
    violations: list[Violation]
    ledger_after: LedgerState
    daily_marks: list[LedgerState] = field(default_factory=list)


class EpisodeEngine:
    """Deterministic paper-trading engine.

    Timeline (per MVP_SPEC §5.1):
      day t (close)   observation as_of = t; model returns Decision
      day t+1 (open)  orders validated and filled
      day t+1..t+7    daily mark-to-market
      day t+7 (close) next decision
    """

    def __init__(
        self,
        snapshot: Snapshot,
        starting_cash: float = 1000.0,
        max_position_weight: float = 0.25,
        min_order_usd: float = 10.0,
        steps: int = 12,
        start_date: date | None = None,
    ):
        self.snapshot = snapshot
        self.starting_cash = starting_cash
        self.max_position_weight = max_position_weight
        self.min_order_usd = min_order_usd
        self.steps = steps

        full_dates = sorted(set(snapshot.prices["date"]))
        if start_date is None:
            start_date = full_dates[0]
        self.start_date = start_date
        self._all_dates = full_dates

        # Decision dates: weekly from start_date, using trading calendar
        self.decision_dates = self._build_decision_calendar(start_date, steps)
        self.ledger = Ledger(cash=starting_cash, as_of=self.decision_dates[0])
        self.violations: list[Violation] = []
        self.daily_rows: list[dict] = []
        self.step_results: list[StepResult] = []
        self._asset_class = self._build_asset_class_map()
        self._last_step_violations: list[Violation] = []

    def _build_asset_class_map(self) -> dict[str, str]:
        u = self.snapshot.universe
        return {row["symbol"]: row["asset_class"] for _, row in u.iterrows()}

    def _build_decision_calendar(self, start: date, steps: int) -> list[date]:
        """Pick `steps` weekly decision dates on or after start that exist in the calendar."""
        dates = [d for d in self._all_dates if d >= start]
        if not dates:
            raise ValueError(f"No trading dates on/after {start}")
        # Align start to first available trading day
        decision = [dates[0]]
        cursor = dates[0]
        while len(decision) < steps:
            target = cursor + timedelta(days=7)
            # next trading day on or after target
            nxt = next((d for d in dates if d >= target), None)
            if nxt is None:
                break
            decision.append(nxt)
            cursor = nxt
        if len(decision) < steps:
            raise ValueError(
                f"Not enough calendar for {steps} weekly steps from {start}; got {len(decision)}"
            )
        return decision

    def agent_store(self, step: int) -> PointInTimeStore:
        """Store for building observations (as_of = decision date t)."""
        return PointInTimeStore(self.snapshot, self.decision_dates[step])

    def sim_store(self, fill_date: date) -> PointInTimeStore:
        """Store for fills (as_of = t+1 fill date). Separate object from agent store."""
        return PointInTimeStore(self.snapshot, fill_date)

    def fill_date_for(self, decision_date: date) -> date | None:
        """Next trading session after decision date."""
        candidates = [d for d in self._all_dates if d > decision_date]
        return candidates[0] if candidates else None

    def last_step_violations(self) -> list[Violation]:
        return list(self._last_step_violations)

    def mark_prices(self, as_of: date, symbols: list[str] | None = None) -> dict[str, float]:
        store = PointInTimeStore(self.snapshot, as_of)
        if symbols is None:
            symbols = list(self.ledger.positions.keys())
        out: dict[str, float] = {}
        for s in symbols:
            px = store.price_on(s)
            if px is not None:
                out[s] = px
        return out

    def current_nav(self, as_of: date | None = None) -> float:
        as_of = as_of or self.ledger.ts
        prices = self.mark_prices(as_of, list(self.ledger.positions.keys()))
        return self.ledger.state(prices, as_of).nav

    def apply_corporate_actions(self, on: date) -> None:
        from tradingbench.data.corporate_actions import apply_corporate_actions

        store = PointInTimeStore(self.snapshot, on)
        apply_corporate_actions(self.ledger, store, on)

    def step(self, step_idx: int, orders: list[Order] | list[dict]) -> StepResult:
        """Run one decision step: validate → fill at t+1 open → MTM until next decision."""
        if step_idx >= len(self.decision_dates):
            raise IndexError(f"step {step_idx} out of range")

        decision_date = self.decision_dates[step_idx]
        fill_date = self.fill_date_for(decision_date)
        if fill_date is None:
            raise RuntimeError(f"No fill date after decision {decision_date}")

        # Normalize orders
        parsed: list[Order] = []
        for o in orders:
            if isinstance(o, Order):
                parsed.append(o)
            else:
                parsed.append(Order.from_dict(o))

        # Agent-side prices (close at t)
        agent_store = self.agent_store(step_idx)
        uni = agent_store.universe()
        universe_symbols = set(uni["symbol"].tolist())
        last_prices = {
            s: agent_store.price_on(s)
            for s in universe_symbols
        }
        last_prices = {s: p for s, p in last_prices.items() if p is not None}

        # Sim-side open prices at t+1
        sim_store = self.sim_store(fill_date)
        open_prices: dict[str, float | None] = {}
        for s in universe_symbols | {o.symbol for o in parsed}:
            bar = sim_store.bar_on(s, fill_date)
            open_prices[s] = bar["open"] if bar else None

        nav = self.ledger.state(last_prices, decision_date).nav

        validated = validate_orders(
            orders=parsed,
            ledger=self.ledger,
            prices=last_prices,
            open_prices=open_prices,
            universe_symbols=universe_symbols,
            asset_class=self._asset_class,
            nav=nav,
            max_position_weight=self.max_position_weight,
            min_order_usd=self.min_order_usd,
            step=step_idx,
        )

        violations = [v.violation for v in validated if v.violation is not None]
        self.violations.extend(violations)
        self._last_step_violations = violations

        # Corporate actions on fill date before trading
        self.apply_corporate_actions(fill_date)

        open_ok = {s: p for s, p in open_prices.items() if p is not None}
        fills = execute_orders(
            validated=validated,
            ledger=self.ledger,
            open_prices=open_ok,
            close_prev=last_prices,
            asset_class=self._asset_class,
            fill_date=fill_date,
        )

        # Daily marks from fill_date through day before next decision (or end)
        if step_idx + 1 < len(self.decision_dates):
            end_mark = self.decision_dates[step_idx + 1]
        else:
            # final week: mark 7 calendar days of trading dates
            end_mark = fill_date + timedelta(days=7)
            later = [d for d in self._all_dates if d >= fill_date]
            if later:
                end_mark = later[min(5, len(later) - 1)]

        mark_dates = [d for d in self._all_dates if fill_date <= d <= end_mark]
        daily_marks: list[LedgerState] = []
        for d in mark_dates:
            self.apply_corporate_actions(d)
            prices = self.mark_prices(d, list(self.ledger.positions.keys()))
            st = self.ledger.state(prices, d)
            daily_marks.append(st)
            self.daily_rows.append({
                "date": d,
                "cash": st.cash,
                "nav": st.nav,
                "realized_pnl": st.realized_pnl,
                "fees_paid": st.fees_paid,
                "n_positions": len(st.positions),
                "step": step_idx,
            })

        self.ledger.bump_held_steps()
        final_prices = self.mark_prices(
            daily_marks[-1].ts if daily_marks else fill_date,
            list(self.ledger.positions.keys()),
        )
        ledger_after = self.ledger.state(
            final_prices,
            daily_marks[-1].ts if daily_marks else fill_date,
        )

        result = StepResult(
            step=step_idx,
            decision_date=decision_date,
            fill_date=fill_date,
            orders_in=parsed,
            validated=validated,
            fills=fills,
            violations=violations,
            ledger_after=ledger_after,
            daily_marks=daily_marks,
        )
        self.step_results.append(result)
        return result

    def ledger_daily_df(self) -> pd.DataFrame:
        if not self.daily_rows:
            return pd.DataFrame(
                columns=["date", "cash", "nav", "realized_pnl", "fees_paid", "n_positions", "step"]
            )
        return pd.DataFrame(self.daily_rows)

    def final_state(self) -> LedgerState:
        if self.daily_rows:
            last = self.daily_rows[-1]["date"]
            prices = self.mark_prices(last, list(self.ledger.positions.keys()))
            return self.ledger.state(prices, last)
        return self.ledger.state({}, self.decision_dates[0])
