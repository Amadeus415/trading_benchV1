"""Corporate-action application for the simulator.

Splits and cash dividends are applied on the ex-date against the ledger.
Prices in the snapshot are *unadjusted*; the simulator is the only place
corporate actions enter the P&L path (avoids adj_close lookahead).
"""

from __future__ import annotations

from datetime import date
from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from tradingbench.data.store import PointInTimeStore, Snapshot
    from tradingbench.sim.ledger import Ledger


def actions_on(snapshot: "Snapshot", on: date) -> pd.DataFrame:
    """Return corporate actions with ex_date == on."""
    df = snapshot.corporate_actions
    if df is None or df.empty:
        return pd.DataFrame(
            columns=["ex_date", "symbol", "action_type", "ratio", "amount"]
        )
    return df[df["ex_date"] == on].copy()


def apply_corporate_actions(
    ledger: "Ledger",
    store: "PointInTimeStore",
    on: date,
) -> list[dict]:
    """Apply all ex-date actions for `on` to the ledger. Returns event log."""
    events: list[dict] = []
    actions = store.corporate_actions_on(on)
    if actions.empty:
        return events

    for _, row in actions.iterrows():
        sym = str(row["symbol"])
        atype = str(row["action_type"])
        if atype == "split":
            ratio = row.get("ratio")
            if ratio is not None and not pd.isna(ratio) and float(ratio) > 0:
                ledger.apply_split(sym, float(ratio))
                events.append(
                    {
                        "ex_date": on.isoformat(),
                        "symbol": sym,
                        "action_type": "split",
                        "ratio": float(ratio),
                    }
                )
        elif atype == "cash_dividend":
            amount = row.get("amount")
            if amount is not None and not pd.isna(amount):
                ledger.apply_dividend(sym, float(amount))
                events.append(
                    {
                        "ex_date": on.isoformat(),
                        "symbol": sym,
                        "action_type": "cash_dividend",
                        "amount": float(amount),
                    }
                )
    return events
