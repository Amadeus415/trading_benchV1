"""Point-in-time data store with enforced no-lookahead guarantees.

The runner owns `as_of`. Callers cannot pass it; every public method filters
to data available at that date and raises FutureDataError if a guard fails.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Iterable

import pandas as pd


class FutureDataError(RuntimeError):
    """Raised when a query would return data after the store's as_of date."""


@dataclass(frozen=True)
class Snapshot:
    """Immutable frozen market data package."""

    snapshot_id: str
    path: Path
    prices: pd.DataFrame
    corporate_actions: pd.DataFrame
    news: pd.DataFrame
    universe: pd.DataFrame
    manifest: dict


def _parse_date(value) -> date:
    if isinstance(value, date) and not isinstance(value, datetime):
        return value
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, pd.Timestamp):
        return value.date()
    return pd.Timestamp(value).date()


def load_snapshot(snapshot_dir: str | Path) -> Snapshot:
    """Load a frozen snapshot from disk."""
    path = Path(snapshot_dir)
    if not path.exists():
        raise FileNotFoundError(f"Snapshot not found: {path}")

    prices = pd.read_parquet(path / "prices.parquet")
    corporate_actions = pd.read_parquet(path / "corporate_actions.parquet")
    news = pd.read_parquet(path / "news.parquet")
    universe = pd.read_parquet(path / "universe.parquet")

    with open(path / "manifest.json") as f:
        manifest = json.load(f)

    # Normalize date columns
    prices["date"] = pd.to_datetime(prices["date"]).dt.date
    if len(corporate_actions):
        corporate_actions["ex_date"] = pd.to_datetime(corporate_actions["ex_date"]).dt.date
    if len(news):
        news["published_at"] = pd.to_datetime(news["published_at"], utc=True)
    if len(universe):
        universe["listed_from"] = pd.to_datetime(universe["listed_from"]).dt.date
        universe["eligible_from"] = pd.to_datetime(universe["eligible_from"]).dt.date
        if "delisted_at" in universe.columns:
            universe["delisted_at"] = universe["delisted_at"].apply(
                lambda x: None if pd.isna(x) else _parse_date(x)
            )

    snapshot_id = manifest.get("snapshot_id", path.name)
    return Snapshot(
        snapshot_id=snapshot_id,
        path=path,
        prices=prices,
        corporate_actions=corporate_actions,
        news=news,
        universe=universe,
        manifest=manifest,
    )


class PointInTimeStore:
    """Read-only view of a snapshot truncated at a fixed as_of date.

    `as_of` is set at construction by the runner. Public query methods never
    accept an as_of argument, so the model cannot influence the time cursor.
    """

    def __init__(self, snapshot: Snapshot, as_of: date):
        self._snapshot = snapshot
        self._as_of = _parse_date(as_of)

    @property
    def as_of(self) -> date:
        return self._as_of

    @property
    def snapshot_id(self) -> str:
        return self._snapshot.snapshot_id

    def _guard(self, df: pd.DataFrame, col: str) -> pd.DataFrame:
        if df.empty:
            return df
        series = df[col]
        if col == "published_at":
            max_val = series.max()
            as_of_ts = pd.Timestamp(self._as_of, tz="UTC") + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
            if pd.isna(max_val):
                return df
            if max_val > as_of_ts:
                raise FutureDataError(
                    f"Guard failed on {col}: max={max_val} > as_of={as_of_ts}"
                )
            return df
        max_date = series.max()
        if isinstance(max_date, datetime):
            max_date = max_date.date()
        elif isinstance(max_date, pd.Timestamp):
            max_date = max_date.date()
        if max_date is not None and not pd.isna(max_date) and max_date > self._as_of:
            raise FutureDataError(
                f"Guard failed on {col}: max={max_date} > as_of={self._as_of}"
            )
        return df

    def prices(
        self,
        symbols: Iterable[str] | None = None,
        lookback_days: int | None = None,
    ) -> pd.DataFrame:
        df = self._snapshot.prices
        mask = df["date"] <= self._as_of
        if symbols is not None:
            symbols = list(symbols)
            mask &= df["symbol"].isin(symbols)
        if lookback_days is not None:
            start = self._as_of - timedelta(days=lookback_days)
            mask &= df["date"] >= start
        out = df.loc[mask].copy()
        return self._guard(out, "date")

    def price_on(self, symbol: str, on: date | None = None) -> float | None:
        """Close price on or before `on` (default as_of)."""
        target = _parse_date(on) if on is not None else self._as_of
        if target > self._as_of:
            raise FutureDataError(f"price_on requested {target} after as_of {self._as_of}")
        df = self._snapshot.prices
        sub = df[(df["symbol"] == symbol) & (df["date"] <= target)]
        if sub.empty:
            return None
        return float(sub.sort_values("date").iloc[-1]["close"])

    def open_on(self, symbol: str, on: date) -> float | None:
        """Open price on a specific date (must be <= as_of)."""
        target = _parse_date(on)
        if target > self._as_of:
            raise FutureDataError(f"open_on requested {target} after as_of {self._as_of}")
        df = self._snapshot.prices
        sub = df[(df["symbol"] == symbol) & (df["date"] == target)]
        if sub.empty:
            return None
        return float(sub.iloc[0]["open"])

    def bar_on(self, symbol: str, on: date) -> dict | None:
        target = _parse_date(on)
        if target > self._as_of:
            raise FutureDataError(f"bar_on requested {target} after as_of {self._as_of}")
        df = self._snapshot.prices
        sub = df[(df["symbol"] == symbol) & (df["date"] == target)]
        if sub.empty:
            return None
        row = sub.iloc[0]
        return {
            "date": target,
            "symbol": symbol,
            "open": float(row["open"]),
            "high": float(row["high"]),
            "low": float(row["low"]),
            "close": float(row["close"]),
            "volume": float(row["volume"]),
        }

    def news(
        self,
        symbols: Iterable[str] | None = None,
        lookback_days: int = 14,
        limit: int = 40,
    ) -> pd.DataFrame:
        df = self._snapshot.news
        if df.empty:
            return df.copy()

        as_of_end = pd.Timestamp(self._as_of, tz="UTC") + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
        start = pd.Timestamp(self._as_of - timedelta(days=lookback_days), tz="UTC")
        mask = (df["published_at"] <= as_of_end) & (df["published_at"] >= start)

        if symbols is not None:
            symbols = set(symbols)
            # include macro (null symbol) + requested symbols
            mask &= df["symbol"].isna() | df["symbol"].isin(symbols)

        out = df.loc[mask].sort_values("published_at", ascending=False).head(limit).copy()
        return self._guard(out, "published_at")

    def universe(self) -> pd.DataFrame:
        """Universe as of as_of — listed, eligible, not yet delisted."""
        df = self._snapshot.universe
        rows = []
        for _, row in df.iterrows():
            listed = _parse_date(row["listed_from"])
            eligible = _parse_date(row["eligible_from"])
            delisted = row.get("delisted_at")
            if delisted is not None and not (isinstance(delisted, float) and pd.isna(delisted)):
                delisted = _parse_date(delisted)
            else:
                delisted = None
            if listed <= self._as_of and eligible <= self._as_of:
                if delisted is None or delisted > self._as_of:
                    rows.append(row)
        out = pd.DataFrame(rows) if rows else df.iloc[0:0].copy()
        return out.reset_index(drop=True)

    def corporate_actions_on(self, on: date) -> pd.DataFrame:
        """Corporate actions with ex_date == on (must be <= as_of)."""
        target = _parse_date(on)
        if target > self._as_of:
            raise FutureDataError(
                f"corporate_actions_on requested {target} after as_of {self._as_of}"
            )
        df = self._snapshot.corporate_actions
        if df.empty:
            return df.copy()
        out = df[df["ex_date"] == target].copy()
        return self._guard(out, "ex_date")

    def trading_dates(self, start: date | None = None, end: date | None = None) -> list[date]:
        """Sorted unique trading dates available up to as_of."""
        df = self._snapshot.prices
        dates = sorted(set(df["date"]))
        dates = [d for d in dates if d <= self._as_of]
        if start is not None:
            dates = [d for d in dates if d >= _parse_date(start)]
        if end is not None:
            dates = [d for d in dates if d <= _parse_date(end)]
        return dates

    def next_trading_date(self, after: date) -> date | None:
        """Next trading date strictly after `after`, if available in full snapshot.

        NOTE: uses full snapshot prices (not as_of-filtered) because the
        simulator store has as_of=t+1 when asking for the next session.
        For agent stores, this still only returns dates <= as_of.
        """
        after = _parse_date(after)
        dates = sorted(set(self._snapshot.prices["date"]))
        candidates = [d for d in dates if d > after and d <= self._as_of]
        return candidates[0] if candidates else None

    def all_trading_dates(self) -> list[date]:
        """All trading dates in the underlying snapshot (unfiltered).

        Used by the runner only to plan the calendar — never passed to agents.
        """
        return sorted(set(self._snapshot.prices["date"]))
