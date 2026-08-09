"""Point-in-time store: no-lookahead property tests + mutation checks."""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import pytest

from tradingbench.data.build_snapshot import build_snapshot
from tradingbench.data.store import FutureDataError, PointInTimeStore, load_snapshot


@pytest.fixture(scope="module")
def snapshot(tmp_path_factory):
    root = tmp_path_factory.mktemp("snap")
    path = build_snapshot(root / "_building", seed=7)
    return load_snapshot(path)


def test_prices_never_after_as_of(snapshot):
    dates = sorted(set(snapshot.prices["date"]))
    # sample ~50 dates across the range
    sample = dates[:: max(1, len(dates) // 50)][:50]
    for as_of in sample:
        store = PointInTimeStore(snapshot, as_of)
        px = store.prices()
        if not px.empty:
            assert px["date"].max() <= as_of


def test_news_never_after_as_of(snapshot):
    dates = sorted(set(snapshot.prices["date"]))
    sample = dates[:: max(1, len(dates) // 40)][:40]
    for as_of in sample:
        store = PointInTimeStore(snapshot, as_of)
        news = store.news(limit=100)
        if not news.empty:
            max_ts = news["published_at"].max()
            as_of_end = pd.Timestamp(as_of, tz="UTC") + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
            assert max_ts <= as_of_end


def test_universe_excludes_delisted(snapshot):
    # FAKECO delisted 2024-03-15
    before = PointInTimeStore(snapshot, date(2024, 3, 1)).universe()
    after = PointInTimeStore(snapshot, date(2024, 3, 16)).universe()
    assert "FAKECO" in before["symbol"].values
    assert "FAKECO" not in after["symbol"].values


def test_future_price_on_raises(snapshot):
    store = PointInTimeStore(snapshot, date(2024, 6, 1))
    with pytest.raises(FutureDataError):
        store.price_on("AAPL", on=date(2025, 1, 1))


def test_guard_detects_injected_future_row(snapshot):
    """Mutation-style: if we inject a future row without filtering, guard fires."""
    store = PointInTimeStore(snapshot, date(2024, 1, 15))
    # Call internal path: build a df with a future date and run guard
    bad = pd.DataFrame({"date": [date(2025, 1, 1)], "close": [1.0]})
    with pytest.raises(FutureDataError):
        store._guard(bad, "date")


def test_property_random_as_of_all_methods(snapshot):
    """Property-ish: many random as_of dates, all public methods clean."""
    import numpy as np

    rng = np.random.default_rng(0)
    dates = sorted(set(snapshot.prices["date"]))
    # skip first 60 days so lookbacks have data
    pool = dates[60:]
    sample = [pool[i] for i in rng.choice(len(pool), size=min(80, len(pool)), replace=False)]
    for as_of in sample:
        store = PointInTimeStore(snapshot, as_of)
        px = store.prices(lookback_days=30)
        if not px.empty:
            assert px["date"].max() <= as_of
        news = store.news(lookback_days=14, limit=20)
        if not news.empty:
            as_of_end = pd.Timestamp(as_of, tz="UTC") + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
            assert news["published_at"].max() <= as_of_end
        uni = store.universe()
        assert len(uni) > 0
        for _, row in uni.iterrows():
            assert row["listed_from"] <= as_of
            assert row["eligible_from"] <= as_of
            if row["delisted_at"] is not None and not (isinstance(row["delisted_at"], float) and pd.isna(row["delisted_at"])):
                assert row["delisted_at"] > as_of
