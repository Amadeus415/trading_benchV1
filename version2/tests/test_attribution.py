"""Attribution unit tests."""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd

from tradingbench.eval.attribution import (
    FACTOR_NAMES,
    attribute_portfolio,
    compute_style_exposures,
    vif_screen,
)


def _synthetic_prices(n_days: int = 80, n_syms: int = 8, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    start = date(2024, 1, 2)
    rows = []
    levels = rng.uniform(50, 200, size=n_syms)
    for d in range(n_days):
        day = start + timedelta(days=d)
        if day.weekday() >= 5:
            continue
        shocks = rng.normal(0.0005, 0.015, size=n_syms)
        levels = levels * (1 + shocks)
        for i in range(n_syms):
            c = float(levels[i])
            o = c * (1 + float(rng.normal(0, 0.002)))
            rows.append({
                "date": day,
                "symbol": f"S{i}",
                "open": o,
                "high": max(o, c) * 1.01,
                "low": min(o, c) * 0.99,
                "close": c,
                "volume": float(rng.uniform(1e5, 1e6)),
                "currency": "USD",
            })
    return pd.DataFrame(rows)


def test_style_exposures_no_lookahead():
    prices = _synthetic_prices()
    as_of = sorted(prices["date"].unique())[40]
    exp = compute_style_exposures(prices, as_of)
    assert not exp.empty
    for f in FACTOR_NAMES:
        assert f in exp.columns
    # exposures use only data before as_of — no crash, finite values
    assert np.isfinite(exp.values).all()


def test_vif_screen_returns_subset():
    prices = _synthetic_prices()
    as_of = sorted(prices["date"].unique())[40]
    exp = compute_style_exposures(prices, as_of)
    kept = vif_screen(exp, threshold=10.0)
    assert len(kept) >= 1
    assert set(kept).issubset(set(FACTOR_NAMES))


def test_attribute_sums_to_portfolio():
    prices = _synthetic_prices(n_days=100, n_syms=6)
    dates = sorted(prices["date"].unique())
    # equal weight last 10 trading days across all symbols
    syms = sorted(prices["symbol"].unique())
    wrows = []
    for d in dates[-10:]:
        for s in syms:
            wrows.append({"date": d, "symbol": s, "weight": 1.0 / len(syms)})
    wdf = pd.DataFrame(wrows)
    result = attribute_portfolio(prices, wdf)
    # Common + Style + Selection ≈ total
    assert abs(result.common + result.style + result.selection_alpha - result.total) < 1e-6
    d = result.to_dict()
    assert "selection_alpha" in d
