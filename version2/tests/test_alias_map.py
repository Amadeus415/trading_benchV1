"""Alias map + 2×2 mask factorial tests."""

from __future__ import annotations

from datetime import date

from tradingbench.eval.contamination.alias_map import (
    AliasMap,
    memory_only_observation,
)


def test_aliases_stable_within_seed_reshuffled_across():
    syms = ["AAPL", "MSFT", "NVDA"]
    a1 = AliasMap.build(syms, seed=1, mask="blinded")
    a1b = AliasMap.build(syms, seed=1, mask="blinded")
    a2 = AliasMap.build(syms, seed=2, mask="blinded")
    assert a1.ticker_real_to_alias == a1b.ticker_real_to_alias
    assert a1.ticker_real_to_alias != a2.ticker_real_to_alias
    # round-trip
    for s in syms:
        alias = a1.mask_ticker(s)
        assert alias.startswith("ASSET_")
        assert a1.unmask_ticker(alias) == s


def test_mask_factorial_ticker_vs_date():
    start = date(2025, 1, 6)
    am_stock = AliasMap.build(["AAPL"], seed=0, episode_start=start, mask="stock_blind")
    am_date = AliasMap.build(["AAPL"], seed=0, episode_start=start, mask="date_blind")
    am_full = AliasMap.build(["AAPL"], seed=0, episode_start=start, mask="blinded")
    am_bright = AliasMap.build(["AAPL"], seed=0, episode_start=start, mask="bright")

    assert am_stock.mask_ticker("AAPL").startswith("ASSET_")
    assert am_stock.mask_date(start) == "2025-01-06"  # real dates

    assert am_date.mask_ticker("AAPL") == "AAPL"  # real tickers
    assert am_date.mask_date(start).startswith("day_")

    assert am_full.mask_ticker("AAPL").startswith("ASSET_")
    assert am_full.mask_date(start).startswith("day_")

    assert am_bright.mask_ticker("AAPL") == "AAPL"
    assert am_bright.mask_date(start) == "2025-01-06"


def test_apply_to_observation_strips_news_when_masked():
    obs = {
        "as_of": "2025-01-06",
        "portfolio": {"nav": 1000, "cash": 1000, "positions": []},
        "market": [
            {
                "symbol": "AAPL",
                "asset_class": "equity",
                "sector": "Tech",
                "last": 180.0,
                "ret_1w": 0.01,
                "ret_1m": 0.02,
                "ret_3m": 0.05,
                "vol_20d": 0.2,
                "drawdown_from_52w_high": -0.1,
            }
        ],
        "news": [{"published_at": "2025-01-05T12:00:00Z", "symbol": "AAPL", "headline": "x", "summary": "y"}],
        "prior_decisions": [],
        "last_step_violations": [],
        "rules": {},
    }
    am = AliasMap.build(["AAPL"], seed=7, episode_start=date(2025, 1, 6), mask="blinded")
    out = am.apply_to_observation(obs)
    assert out["news"] == []
    assert out["market"][0]["symbol"].startswith("ASSET_")
    # asset_class stays unmasked (documented exception)
    assert out["market"][0]["asset_class"] == "equity"
    # numeric features preserved
    assert out["market"][0]["ret_3m"] == 0.05


def test_memory_only_strips_factors():
    obs = {
        "market": [
            {
                "symbol": "AAPL",
                "asset_class": "equity",
                "last": 180.0,
                "ret_1w": 0.01,
                "ret_1m": 0.02,
                "ret_3m": 0.05,
                "vol_20d": 0.2,
                "drawdown_from_52w_high": -0.1,
            }
        ],
        "portfolio": {
            "nav": 1000,
            "cash": 500,
            "positions": [
                {
                    "symbol": "AAPL",
                    "qty": 1,
                    "weight": 0.5,
                    "market_value": 500,
                    "held_steps": 1,
                    "avg_cost": 170,
                    "last": 180,
                    "unrealized_pnl_pct": 0.05,
                }
            ],
        },
        "news": [{"headline": "x"}],
    }
    out = memory_only_observation(obs)
    assert out["decision_mode"] == "memory_only"
    assert out["market"][0]["ret_3m"] is None
    assert out["market"][0]["last"] is None
    assert out["news"] == []
    assert out["portfolio"]["positions"][0]["symbol"] == "AAPL"
