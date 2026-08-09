"""Full baseline suite smoke tests."""

from __future__ import annotations

import numpy as np

from tradingbench.agent.baselines import BASELINES, get_baseline, seed_rng


def _obs(step: int = 0) -> dict:
    market = []
    for i, (sym, r3) in enumerate(
        [
            ("AAPL", 0.12),
            ("MSFT", 0.08),
            ("NVDA", 0.20),
            ("GOOGL", -0.05),
            ("AMZN", 0.03),
            ("JPM", -0.10),
            ("XOM", 0.01),
            ("JNJ", -0.02),
        ]
    ):
        market.append({
            "symbol": sym,
            "asset_class": "equity",
            "sector": "Tech",
            "last": 100 + i,
            "ret_1w": 0.01 * (i - 3),
            "ret_1m": 0.02 * (i - 3),
            "ret_3m": r3,
            "vol_20d": 0.25,
            "drawdown_from_52w_high": -0.1,
        })
    return {
        "as_of": "2025-01-06",
        "episode": {"step": step, "total_steps": 12, "cadence": "weekly"},
        "portfolio": {"nav": 1000.0, "cash": 1000.0 if step == 0 else 50.0, "positions": []},
        "market": market,
        "news": [],
        "prior_decisions": [],
        "last_step_violations": [],
        "rules": {
            "max_position_weight": 0.25,
            "min_order_usd": 10.0,
            "shorting": False,
            "leverage": False,
        },
    }


def test_all_baselines_registered():
    required = [
        "buy_and_hold",
        "equal_weight_rebal",
        "momentum_3m",
        "mean_reversion_3m",
        "ma_crossover",
        "sixty_forty",
        "random_agent",
    ]
    for name in required:
        assert name in BASELINES


def test_each_baseline_returns_valid_decision():
    for name, fn in BASELINES.items():
        rng = seed_rng(1, 0)
        d = fn(_obs(0), rng)
        assert "orders" in d
        assert "portfolio_view" in d
        for o in d["orders"]:
            assert o["side"] in ("buy", "sell")
            assert o["notional_usd"] >= 10
            assert "symbol" in o


def test_momentum_3m_picks_top_half():
    d = get_baseline("momentum_3m")(_obs(0), seed_rng(1, 0))
    bought = {o["symbol"] for o in d["orders"] if o["side"] == "buy"}
    # Top half by ret_3m: NVDA 0.20, AAPL 0.12, MSFT 0.08, AMZN 0.03
    assert "NVDA" in bought or "AAPL" in bought
    # Bottom names should not be bought on step 0 with empty portfolio
    assert "JPM" not in bought


def test_buy_and_hold_holds_after_step0():
    d0 = get_baseline("buy_and_hold")(_obs(0), seed_rng(1, 0))
    d1 = get_baseline("buy_and_hold")(_obs(1), seed_rng(1, 1))
    assert len(d0["orders"]) > 0
    assert d1["orders"] == []
