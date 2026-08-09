"""Engine + baseline integration tests."""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

from tradingbench.agent.baselines import buy_and_hold, seed_rng
from tradingbench.agent.observation import build_observation
from tradingbench.data.build_snapshot import build_snapshot
from tradingbench.data.store import load_snapshot
from tradingbench.sim.engine import EpisodeEngine
from tradingbench.sim.validate import Order


@pytest.fixture(scope="module")
def snapshot(tmp_path_factory):
    root = tmp_path_factory.mktemp("snap")
    path = build_snapshot(root / "_building", seed=42)
    return load_snapshot(path)


def test_buy_and_hold_episode_completes(snapshot):
    engine = EpisodeEngine(
        snapshot=snapshot,
        starting_cash=1000.0,
        steps=4,
        start_date=date(2025, 1, 6),
    )
    prior = []
    for step in range(4):
        store = engine.agent_store(step)
        obs = build_observation(
            store, engine.ledger, step, 4, prior, engine.last_step_violations()
        )
        decision = buy_and_hold(obs)
        orders = [Order.from_dict(o) for o in decision["orders"]]
        result = engine.step(step, orders)
        assert result.ledger_after.nav > 0
        prior.append({
            "step": step,
            "thesis_summary": decision["portfolio_view"][:100],
            "orders": [f"{o['side']} {o['symbol']}" for o in decision["orders"]],
        })

    df = engine.ledger_daily_df()
    assert not df.empty
    assert df["nav"].iloc[-1] > 0
    # After step 0, buy-and-hold should have positions
    assert len(engine.ledger.positions) > 0


def test_buy_and_hold_matches_manual_nav_within_tolerance(snapshot):
    """After opening BH orders, cash + marked positions equals ledger NAV."""
    engine = EpisodeEngine(
        snapshot=snapshot,
        starting_cash=1000.0,
        steps=2,
        start_date=date(2025, 1, 6),
    )
    store = engine.agent_store(0)
    obs = build_observation(store, engine.ledger, 0, 2, [], [])
    decision = buy_and_hold(obs)
    orders = [Order.from_dict(o) for o in decision["orders"]]
    assert orders, "expected BH to place opening orders"
    result = engine.step(0, orders)
    st = result.ledger_after
    # Reconstruct NAV from cash + qty * mark prices used in final state
    mark_store = engine.sim_store(result.daily_marks[-1].ts if result.daily_marks else result.fill_date)
    pos_value = 0.0
    for s, p in st.positions.items():
        px = mark_store.price_on(s)
        assert px is not None
        pos_value += p.qty * px
    assert abs(st.nav - (st.cash + pos_value)) < 1e-6
    # Soft check: NAV near 1000 after fees/slippage
    assert 900 < st.nav < 1100


def test_position_cap_rejects(snapshot):
    engine = EpisodeEngine(
        snapshot=snapshot,
        starting_cash=1000.0,
        steps=2,
        start_date=date(2025, 1, 6),
        max_position_weight=0.25,
    )
    # Attempt to buy 50% of NAV in one name
    orders = [Order(symbol="AAPL", side="buy", notional_usd=500.0)]
    result = engine.step(0, orders)
    rejected = [v for v in result.validated if not v.accepted]
    assert any(v.violation and v.violation.code == "POSITION_CAP" for v in rejected)


def test_replay_deterministic(snapshot):
    """Same baseline + seed → identical daily NAV series."""

    def run_once():
        engine = EpisodeEngine(
            snapshot=snapshot, starting_cash=1000.0, steps=3, start_date=date(2025, 1, 6)
        )
        prior = []
        for step in range(3):
            store = engine.agent_store(step)
            obs = build_observation(store, engine.ledger, step, 3, prior, engine.last_step_violations())
            d = buy_and_hold(obs)
            engine.step(step, [Order.from_dict(o) for o in d["orders"]])
            prior.append({"step": step, "thesis_summary": d["portfolio_view"][:80], "orders": []})
        return engine.ledger_daily_df()["nav"].tolist()

    a = run_once()
    b = run_once()
    assert a == b
