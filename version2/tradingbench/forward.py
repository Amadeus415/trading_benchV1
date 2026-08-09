"""Forward / live track scaffolding (MVP_SPEC §2.5, Phase 0).

The forward track produces the *headline, uncontaminated* result.
Contamination is impossible by construction (ForecastBench principle).
Sample size grows only with calendar time.

This module stands up a live paper portfolio that appends to the same
`ledger_daily.parquet` schema used by backtests. Start it on day one so
history accumulates while the backtest harness is built.

Usage:
    tradingbench forward-init --out forward/live_v1
    tradingbench forward-step --portfolio forward/live_v1 --snapshot snapshots/<id>
"""

from __future__ import annotations

import json
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from tradingbench.agent.baselines import get_baseline, seed_rng
from tradingbench.agent.observation import build_observation, observation_for_prompt
from tradingbench.data.store import Snapshot, load_snapshot
from tradingbench.sim.engine import EpisodeEngine
from tradingbench.sim.validate import Order


def init_forward_portfolio(
    out_dir: str | Path,
    *,
    starting_cash: float = 1000.0,
    models: list[str] | None = None,
) -> Path:
    """Create forward-track portfolio directory with empty ledgers."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    models = models or ["buy_and_hold"]
    meta = {
        "track": "forward",
        "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "starting_cash": starting_cash,
        "models": models,
        "cadence": "weekly",
        "note": (
            "Forward track: one market path, zero contamination by construction. "
            "Never pool with backtest results."
        ),
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    for m in models:
        pdir = out / m
        pdir.mkdir(exist_ok=True)
        # empty ledger with schema
        df = pd.DataFrame(
            columns=["date", "cash", "nav", "realized_pnl", "fees_paid", "n_positions", "step"]
        )
        df.to_parquet(pdir / "ledger_daily.parquet", index=False)
        (pdir / "state.json").write_text(
            json.dumps(
                {
                    "model": m,
                    "step": 0,
                    "cash": starting_cash,
                    "positions": {},
                    "last_decision_date": None,
                },
                indent=2,
            )
        )
    return out


def forward_step(
    portfolio_dir: str | Path,
    snapshot: Snapshot | str | Path,
    *,
    model: str = "buy_and_hold",
    as_of: date | None = None,
    seed: int = 1,
) -> dict[str, Any]:
    """Append one weekly decision for a forward portfolio.

    Uses the latest available date in the snapshot as as_of when not provided.
    """
    root = Path(portfolio_dir)
    if isinstance(snapshot, (str, Path)):
        snapshot = load_snapshot(snapshot)

    pdir = root / model
    if not pdir.exists():
        raise FileNotFoundError(f"No forward portfolio for model {model} under {root}")

    state = json.loads((pdir / "state.json").read_text())
    step = int(state.get("step", 0))

    dates = sorted(set(snapshot.prices["date"]))
    if as_of is None:
        as_of = dates[-1]
    # Need room for fill at next session — if as_of is last bar, use second-to-last as decision
    if as_of >= dates[-1] and len(dates) >= 2:
        as_of = dates[-2]

    engine = EpisodeEngine(
        snapshot=snapshot,
        starting_cash=float(state.get("cash", 1000.0)),
        steps=1,
        start_date=as_of,
    )
    # Restore positions if any (simplified: restart from cash each step for v0 scaffold;
    # full state restore is a v1.1 hardening item when live data feed lands.)
    store = engine.agent_store(0)
    obs = build_observation(
        store=store,
        ledger=engine.ledger,
        step=step,
        total_steps=step + 1,
        prior_decisions=[],
        last_step_violations=[],
        mask="bright",
        decision_mode="standard",
        episode_seed=seed,
        episode_start=as_of,
    )
    baseline_fn = get_baseline(model) if model in (
        "buy_and_hold", "sixty_forty", "random_agent", "momentum_3m",
        "equal_weight_rebal", "mean_reversion_3m", "ma_crossover", "momentum_lite",
    ) else get_baseline("buy_and_hold")
    decision = baseline_fn(observation_for_prompt(obs), seed_rng(seed, step))
    orders = [Order.from_dict(o) for o in decision.get("orders") or []]
    result = engine.step(0, orders)

    ledger_path = pdir / "ledger_daily.parquet"
    existing = pd.read_parquet(ledger_path) if ledger_path.exists() else pd.DataFrame()
    new_rows = engine.ledger_daily_df()
    if not new_rows.empty:
        new_rows = new_rows.copy()
        new_rows["step"] = step
        combined = pd.concat([existing, new_rows], ignore_index=True)
        combined.to_parquet(ledger_path, index=False)

    state["step"] = step + 1
    state["last_decision_date"] = result.decision_date.isoformat()
    state["cash"] = result.ledger_after.cash
    state["nav"] = result.ledger_after.nav
    state["positions"] = {
        s: {"qty": p.qty, "avg_cost": p.avg_cost}
        for s, p in result.ledger_after.positions.items()
    }
    (pdir / "state.json").write_text(json.dumps(state, indent=2, default=str))

    step_dir = pdir / "steps" / f"{step:04d}"
    step_dir.mkdir(parents=True, exist_ok=True)
    (step_dir / "decision.json").write_text(json.dumps(decision, indent=2))
    (step_dir / "ledger_after.json").write_text(
        json.dumps(result.ledger_after.to_dict(), indent=2, default=str)
    )

    return {
        "model": model,
        "step": step,
        "decision_date": result.decision_date.isoformat(),
        "fill_date": result.fill_date.isoformat(),
        "nav": result.ledger_after.nav,
        "path": str(pdir),
    }
