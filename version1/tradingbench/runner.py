"""Episode + campaign orchestration."""

from __future__ import annotations

import json
import subprocess
from copy import deepcopy
from dataclasses import asdict
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from tradingbench.agent.baselines import get_baseline, seed_rng
from tradingbench.agent.clients.base import get_client
from tradingbench.agent.observation import build_observation, observation_for_prompt
from tradingbench.agent.parser import parse_decision
from tradingbench.agent.prompt import (
    PROMPT_VERSION,
    SYSTEM_PROMPT,
    prompt_hash,
    render_repair_prompt,
    render_user_prompt,
)
from tradingbench.data.store import Snapshot, load_snapshot
from tradingbench.eval.contamination.leak_detect import scan_for_leaks
from tradingbench.eval.contamination.probe import run_memory_probe
from tradingbench.eval.metrics import compute_episode_metrics
from tradingbench.report.run_report import write_run_report
from tradingbench.sim.engine import EpisodeEngine
from tradingbench.sim.validate import Order


def _git_sha() -> str:
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL
            )
            .decode()
            .strip()
        )
    except Exception:
        return "unknown"


def _json_dump(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=str)


def _parse_date(s: str | date) -> date:
    if isinstance(s, date) and not isinstance(s, datetime):
        return s
    return date.fromisoformat(str(s))


def load_config(path: str | Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def run_id_for(
    campaign: str,
    model: str,
    window: str,
    seed: int,
    mode: str,
) -> str:
    return f"{campaign}_{model}_{window}_{seed}_{mode}"


def run_episode(
    snapshot: Snapshot,
    *,
    model_cfg: dict,
    window: dict,
    seed: int,
    mode: str = "standard",
    starting_cash: float = 1000.0,
    steps: int = 12,
    max_position_weight: float = 0.25,
    runs_dir: Path,
    campaign: str = "v1",
    prompt_version: str = PROMPT_VERSION,
    run_probe: bool = False,
    response_cache: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Run one full episode; write artifacts under runs/{run_id}/."""
    model_id = model_cfg["id"]
    window_id = window["id"]
    start = _parse_date(window["start"])
    rid = run_id_for(campaign, model_id, window_id, seed, mode)
    out = Path(runs_dir) / rid
    out.mkdir(parents=True, exist_ok=True)
    steps_dir = out / "steps"
    steps_dir.mkdir(exist_ok=True)

    engine = EpisodeEngine(
        snapshot=snapshot,
        starting_cash=starting_cash,
        max_position_weight=max_position_weight,
        steps=steps,
        start_date=start,
    )

    provider = model_cfg.get("provider", "baseline")
    is_baseline = provider in ("baseline", "mock", "local") or model_id in (
        "buy_and_hold",
        "sixty_forty",
        "random_agent",
        "momentum_lite",
    )

    client = None
    baseline_fn = None
    if is_baseline:
        bname = model_cfg.get("name") or model_id
        if bname in ("buy_and_hold", "sixty_forty", "random_agent", "momentum_lite"):
            baseline_fn = get_baseline(bname)
        else:
            client = get_client("mock", bname, baseline=bname)
    else:
        client = get_client(
            provider,
            model_cfg["name"],
            temperature=model_cfg.get("temperature", 0.7),
        )

    prior_decisions: list[dict] = []
    step_artifacts: list[dict] = []
    all_violations: list[dict] = []
    total_in_tok = 0
    total_out_tok = 0
    texts_for_leak: list[str] = []

    # Optional memory probe (pre-episode)
    probe_result = None
    if run_probe:
        probe_symbols = ["NVDA", "AAPL", "BTC"]
        probe_result = run_memory_probe(
            client if not is_baseline else None,
            start,
            probe_symbols,
            is_baseline=is_baseline,
        )
        _json_dump(out / "probe.json", probe_result)

    for step_idx in range(steps):
        store = engine.agent_store(step_idx)
        obs = build_observation(
            store=store,
            ledger=engine.ledger,
            step=step_idx,
            total_steps=steps,
            prior_decisions=prior_decisions,
            last_step_violations=engine.last_step_violations(),
            mode=mode if mode != "named_control" else "named_control",
            max_position_weight=max_position_weight,
        )
        # named_control: strip news
        if mode == "named_control":
            obs = dict(obs)
            obs["news"] = []
            obs["mode"] = "named_control"

        blind_inv = obs.get("_blind_map_inv")
        user_prompt = render_user_prompt(obs)
        system = SYSTEM_PROMPT

        step_dir = steps_dir / f"{step_idx:02d}"
        step_dir.mkdir(exist_ok=True)
        _json_dump(step_dir / "observation.json", observation_for_prompt(obs))
        (step_dir / "prompt.txt").write_text(f"SYSTEM:\n{system}\n\nUSER:\n{user_prompt}")

        raw_text = None
        cache_key = f"{rid}:{step_idx}"
        if response_cache is not None and cache_key in response_cache:
            raw_text = response_cache[cache_key]
        elif baseline_fn is not None:
            rng = seed_rng(seed, step_idx)
            if model_id == "random_agent" or (model_cfg.get("name") == "random_agent"):
                decision_dict = baseline_fn(observation_for_prompt(obs), rng)
            else:
                decision_dict = baseline_fn(observation_for_prompt(obs), rng)
            raw_text = json.dumps(decision_dict)
        else:
            assert client is not None
            if hasattr(client, "set_episode_context"):
                client.set_episode_context(seed, step_idx)
            comp = client.complete(
                system,
                user_prompt,
                seed=seed,
                temperature=float(model_cfg.get("temperature", 0.7)),
                max_tokens=int(model_cfg.get("max_tokens", 2000)),
                response_format={"type": "json_object"},
            )
            raw_text = comp.text
            total_in_tok += comp.input_tokens
            total_out_tok += comp.output_tokens

        (step_dir / "response_raw.txt").write_text(raw_text or "")

        prior_symbols = set(engine.ledger.positions.keys())
        decision, parse_viols = parse_decision(
            raw_text or "",
            prior_open_symbols=prior_symbols,
            blind_map_inv=blind_inv,
        )

        # One repair attempt
        if decision is None and client is not None and not is_baseline:
            repair_user = render_repair_prompt(
                parse_viols[0].detail if parse_viols else "invalid",
                raw_text or "",
            )
            comp = client.complete(
                system,
                repair_user,
                seed=seed,
                temperature=0.0,
                max_tokens=int(model_cfg.get("max_tokens", 2000)),
            )
            raw_text = comp.text
            total_in_tok += comp.input_tokens
            total_out_tok += comp.output_tokens
            (step_dir / "response_raw.txt").write_text(raw_text or "")
            decision, parse_viols = parse_decision(
                raw_text or "",
                prior_open_symbols=prior_symbols,
                blind_map_inv=blind_inv,
            )

        orders: list[Order] = []
        decision_dict: dict
        if decision is None:
            decision_dict = {
                "portfolio_view": "",
                "orders": [],
                "changed_view_because": None,
                "risk_note": "MALFORMED — zero orders executed",
            }
            for v in parse_viols:
                vd = v.to_dict()
                vd["step"] = step_idx
                all_violations.append(vd)
        else:
            decision_dict = decision.to_dict()
            orders = decision.orders
            texts_for_leak.extend(
                [
                    decision.portfolio_view,
                    decision.changed_view_because or "",
                    decision.risk_note,
                ]
                + [o.thesis for o in orders]
            )

        _json_dump(step_dir / "decision.json", decision_dict)

        result = engine.step(step_idx, orders)

        validated_out = [v.to_dict() for v in result.validated]
        fills_out = [f.to_dict() for f in result.fills]
        _json_dump(step_dir / "orders_validated.json", validated_out)
        _json_dump(step_dir / "fills.json", fills_out)
        _json_dump(step_dir / "ledger_after.json", result.ledger_after.to_dict())

        for v in result.violations:
            all_violations.append(v.to_dict())

        # prior decision summary for next observation
        prior_decisions.append({
            "step": step_idx,
            "thesis_summary": (decision_dict.get("portfolio_view") or "")[:200],
            "orders": [
                f"{o.get('side')} {o.get('symbol')} {o.get('notional_usd')}"
                for o in decision_dict.get("orders") or []
            ],
        })

        step_artifacts.append({
            "step": step_idx,
            "decision_date": result.decision_date.isoformat(),
            "fill_date": result.fill_date.isoformat(),
            "observation": observation_for_prompt(obs),
            "decision": decision_dict,
            "fills": fills_out,
            "ledger_after": result.ledger_after.to_dict(),
        })

    # Leak scan
    leak_flags = scan_for_leaks(texts_for_leak, start)
    for lf in leak_flags:
        all_violations.append(lf)

    # Persist ledger + violations
    ledger_df = engine.ledger_daily_df()
    ledger_path = out / "ledger_daily.parquet"
    ledger_df.to_parquet(ledger_path, index=False)

    with open(out / "violations.jsonl", "w") as f:
        for v in all_violations:
            f.write(json.dumps(v, default=str) + "\n")

    metrics = compute_episode_metrics(
        ledger_df,
        all_violations,
        step_artifacts=step_artifacts,
        starting_cash=starting_cash,
        input_tokens=total_in_tok,
        output_tokens=total_out_tok,
    )
    metrics["model"] = model_id
    metrics["window"] = window_id
    metrics["seed"] = seed
    metrics["mode"] = mode
    metrics["track"] = window.get("track", "historical")
    if probe_result:
        metrics["probe_mean"] = probe_result.get("mean_score")
    _json_dump(out / "metrics.json", metrics)

    manifest = {
        "run_id": rid,
        "campaign": campaign,
        "snapshot_id": snapshot.snapshot_id,
        "model": model_id,
        "model_cfg": model_cfg,
        "window": window_id,
        "window_start": start.isoformat(),
        "seed": seed,
        "mode": mode,
        "prompt_version": prompt_version,
        "prompt_hash": prompt_hash(),
        "git_sha": _git_sha(),
        "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "steps": steps,
        "starting_cash": starting_cash,
    }
    _json_dump(out / "manifest.json", manifest)
    _json_dump(out / "config.json", {
        "model": model_cfg,
        "window": window,
        "seed": seed,
        "mode": mode,
        "steps": steps,
        "starting_cash": starting_cash,
    })

    write_run_report(
        out / "report.html",
        run_id=rid,
        manifest=manifest,
        metrics=metrics,
        ledger_daily=ledger_df,
        steps=step_artifacts,
        violations=all_violations,
    )

    return {
        "run_id": rid,
        "path": str(out),
        "metrics": metrics,
        "manifest": manifest,
    }


def run_campaign(
    config_path: str | Path,
    *,
    snapshot_dir: str | Path | None = None,
    runs_dir: str | Path | None = None,
    models_filter: list[str] | None = None,
    max_episodes: int | None = None,
) -> dict[str, Any]:
    """Run a full campaign from YAML config."""
    cfg = load_config(config_path)
    root = Path(config_path).resolve().parent.parent  # configs/ -> version1/
    snap_id = cfg.get("snapshot_id") or "auto"
    alt = root / "snapshots"

    if snapshot_dir:
        snap_path = Path(snapshot_dir)
    elif snap_id not in ("auto", "", None) and (alt / snap_id).exists():
        snap_path = alt / snap_id
    else:
        candidates = sorted(alt.glob("*/manifest.json"))
        found = None
        for c in candidates:
            m = json.loads(c.read_text())
            if snap_id not in ("auto", "", None) and (
                m.get("snapshot_id") == snap_id or c.parent.name == snap_id
            ):
                found = c.parent
                break
        if found is None and candidates:
            found = candidates[0].parent
        if found is None:
            raise FileNotFoundError(
                f"Snapshot {snap_id!r} not found under {alt}. "
                "Run: tradingbench build-snapshot --out snapshots"
            )
        snap_path = found

    snapshot = load_snapshot(snap_path)
    runs = Path(runs_dir) if runs_dir else root / "runs"
    runs.mkdir(parents=True, exist_ok=True)

    ep = cfg.get("episode", {})
    models = cfg.get("models", [])
    baselines = cfg.get("baselines", [])
    # Expand baselines into model configs
    all_models = list(models)
    for b in baselines:
        all_models.append({"id": b, "provider": "baseline", "name": b, "temperature": 0.0})

    if models_filter:
        all_models = [m for m in all_models if m["id"] in models_filter]

    modes = cfg.get("modes", ["standard"])
    windows = cfg.get("windows", [])
    seeds = cfg.get("seeds", [1])
    campaign = cfg.get("campaign", "v1")

    results = []
    n = 0
    for mode in modes:
        for window in windows:
            for model_cfg in all_models:
                for seed in seeds:
                    if max_episodes is not None and n >= max_episodes:
                        break
                    r = run_episode(
                        snapshot,
                        model_cfg=model_cfg,
                        window=window,
                        seed=int(seed),
                        mode=mode,
                        starting_cash=float(ep.get("starting_cash", 1000.0)),
                        steps=int(ep.get("steps", 12)),
                        max_position_weight=float(ep.get("max_position_weight", 0.25)),
                        runs_dir=runs,
                        campaign=campaign,
                        prompt_version=cfg.get("prompt_version", PROMPT_VERSION),
                        run_probe=bool(cfg.get("run_probe", False)),
                    )
                    results.append(r)
                    n += 1
                    print(f"[{n}] {r['run_id']}  return={r['metrics']['total_return']:+.2%}")
                if max_episodes is not None and n >= max_episodes:
                    break
            if max_episodes is not None and n >= max_episodes:
                break
        if max_episodes is not None and n >= max_episodes:
            break

    episode_metrics = [r["metrics"] for r in results]
    from tradingbench.eval.aggregate import aggregate_metrics
    from tradingbench.report.compare_report import write_compare_report

    agg = aggregate_metrics(episode_metrics)
    campaign_dir = runs / f"_campaign_{campaign}"
    campaign_dir.mkdir(exist_ok=True)
    _json_dump(campaign_dir / "aggregate.json", agg)
    _json_dump(campaign_dir / "episodes.json", episode_metrics)
    report = write_compare_report(
        campaign_dir / "compare_report.html",
        campaign=campaign,
        aggregate=agg,
        episode_metrics=episode_metrics,
    )
    print(f"\nCampaign complete: {n} episodes")
    print(f"Comparison report: {report}")
    return {"results": results, "aggregate": agg, "report": str(report)}
