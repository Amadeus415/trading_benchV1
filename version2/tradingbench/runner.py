"""Episode + campaign orchestration."""

from __future__ import annotations

import json
import subprocess
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from tradingbench.agent.baselines import BASELINES, get_baseline, seed_rng
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
from tradingbench.eval.attribution import attribute_portfolio, weights_from_step_artifacts
from tradingbench.eval.contamination.alias_map import AliasMap, DECISION_MODES, MASKS
from tradingbench.eval.contamination.leak_detect import scan_for_leaks
from tradingbench.eval.contamination.probe import run_memory_probe
from tradingbench.eval.contamination.synthetic import block_bootstrap_prices
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


def _normalize_mode(
    mode: str | None,
    mask: str | None,
    decision_mode: str | None,
) -> tuple[str, str, str]:
    """Return (mask, decision_mode, composite_mode_tag).

    Accepts either the new (mask, decision_mode) pair or a legacy mode string.
    """
    legacy = {
        "standard": ("bright", "standard"),
        "named_control": ("bright", "standard"),
        "blind": ("blinded", "standard"),
        "synthetic": ("bright", "standard"),
        "memory_only": ("bright", "memory_only"),
    }
    if mask is None and decision_mode is None and mode:
        # composite like "blinded:standard" or legacy
        if ":" in mode:
            parts = mode.split(":", 1)
            mask, decision_mode = parts[0], parts[1]
        elif mode in legacy:
            mask, decision_mode = legacy[mode]
        else:
            mask, decision_mode = mode, "standard"
    mask = mask or "bright"
    decision_mode = decision_mode or "standard"
    if mask not in MASKS and mask not in ("synthetic",):
        # allow synthetic as price_path flag, treat as bright for mask
        if mask == "synthetic":
            mask = "bright"
        else:
            raise ValueError(f"Unknown mask {mask!r}; choose from {MASKS}")
    if decision_mode not in DECISION_MODES:
        raise ValueError(f"Unknown decision_mode {decision_mode!r}; choose from {DECISION_MODES}")
    tag = f"{mask}_{decision_mode}"
    return mask, decision_mode, tag


def run_episode(
    snapshot: Snapshot,
    *,
    model_cfg: dict,
    window: dict,
    seed: int,
    mode: str | None = "standard",
    mask: str | None = None,
    decision_mode: str | None = None,
    price_path: str = "real",
    starting_cash: float = 1000.0,
    steps: int = 12,
    max_position_weight: float = 0.25,
    runs_dir: Path,
    campaign: str = "v2",
    prompt_version: str = PROMPT_VERSION,
    run_probe: bool = False,
    run_attribution: bool = True,
    response_cache: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Run one full episode; write artifacts under runs/{run_id}/."""
    mask, decision_mode, mode_tag = _normalize_mode(mode, mask, decision_mode)
    # synthetic price path
    if price_path == "synthetic" or (mode == "synthetic"):
        mode_tag = f"{mode_tag}_synth"
        snap = _synthetic_snapshot(snapshot, seed)
    else:
        snap = snapshot

    model_id = model_cfg["id"]
    window_id = window["id"]
    start = _parse_date(window["start"])
    rid = run_id_for(campaign, model_id, window_id, seed, mode_tag)
    out = Path(runs_dir) / rid
    out.mkdir(parents=True, exist_ok=True)
    steps_dir = out / "steps"
    steps_dir.mkdir(exist_ok=True)

    engine = EpisodeEngine(
        snapshot=snap,
        starting_cash=starting_cash,
        max_position_weight=max_position_weight,
        steps=steps,
        start_date=start,
    )

    # Episode-stable alias map (reshuffled across episodes via seed)
    uni0 = engine.agent_store(0).universe()
    symbols0 = uni0["symbol"].tolist()
    alias_map = AliasMap.build(
        symbols0,
        seed=seed,
        episode_start=start,
        mask=mask,
    )

    provider = model_cfg.get("provider", "baseline")
    is_baseline = (
        provider in ("baseline", "mock", "local")
        or model_id in BASELINES
        or model_cfg.get("name") in BASELINES
    )

    client = None
    baseline_fn = None
    if is_baseline:
        bname = model_cfg.get("name") or model_id
        if bname in BASELINES:
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
    schema_errors = 0
    arithmetic_errors = 0  # notional orders suppress most; still track

    # Optional memory probe (pre-episode)
    probe_result = None
    if run_probe:
        probe_symbols = symbols0[:5] if symbols0 else ["NVDA", "AAPL", "BTC"]
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
            mask=mask,
            decision_mode=decision_mode,
            mode=mode if mode in ("named_control",) else None,
            max_position_weight=max_position_weight,
            alias_map=alias_map,
            episode_seed=seed,
            episode_start=start,
        )

        blind_inv = obs.get("_blind_map_inv") or alias_map.ticker_alias_to_real
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
            # Baselines see unmasked observation (they are not contamination subjects);
            # for fair mask A/B on LLMs only. Spec: baselines run on bright paths.
            # When evaluating mask effect on baselines themselves, use prompt view.
            decision_dict = baseline_fn(observation_for_prompt(obs), rng)
            # Unmask any aliased symbols baselines echoed from observation
            if mask in ("stock_blind", "blinded"):
                for o in decision_dict.get("orders") or []:
                    if o.get("symbol") in blind_inv:
                        o["symbol"] = blind_inv[o["symbol"]]
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
            blind_map_inv=blind_inv if mask in ("stock_blind", "blinded") else None,
        )

        # One repair attempt
        if decision is None and client is not None and not is_baseline:
            schema_errors += 1
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
                blind_map_inv=blind_inv if mask in ("stock_blind", "blinded") else None,
            )

        orders: list[Order] = []
        decision_dict: dict
        if decision is None:
            schema_errors += 1
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
            if v.code in ("INSUFFICIENT_CASH", "INSUFFICIENT_POSITION", "DUST"):
                arithmetic_errors += 1

        # prior decision summary for next observation (real symbols for internal state)
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
    metrics["mode"] = mode_tag
    metrics["mask"] = mask
    metrics["decision_mode"] = decision_mode
    metrics["price_path"] = price_path if price_path != "real" or mode != "synthetic" else (
        "synthetic" if mode == "synthetic" else "real"
    )
    metrics["track"] = window.get("track", "historical")
    metrics["schema_error_rate"] = round(schema_errors / max(1, steps), 4)
    metrics["arithmetic_error_rate"] = round(arithmetic_errors / max(1, steps), 4)
    if probe_result:
        metrics["probe_mean"] = probe_result.get("mean_score")

    # Attribution (selection alpha)
    if run_attribution and not ledger_df.empty:
        try:
            wdf = weights_from_step_artifacts(step_artifacts, ledger_df)
            if not wdf.empty:
                attr = attribute_portfolio(snap.prices, wdf)
                metrics["selection_alpha"] = attr.selection_alpha
                metrics["attribution_common"] = attr.common
                metrics["attribution_style"] = attr.style
                _json_dump(out / "attribution.json", attr.to_dict())
        except Exception as e:
            metrics["attribution_error"] = str(e)

    _json_dump(out / "metrics.json", metrics)

    manifest = {
        "run_id": rid,
        "campaign": campaign,
        "snapshot_id": snap.snapshot_id,
        "model": model_id,
        "model_cfg": model_cfg,
        "window": window_id,
        "window_start": start.isoformat(),
        "seed": seed,
        "mode": mode_tag,
        "mask": mask,
        "decision_mode": decision_mode,
        "price_path": metrics["price_path"],
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
        "mode": mode_tag,
        "mask": mask,
        "decision_mode": decision_mode,
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


def _synthetic_snapshot(snapshot: Snapshot, seed: int) -> Snapshot:
    """Return a shallow copy of snapshot with block-bootstrapped prices."""
    from dataclasses import replace

    synth_prices = block_bootstrap_prices(snapshot.prices, seed=seed)
    return Snapshot(
        snapshot_id=f"{snapshot.snapshot_id}_synth{seed}",
        path=snapshot.path,
        prices=synth_prices,
        corporate_actions=snapshot.corporate_actions,
        news=snapshot.news,
        universe=snapshot.universe,
        manifest={**snapshot.manifest, "synthetic": True, "synth_seed": seed},
    )


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
    root = Path(config_path).resolve().parent.parent  # configs/ -> version2/
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
    all_models = list(models)
    for b in baselines:
        all_models.append({"id": b, "provider": "baseline", "name": b, "temperature": 0.0})

    if models_filter:
        all_models = [m for m in all_models if m["id"] in models_filter]

    # New config shape: masks × decision_modes × price_paths
    # Fallback: modes: [standard, blind, ...]
    masks = cfg.get("masks")
    decision_modes = cfg.get("decision_modes", ["standard"])
    price_paths = cfg.get("price_paths", ["real"])
    if masks is None:
        # legacy modes list
        legacy_modes = cfg.get("modes", ["standard"])
        cell_specs = []
        for m in legacy_modes:
            mask, dm, _ = _normalize_mode(m, None, None)
            pp = "synthetic" if m == "synthetic" else "real"
            cell_specs.append((mask, dm, pp))
    else:
        cell_specs = [
            (mask, dm, pp)
            for mask in masks
            for dm in decision_modes
            for pp in price_paths
        ]

    # Anchor-deep / others-shallow (§10)
    anchor = cfg.get("anchor_model")
    shallow_masks = cfg.get("shallow_masks", ["blinded"])
    shallow_modes = cfg.get("shallow_decision_modes", ["standard"])

    windows = cfg.get("windows", [])
    seeds = cfg.get("seeds", [1])
    campaign = cfg.get("campaign", "v2")

    results = []
    n = 0
    for mask, dm, pp in cell_specs:
        for window in windows:
            for model_cfg in all_models:
                mid = model_cfg["id"]
                # Shallow grid for non-anchor LLM models
                if (
                    anchor
                    and mid != anchor
                    and model_cfg.get("provider") not in ("baseline", "mock", "local")
                    and mid not in BASELINES
                ):
                    if mask not in shallow_masks or dm not in shallow_modes:
                        continue
                    if pp != "real":
                        continue
                for seed in seeds:
                    if max_episodes is not None and n >= max_episodes:
                        break
                    r = run_episode(
                        snapshot,
                        model_cfg=model_cfg,
                        window=window,
                        seed=int(seed),
                        mask=mask,
                        decision_mode=dm,
                        price_path=pp,
                        starting_cash=float(ep.get("starting_cash", 1000.0)),
                        steps=int(ep.get("steps", 12)),
                        max_position_weight=float(ep.get("max_position_weight", 0.25)),
                        runs_dir=runs,
                        campaign=campaign,
                        prompt_version=cfg.get("prompt_version", PROMPT_VERSION),
                        run_probe=bool(cfg.get("run_probe", False)),
                        run_attribution=bool(cfg.get("run_attribution", True)),
                    )
                    results.append(r)
                    n += 1
                    sa = r["metrics"].get("selection_alpha")
                    sa_s = f"  α={sa:+.4f}" if sa is not None else ""
                    print(
                        f"[{n}] {r['run_id']}  return={r['metrics']['total_return']:+.2%}"
                        f"{sa_s}"
                    )
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
