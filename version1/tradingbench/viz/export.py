"""Export a run directory into a compact JSON bundle for the interactive viz."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


def export_run_bundle(run_dir: str | Path, out_path: str | Path | None = None) -> dict[str, Any]:
    """Read episode artifacts and produce a browser-friendly bundle."""
    run_dir = Path(run_dir)
    if not run_dir.exists():
        raise FileNotFoundError(run_dir)

    manifest = _read_json(run_dir / "manifest.json")
    metrics = _read_json(run_dir / "metrics.json")
    config = _read_json(run_dir / "config.json") if (run_dir / "config.json").exists() else {}

    ledger_path = run_dir / "ledger_daily.parquet"
    if ledger_path.exists():
        ledger_df = pd.read_parquet(ledger_path)
        ledger = [
            {
                "date": str(row["date"]),
                "cash": round(float(row["cash"]), 4),
                "nav": round(float(row["nav"]), 4),
                "fees_paid": round(float(row["fees_paid"]), 4),
                "n_positions": int(row["n_positions"]),
                "step": int(row["step"]),
            }
            for _, row in ledger_df.iterrows()
        ]
    else:
        ledger = []

    steps: list[dict[str, Any]] = []
    steps_dir = run_dir / "steps"
    if steps_dir.exists():
        for step_path in sorted(steps_dir.iterdir()):
            if not step_path.is_dir():
                continue
            obs = _read_json(step_path / "observation.json")
            decision = _read_json(step_path / "decision.json")
            fills = _read_json(step_path / "fills.json")
            validated = _read_json(step_path / "orders_validated.json")
            ledger_after = _read_json(step_path / "ledger_after.json")

            # Compact market: top movers by |1w| for viz readability
            market = obs.get("market") or []
            ranked = sorted(
                market,
                key=lambda m: abs(m.get("ret_1w") or 0),
                reverse=True,
            )
            market_top = ranked[:12]
            market_all_lite = [
                {
                    "symbol": m["symbol"],
                    "asset_class": m.get("asset_class"),
                    "sector": m.get("sector"),
                    "last": m.get("last"),
                    "ret_1w": m.get("ret_1w"),
                    "ret_1m": m.get("ret_1m"),
                    "vol_20d": m.get("vol_20d"),
                }
                for m in market
            ]

            steps.append({
                "step": int(step_path.name),
                "as_of": obs.get("as_of"),
                "episode": obs.get("episode"),
                "portfolio": obs.get("portfolio"),
                "rules": obs.get("rules"),
                "news": (obs.get("news") or [])[:6],
                "prior_decisions": obs.get("prior_decisions") or [],
                "last_step_violations": obs.get("last_step_violations") or [],
                "market_top": market_top,
                "market": market_all_lite,
                "decision": decision,
                "fills": fills,
                "validated": validated,
                "ledger_after": ledger_after,
            })

    violations = []
    viol_path = run_dir / "violations.jsonl"
    if viol_path.exists():
        for line in viol_path.read_text().splitlines():
            if line.strip():
                violations.append(json.loads(line))

    bundle = {
        "run_id": manifest.get("run_id") or run_dir.name,
        "manifest": manifest,
        "config": config,
        "metrics": metrics,
        "ledger": ledger,
        "steps": steps,
        "violations": violations,
        "guide": GUIDE,
    }

    if out_path is None:
        out_path = run_dir / "viz_bundle.json"
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(bundle, indent=2, default=str))
    return bundle


def write_data_js(bundle: dict, out_path: str | Path) -> Path:
    """Write window.TB_DATA = {...} for file:// friendly loading."""
    out_path = Path(out_path)
    out_path.write_text(
        "window.TB_DATA = " + json.dumps(bundle, default=str) + ";\n"
    )
    return out_path


def _read_json(path: Path) -> Any:
    if not path.exists():
        return {} if path.suffix == ".json" else []
    return json.loads(path.read_text())


GUIDE = {
    "title": "How Trading Bench works",
    "subtitle": "A measurement instrument for sequential decisions — not a broker.",
    "pipeline": [
        {
            "id": "observe",
            "label": "Observe",
            "title": "Build the observation",
            "body": "At each week’s close (day t), the system freezes a point-in-time snapshot: portfolio, derived market stats, news, rules, and recent violations. The model cannot see the future.",
        },
        {
            "id": "decide",
            "label": "Decide",
            "title": "Model returns a decision",
            "body": "The agent outputs JSON only: a portfolio view, zero or more notional orders (buy/sell in dollars), confidence, and optional change-of-view notes. Holding (empty orders) is valid.",
        },
        {
            "id": "validate",
            "label": "Validate",
            "title": "Exchange checks the rules",
            "body": "Each order is checked for unknown symbols, cash, position caps (25%), min size ($10), no shorting, and market availability. Rejections become violations the agent sees next week.",
        },
        {
            "id": "fill",
            "label": "Fill",
            "title": "Next open execution",
            "body": "Accepted orders fill at the next session open with slippage (10 bps equity / 25 bps crypto) and 5 bps commission. Fractional shares are allowed — necessary at $1,000 NAV.",
        },
        {
            "id": "mark",
            "label": "Mark",
            "title": "Daily mark-to-market",
            "body": "Between decision weeks the ledger marks positions to market, applies corporate actions, and records NAV. After ~7 days, the loop repeats.",
        },
    ],
    "principles": [
        "Fake money only — models never touch a real brokerage.",
        "The system owns pricing, fills, and accounting — not the model.",
        "Same rules, data, and budget for every model (Comparable).",
        "Runs are reproducible and fully audited step-by-step.",
    ],
}
