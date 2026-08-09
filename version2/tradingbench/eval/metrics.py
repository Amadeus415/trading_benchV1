"""Per-episode scorecard metrics."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def compute_episode_metrics(
    ledger_daily: pd.DataFrame,
    violations: list[dict],
    step_artifacts: list[dict] | None = None,
    starting_cash: float = 1000.0,
    cost_usd: float = 0.0,
    input_tokens: int = 0,
    output_tokens: int = 0,
) -> dict[str, Any]:
    """Compute scorecard for a single episode."""
    if ledger_daily is None or ledger_daily.empty:
        return {
            "total_return": 0.0,
            "final_nav": starting_cash,
            "max_drawdown": 0.0,
            "n_steps": 0,
            "error": "empty_ledger",
        }

    df = ledger_daily.sort_values("date").copy()
    nav = df["nav"].astype(float).values
    final_nav = float(nav[-1])
    total_return = final_nav / starting_cash - 1.0

    # Weekly-ish returns from daily marks: use decision-step end navs when available
    if "step" in df.columns:
        step_navs = df.groupby("step")["nav"].last().values.astype(float)
    else:
        step_navs = nav[::5] if len(nav) > 5 else nav
    if len(step_navs) < 2:
        step_rets = np.array([0.0])
    else:
        step_rets = step_navs[1:] / step_navs[:-1] - 1.0

    mean_r = float(np.mean(step_rets)) if len(step_rets) else 0.0
    std_r = float(np.std(step_rets, ddof=1)) if len(step_rets) > 1 else 0.0
    # Annualize weekly-ish (52)
    sharpe = (mean_r / std_r * np.sqrt(52)) if std_r > 1e-12 else 0.0
    downside = step_rets[step_rets < 0]
    downside_std = float(np.std(downside, ddof=1)) if len(downside) > 1 else 0.0
    sortino = (mean_r / downside_std * np.sqrt(52)) if downside_std > 1e-12 else 0.0

    # Max drawdown
    peak = np.maximum.accumulate(nav)
    dd = nav / peak - 1.0
    max_dd = float(dd.min()) if len(dd) else 0.0
    realized_vol = float(np.std(step_rets, ddof=1) * np.sqrt(52)) if len(step_rets) > 1 else 0.0

    best_step = float(step_rets.max()) if len(step_rets) else 0.0
    worst_step = float(step_rets.min()) if len(step_rets) else 0.0

    # Behaviour from steps
    trade_count = 0
    traded_notional = 0.0
    fees = float(df["fees_paid"].iloc[-1]) if "fees_paid" in df.columns else 0.0
    position_counts = []
    cash_weights = []
    holding_periods = []
    churn_closes = 0
    opens = 0
    theses_ok = 0

    if step_artifacts:
        for art in step_artifacts:
            fills = art.get("fills") or []
            for f in fills:
                trade_count += 1
                traded_notional += abs(float(f.get("notional") or 0))
            ledger = art.get("ledger_after") or {}
            nav_s = float(ledger.get("nav") or starting_cash)
            cash_s = float(ledger.get("cash") or 0)
            n_pos = len(ledger.get("positions") or {})
            position_counts.append(n_pos)
            cash_weights.append(cash_s / nav_s if nav_s else 1.0)

            decision = art.get("decision") or {}
            for o in decision.get("orders") or []:
                if o.get("side") == "buy":
                    opens += 1
                    thesis = (o.get("thesis") or "").strip()
                    if len(thesis) >= 20 and thesis.lower() not in ("n/a", "none", "test"):
                        theses_ok += 1
                if o.get("side") == "sell":
                    # churn proxy: sold after 1 step — approximate via held_steps in obs if present
                    churn_closes += 1

    mean_positions = float(np.mean(position_counts)) if position_counts else 0.0
    mean_cash_w = float(np.mean(cash_weights)) if cash_weights else 1.0
    mean_nav = float(np.mean(nav)) if len(nav) else starting_cash
    turnover = traded_notional / mean_nav if mean_nav > 0 else 0.0

    # HHI concentration from last ledger
    hhi = 0.0
    if step_artifacts:
        last_pos = (step_artifacts[-1].get("ledger_after") or {}).get("positions") or {}
        last_nav = float((step_artifacts[-1].get("ledger_after") or {}).get("nav") or 1)
        weights = []
        # need market values — approximate with equal if only qty
        for p in last_pos.values() if isinstance(last_pos, dict) else []:
            # positions stored as {sym: {qty, avg_cost}}
            pass
        obs_pos = (step_artifacts[-1].get("observation") or {}).get("portfolio", {}).get("positions") or []
        if obs_pos:
            for p in obs_pos:
                w = float(p.get("weight") or 0)
                weights.append(w)
            hhi = float(sum(w * w for w in weights))

    # Compliance
    viol_by_code: dict[str, int] = {}
    for v in violations:
        code = v.get("code", "UNKNOWN")
        viol_by_code[code] = viol_by_code.get(code, 0) + 1
    malformed = viol_by_code.get("MALFORMED", 0)
    n_orders = trade_count  # accepted fills
    invalid_rate = (sum(viol_by_code.values()) / max(1, sum(viol_by_code.values()) + n_orders))

    thesis_coverage = theses_ok / opens if opens else None
    churn_rate = churn_closes / max(1, opens) if opens else 0.0

    # Schema vs arithmetic error rates reported separately (StockBench finding)
    schema_codes = {"MALFORMED"}
    arithmetic_codes = {"INSUFFICIENT_CASH", "INSUFFICIENT_POSITION", "DUST"}
    schema_n = sum(viol_by_code.get(c, 0) for c in schema_codes)
    arith_n = sum(viol_by_code.get(c, 0) for c in arithmetic_codes)
    n_steps = int(df["step"].nunique()) if "step" in df.columns else max(1, len(step_navs))

    return {
        "total_return": round(total_return, 6),
        "final_nav": round(final_nav, 4),
        "weekly_sharpe": round(float(sharpe), 4),
        "sortino": round(float(sortino), 4),
        "max_drawdown": round(max_dd, 6),
        "realized_vol": round(realized_vol, 6),
        "best_step": round(best_step, 6),
        "worst_step": round(worst_step, 6),
        "turnover": round(turnover, 4),
        "fees_paid": round(fees, 4),
        "trade_count": trade_count,
        "mean_positions": round(mean_positions, 3),
        "hhi": round(hhi, 4),
        "cash_drag": round(mean_cash_w, 4),
        "violation_count": sum(viol_by_code.values()),
        "violations_by_code": viol_by_code,
        "invalid_order_rate": round(invalid_rate, 4),
        "malformed_rate": round(malformed / max(1, len(step_artifacts or [1])), 4),
        "schema_error_rate": round(schema_n / max(1, n_steps), 4),
        "arithmetic_error_rate": round(arith_n / max(1, n_steps), 4),
        "thesis_coverage": None if thesis_coverage is None else round(thesis_coverage, 4),
        "churn_rate": round(churn_rate, 4),
        "cost_usd": cost_usd,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "n_steps": n_steps,
        # selection_alpha filled by runner via attribution when available
        "selection_alpha": None,
    }
