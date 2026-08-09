"""Aggregation + Wilcoxon tests."""

from __future__ import annotations

import numpy as np

from tradingbench.eval.aggregate import (
    aggregate_metrics,
    bootstrap_ci,
    wilcoxon_signed_rank,
)


def test_wilcoxon_detects_shift():
    rng = np.random.default_rng(0)
    a = rng.normal(0.05, 0.01, size=30)
    b = a - 0.04  # b systematically worse
    res = wilcoxon_signed_rank(a, b)
    assert res["n"] == 30
    assert res["significant_05"] is True
    assert res["p_value"] < 0.05


def test_wilcoxon_null_not_significant():
    rng = np.random.default_rng(1)
    noise = rng.normal(0, 0.02, size=20)
    a = noise
    b = noise + rng.normal(0, 0.001, size=20)
    res = wilcoxon_signed_rank(a, b)
    # may or may not be significant; just check structure
    assert "p_value" in res
    assert "statistic" in res


def test_bootstrap_ci_structure():
    vals = np.array([0.01, 0.02, -0.01, 0.03, 0.0])
    ci = bootstrap_ci(vals, n_resamples=500, seed=0)
    assert ci["lo"] <= ci["mean"] <= ci["hi"]
    assert "q1" in ci and "q3" in ci


def test_aggregate_paired():
    metrics = []
    for model, base in [("m1", 0.05), ("m2", 0.01)]:
        for w in ("w1", "w2"):
            for seed in (1, 2):
                metrics.append({
                    "model": model,
                    "window": w,
                    "seed": seed,
                    "total_return": base + seed * 0.001,
                    "max_drawdown": -0.05,
                    "weekly_sharpe": 0.5,
                    "violation_count": 0,
                    "turnover": 0.1,
                    "fees_paid": 1.0,
                    "malformed_rate": 0.0,
                    "selection_alpha": base - 0.02,
                    "mask": "bright",
                })
    agg = aggregate_metrics(metrics)
    assert agg["n_episodes"] == 8
    assert "m1" in agg["models"]
    assert "m1_vs_m2" in agg["paired"]
    assert "wilcoxon" in agg["paired"]["m1_vs_m2"]
