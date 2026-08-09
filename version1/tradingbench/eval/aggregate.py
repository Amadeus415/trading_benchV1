"""Cross-episode aggregation: paired stats, bootstrap CIs, MDE."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def bootstrap_ci(
    values: np.ndarray,
    n_resamples: int = 10_000,
    alpha: float = 0.05,
    seed: int = 0,
) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    values = np.asarray(values, dtype=float)
    values = values[~np.isnan(values)]
    if len(values) == 0:
        return {"median": float("nan"), "mean": float("nan"), "lo": float("nan"), "hi": float("nan"), "iqr": float("nan")}
    if len(values) == 1:
        v = float(values[0])
        return {"median": v, "mean": v, "lo": v, "hi": v, "iqr": 0.0}

    med = float(np.median(values))
    mean = float(np.mean(values))
    q25, q75 = np.percentile(values, [25, 75])
    boots = []
    n = len(values)
    for _ in range(n_resamples):
        sample = rng.choice(values, size=n, replace=True)
        boots.append(np.mean(sample))
    lo, hi = np.percentile(boots, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {
        "median": med,
        "mean": mean,
        "lo": float(lo),
        "hi": float(hi),
        "iqr": float(q75 - q25),
    }


def paired_delta_ci(
    a: np.ndarray,
    b: np.ndarray,
    n_resamples: int = 10_000,
    seed: int = 0,
) -> dict[str, float]:
    """Bootstrap CI on paired mean(a - b)."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    d = a - b
    return bootstrap_ci(d, n_resamples=n_resamples, seed=seed)


def minimum_detectable_effect(
    n: int,
    sd: float,
    alpha: float = 0.05,
    power: float = 0.8,
) -> float:
    """Approximate two-sided MDE for paired mean difference (z approximation)."""
    if n <= 1 or sd <= 0 or np.isnan(sd):
        return float("nan")
    # z_alpha/2 ≈ 1.96, z_power ≈ 0.84 for 80%
    z_a = 1.96 if abs(alpha - 0.05) < 1e-9 else 1.96
    z_b = 0.8416 if abs(power - 0.8) < 1e-9 else 0.8416
    return float((z_a + z_b) * sd / np.sqrt(n))


def aggregate_metrics(
    episode_metrics: list[dict[str, Any]],
    group_key: str = "model",
) -> dict[str, Any]:
    """Aggregate a list of per-episode metrics dicts (must include model, window, seed, total_return)."""
    if not episode_metrics:
        return {"models": {}, "n_episodes": 0}

    df = pd.DataFrame(episode_metrics)
    models = {}
    for model, g in df.groupby(group_key):
        rets = g["total_return"].astype(float).values
        models[str(model)] = {
            "n": len(g),
            "total_return": bootstrap_ci(rets),
            "max_drawdown": bootstrap_ci(g["max_drawdown"].astype(float).values),
            "sharpe": bootstrap_ci(g.get("weekly_sharpe", pd.Series([0.0] * len(g))).astype(float).values),
            "violation_count_mean": float(g["violation_count"].mean()) if "violation_count" in g else 0.0,
            "mean_turnover": float(g["turnover"].mean()) if "turnover" in g else 0.0,
            "mean_fees": float(g["fees_paid"].mean()) if "fees_paid" in g else 0.0,
            "malformed_rate_mean": float(g["malformed_rate"].mean()) if "malformed_rate" in g else 0.0,
        }
        sd = float(np.std(rets, ddof=1)) if len(rets) > 1 else 0.0
        models[str(model)]["mde_return"] = minimum_detectable_effect(len(rets), sd)

    # Paired comparisons if window+seed present
    paired = {}
    if {"window", "seed", group_key, "total_return"}.issubset(df.columns):
        model_list = sorted(df[group_key].unique())
        for i, m1 in enumerate(model_list):
            for m2 in model_list[i + 1 :]:
                a = df[df[group_key] == m1].set_index(["window", "seed"])["total_return"]
                b = df[df[group_key] == m2].set_index(["window", "seed"])["total_return"]
                common = a.index.intersection(b.index)
                if len(common) == 0:
                    continue
                ci = paired_delta_ci(a.loc[common].values, b.loc[common].values)
                sd = float(np.std(a.loc[common].values - b.loc[common].values, ddof=1)) if len(common) > 1 else 0.0
                paired[f"{m1}_vs_{m2}"] = {
                    "n_pairs": len(common),
                    "delta_return": ci,
                    "mde": minimum_detectable_effect(len(common), sd),
                    "indistinguishable": abs(ci["mean"]) < minimum_detectable_effect(len(common), sd)
                    if not np.isnan(minimum_detectable_effect(len(common), sd))
                    else True,
                }

    return {
        "n_episodes": len(df),
        "models": models,
        "paired": paired,
    }
