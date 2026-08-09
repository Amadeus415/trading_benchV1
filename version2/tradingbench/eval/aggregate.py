"""Cross-episode aggregation: paired stats, bootstrap CIs, Wilcoxon, MDE."""

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
        return {
            "median": float("nan"),
            "mean": float("nan"),
            "lo": float("nan"),
            "hi": float("nan"),
            "iqr": float("nan"),
            "q1": float("nan"),
            "q3": float("nan"),
        }
    if len(values) == 1:
        v = float(values[0])
        return {"median": v, "mean": v, "lo": v, "hi": v, "iqr": 0.0, "q1": v, "q3": v}

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
        "q1": float(q25),
        "q3": float(q75),
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
    z_a = 1.96
    z_b = 0.8416 if abs(power - 0.8) < 1e-9 else 0.8416
    return float((z_a + z_b) * sd / np.sqrt(n))


def wilcoxon_signed_rank(
    a: np.ndarray,
    b: np.ndarray,
) -> dict[str, Any]:
    """Two-sided Wilcoxon signed-rank test on paired differences a-b.

    Pure numpy implementation (no scipy dependency). Returns statistic,
    approximate normal p-value, and whether significant at α=0.05.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    d = a - b
    d = d[~np.isnan(d)]
    # drop zeros
    d = d[d != 0]
    n = len(d)
    if n < 5:
        return {
            "n": n,
            "statistic": float("nan"),
            "p_value": float("nan"),
            "significant_05": False,
            "note": "n<5; test not reliable",
        }

    abs_d = np.abs(d)
    # average ranks for ties
    order = np.argsort(abs_d)
    ranks = np.empty(n, dtype=float)
    ranks[order] = np.arange(1, n + 1, dtype=float)
    # tie correction
    _, inv, counts = np.unique(abs_d, return_inverse=True, return_counts=True)
    for i, c in enumerate(counts):
        if c > 1:
            idx = np.where(inv == i)[0]
            ranks[idx] = ranks[idx].mean()

    w_plus = float(ranks[d > 0].sum())
    w_minus = float(ranks[d < 0].sum())
    w = min(w_plus, w_minus)

    # Normal approximation with tie correction
    mean_w = n * (n + 1) / 4.0
    tie_term = np.sum(counts ** 3 - counts) / 48.0
    var_w = n * (n + 1) * (2 * n + 1) / 24.0 - tie_term
    if var_w <= 0:
        return {
            "n": n,
            "statistic": w,
            "p_value": float("nan"),
            "significant_05": False,
        }
    # continuity correction
    z = (w - mean_w + 0.5 * np.sign(mean_w - w)) / np.sqrt(var_w)
    # two-sided p from normal: erfc
    from math import erfc, sqrt

    p = erfc(abs(z) / sqrt(2.0))
    return {
        "n": n,
        "statistic": w,
        "w_plus": w_plus,
        "w_minus": w_minus,
        "z": float(z),
        "p_value": float(p),
        "significant_05": bool(p < 0.05),
    }


def aggregate_metrics(
    episode_metrics: list[dict[str, Any]],
    group_key: str = "model",
    primary_metric: str = "selection_alpha",
    fallback_metric: str = "total_return",
) -> dict[str, Any]:
    """Aggregate per-episode metrics.

    Primary leaderboard column is selection_alpha when present; total_return
    is always reported. Never bare means — median, IQR, bootstrap 95% CI.
    """
    if not episode_metrics:
        return {"models": {}, "n_episodes": 0, "primary_metric": primary_metric}

    df = pd.DataFrame(episode_metrics)
    # Choose metric column
    if primary_metric not in df.columns or df[primary_metric].isna().all():
        metric_col = fallback_metric
    else:
        metric_col = primary_metric

    models: dict[str, Any] = {}
    for model, g in df.groupby(group_key):
        rets = g["total_return"].astype(float).values if "total_return" in g else np.array([])
        primary_vals = (
            g[metric_col].astype(float).values
            if metric_col in g
            else rets
        )
        entry: dict[str, Any] = {
            "n": len(g),
            "total_return": bootstrap_ci(rets) if len(rets) else {},
            "max_drawdown": bootstrap_ci(g["max_drawdown"].astype(float).values)
            if "max_drawdown" in g
            else {},
            "sharpe": bootstrap_ci(
                g["weekly_sharpe"].astype(float).values
                if "weekly_sharpe" in g
                else np.zeros(len(g))
            ),
            "violation_count_mean": float(g["violation_count"].mean())
            if "violation_count" in g
            else 0.0,
            "mean_turnover": float(g["turnover"].mean()) if "turnover" in g else 0.0,
            "mean_fees": float(g["fees_paid"].mean()) if "fees_paid" in g else 0.0,
            "malformed_rate_mean": float(g["malformed_rate"].mean())
            if "malformed_rate" in g
            else 0.0,
            "schema_error_rate_mean": float(g["schema_error_rate"].mean())
            if "schema_error_rate" in g
            else 0.0,
            "arithmetic_error_rate_mean": float(g["arithmetic_error_rate"].mean())
            if "arithmetic_error_rate" in g
            else 0.0,
        }
        if metric_col in g:
            entry[metric_col] = bootstrap_ci(primary_vals)
        if "selection_alpha" in g and not g["selection_alpha"].isna().all():
            entry["selection_alpha"] = bootstrap_ci(
                g["selection_alpha"].astype(float).values
            )
        sd = float(np.std(primary_vals[~np.isnan(primary_vals)], ddof=1)) if (
            len(primary_vals) > 1 and np.any(~np.isnan(primary_vals))
        ) else 0.0
        n_valid = int(np.sum(~np.isnan(primary_vals)))
        entry["mde"] = minimum_detectable_effect(n_valid, sd)
        entry["mde_return"] = minimum_detectable_effect(
            len(rets), float(np.std(rets, ddof=1)) if len(rets) > 1 else 0.0
        )
        models[str(model)] = entry

    # Paired comparisons on (window, seed) — same path, different model
    paired: dict[str, Any] = {}
    pair_keys = ["window", "seed"]
    # Prefer matching on mask/decision_mode when present
    for extra in ("mask", "decision_mode", "mode"):
        if extra in df.columns:
            pair_keys.append(extra)
            break

    if set(pair_keys + [group_key, "total_return"]).issubset(df.columns):
        model_list = sorted(df[group_key].unique())
        for i, m1 in enumerate(model_list):
            for m2 in model_list[i + 1 :]:
                idx_cols = pair_keys
                a = df[df[group_key] == m1].drop_duplicates(idx_cols).set_index(idx_cols)
                b = df[df[group_key] == m2].drop_duplicates(idx_cols).set_index(idx_cols)
                common = a.index.intersection(b.index)
                if len(common) == 0:
                    continue
                # Primary metric delta
                col = metric_col if metric_col in a.columns else "total_return"
                av = a.loc[common, col].astype(float).values
                bv = b.loc[common, col].astype(float).values
                # Also total return
                ar = a.loc[common, "total_return"].astype(float).values
                br = b.loc[common, "total_return"].astype(float).values
                ci = paired_delta_ci(av, bv)
                ci_ret = paired_delta_ci(ar, br)
                sd = float(np.std(av - bv, ddof=1)) if len(common) > 1 else 0.0
                mde = minimum_detectable_effect(len(common), sd)
                wx = wilcoxon_signed_rank(av, bv)
                paired[f"{m1}_vs_{m2}"] = {
                    "n_pairs": len(common),
                    "metric": col,
                    "delta": ci,
                    "delta_return": ci_ret,
                    "mde": mde,
                    "indistinguishable": (
                        abs(ci["mean"]) < mde if not np.isnan(mde) else True
                    ),
                    "wilcoxon": wx,
                }

    # Blind premium: mean_paired(bright − blinded) if both present
    contamination: dict[str, Any] = {}
    if "mask" in df.columns and "total_return" in df.columns:
        for model, g in df.groupby(group_key):
            bright = g[g["mask"] == "bright"]
            blinded = g[g["mask"] == "blinded"]
            if bright.empty or blinded.empty:
                continue
            a = bright.set_index(["window", "seed"])["total_return"]
            b = blinded.set_index(["window", "seed"])["total_return"]
            common = a.index.intersection(b.index)
            if len(common) == 0:
                continue
            ci = paired_delta_ci(a.loc[common].values, b.loc[common].values)
            contamination[str(model)] = {
                "blind_premium": ci,
                "n_pairs": len(common),
                "note": "positive premium on historical windows ⇒ recall signature",
            }

    return {
        "n_episodes": len(df),
        "primary_metric": metric_col,
        "models": models,
        "paired": paired,
        "contamination": contamination,
    }
