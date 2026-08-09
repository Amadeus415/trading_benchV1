"""Cross-sectional return attribution (Barra-style, MVP_SPEC §8.3).

Decomposes portfolio daily return into:
  Common  — unit-exposure common-return component f_0,t  (NOT a market index)
  Style   — sum of weighted style factor contributions
  Selection alpha — residual: the thing being measured

Primary leaderboard column is selection_alpha; total return is secondary.

Factor set is VIF-screened on a pre-evaluation calibration window so no
evaluation-period information enters the screen. Do NOT reuse these factors
as features for learned baselines (KTD-Fin coupling trap).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from typing import Any

import numpy as np
import pandas as pd


FACTOR_NAMES = [
    "MOM_12_1",  # 12−1 momentum
    "RV_60",     # 60d realized vol
    "ILLIQ",     # Amihud illiquidity
    "REV_ON",    # overnight return (open/prev_close - 1)
    "MOM_ID",    # intradaily momentum (close/open - 1)
    "SKEW",      # negative return skew
    "CORR_PV",   # volume–return correlation
    "HIGH_52W",  # distance from 52w high
    "CV_VOL",    # coefficient of variation of volume
]


@dataclass
class AttributionResult:
    common: float
    style: float
    selection_alpha: float
    total: float
    daily: list[dict]
    factor_contrib: dict[str, float]

    def to_dict(self) -> dict[str, Any]:
        return {
            "common": round(self.common, 6),
            "style": round(self.style, 6),
            "selection_alpha": round(self.selection_alpha, 6),
            "total": round(self.total, 6),
            "factor_contrib": {k: round(v, 6) for k, v in self.factor_contrib.items()},
            "n_days": len(self.daily),
            "check_sum": round(self.common + self.style + self.selection_alpha, 6),
        }


def _winsorize(x: np.ndarray, p: float = 0.01) -> np.ndarray:
    if len(x) < 5:
        return x
    lo, hi = np.nanpercentile(x, [p * 100, (1 - p) * 100])
    return np.clip(x, lo, hi)


def _cs_standardize(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    mu = np.nanmean(x)
    sd = np.nanstd(x)
    if sd < 1e-12 or np.isnan(sd):
        return np.zeros_like(x)
    return (x - mu) / sd


def compute_style_exposures(prices: pd.DataFrame, as_of: date) -> pd.DataFrame:
    """Compute style exposures strictly from information before day as_of.

    Returns DataFrame indexed by symbol with columns = FACTOR_NAMES.
    """
    df = prices.copy()
    df["date"] = pd.to_datetime(df["date"]).dt.date
    hist = df[df["date"] < as_of].sort_values("date")
    if hist.empty:
        return pd.DataFrame(columns=FACTOR_NAMES)

    rows = []
    for sym, g in hist.groupby("symbol"):
        g = g.sort_values("date")
        closes = g["close"].astype(float).values
        opens = g["open"].astype(float).values
        volumes = g["volume"].astype(float).values
        n = len(closes)
        if n < 5:
            continue

        rets = np.diff(closes) / np.maximum(closes[:-1], 1e-9)

        # MOM_12_1: ~252d return excluding last ~21d
        if n >= 63:
            end = closes[-22] if n > 22 else closes[-1]
            start = closes[max(0, n - 252)]
            mom = end / start - 1.0 if start > 0 else 0.0
        else:
            mom = closes[-1] / closes[0] - 1.0 if closes[0] > 0 else 0.0

        # RV_60
        window = rets[-60:] if len(rets) >= 60 else rets
        rv = float(np.std(window) * np.sqrt(252)) if len(window) > 1 else 0.0

        # ILLIQ Amihud: mean(|r| / dollar_vol)
        dollar_vol = np.maximum(closes[-len(rets):] * volumes[-len(rets):], 1.0)
        illiq = float(np.mean(np.abs(rets) / dollar_vol[-len(rets):])) if len(rets) else 0.0

        # REV_ON: last overnight
        if n >= 2 and closes[-2] > 0:
            rev_on = opens[-1] / closes[-2] - 1.0
        else:
            rev_on = 0.0

        # MOM_ID: last intradaily
        if opens[-1] > 0:
            mom_id = closes[-1] / opens[-1] - 1.0
        else:
            mom_id = 0.0

        # SKEW of returns (negated so high = crash risk)
        skew = float(-np.sign(np.mean((window - np.mean(window)) ** 3)) *
                     abs(np.mean((window - np.mean(window)) ** 3)) ** (1 / 3)
                     ) if len(window) > 5 else 0.0
        # simpler: sample skewness of window
        if len(window) > 5:
            m = np.mean(window)
            s = np.std(window)
            skew = float(np.mean(((window - m) / s) ** 3)) if s > 1e-12 else 0.0
            skew = -skew  # negative skew = left tail

        # CORR_PV
        if len(rets) >= 20:
            r_w = rets[-20:]
            v_w = volumes[-20:]
            if np.std(r_w) > 1e-12 and np.std(v_w) > 1e-12:
                corr_pv = float(np.corrcoef(r_w, v_w)[0, 1])
            else:
                corr_pv = 0.0
        else:
            corr_pv = 0.0

        # HIGH_52W
        look = closes[-252:] if n >= 252 else closes
        peak = look.max()
        high_52w = closes[-1] / peak - 1.0 if peak > 0 else 0.0

        # CV_VOL
        v_w = volumes[-20:] if n >= 20 else volumes
        cv_vol = float(np.std(v_w) / np.mean(v_w)) if np.mean(v_w) > 0 else 0.0

        rows.append({
            "symbol": sym,
            "MOM_12_1": mom,
            "RV_60": rv,
            "ILLIQ": illiq,
            "REV_ON": rev_on,
            "MOM_ID": mom_id,
            "SKEW": skew,
            "CORR_PV": corr_pv,
            "HIGH_52W": high_52w,
            "CV_VOL": cv_vol,
        })

    if not rows:
        return pd.DataFrame(columns=FACTOR_NAMES)

    exp = pd.DataFrame(rows).set_index("symbol")
    # Winsorize + cross-sectionally standardize each factor
    for col in FACTOR_NAMES:
        if col not in exp.columns:
            exp[col] = 0.0
        vals = _winsorize(exp[col].values.astype(float))
        exp[col] = _cs_standardize(vals)
    return exp[FACTOR_NAMES]


def vif_screen(
    exposures: pd.DataFrame,
    threshold: float = 10.0,
) -> list[str]:
    """Drop factors with VIF > threshold. Returns kept factor names."""
    kept = list(exposures.columns)
    if len(exposures) < len(kept) + 2:
        return kept

    changed = True
    while changed and len(kept) > 1:
        changed = False
        X = exposures[kept].values.astype(float)
        # drop rows with nan
        mask = ~np.any(np.isnan(X), axis=1)
        X = X[mask]
        if len(X) < len(kept) + 2:
            break
        vifs = []
        for i, name in enumerate(kept):
            y = X[:, i]
            Z = np.delete(X, i, axis=1)
            Z = np.column_stack([np.ones(len(Z)), Z])
            try:
                beta, *_ = np.linalg.lstsq(Z, y, rcond=None)
                yhat = Z @ beta
                ss_res = np.sum((y - yhat) ** 2)
                ss_tot = np.sum((y - np.mean(y)) ** 2)
                r2 = 1 - ss_res / ss_tot if ss_tot > 1e-12 else 0.0
                r2 = min(max(r2, 0.0), 0.9999)
                vif = 1.0 / (1.0 - r2)
            except np.linalg.LinAlgError:
                vif = float("inf")
            vifs.append((name, vif))
        worst_name, worst_vif = max(vifs, key=lambda t: t[1])
        if worst_vif > threshold and len(kept) > 2:
            kept.remove(worst_name)
            changed = True
        else:
            break
    return kept


def daily_cross_section_wls(
    returns: pd.Series,
    exposures: pd.DataFrame,
    factors: list[str],
) -> tuple[float, dict[str, float], pd.Series]:
    """WLS cross-sectional regression for one day.

    r_i = f0 + sum_k x_ik * lambda_k + eps_i

    Returns (f0, {factor: lambda}, residuals indexed by symbol).
    """
    common_syms = returns.index.intersection(exposures.index)
    if len(common_syms) < len(factors) + 2:
        return 0.0, {f: 0.0 for f in factors}, pd.Series(dtype=float)

    y = returns.loc[common_syms].astype(float).values
    X_style = exposures.loc[common_syms, factors].astype(float).values
    X = np.column_stack([np.ones(len(y)), X_style])
    # Drop nan rows
    ok = ~np.any(np.isnan(X), axis=1) & ~np.isnan(y)
    if ok.sum() < len(factors) + 2:
        return 0.0, {f: 0.0 for f in factors}, pd.Series(dtype=float)
    y, X = y[ok], X[ok]
    syms = common_syms[ok]

    try:
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    except np.linalg.LinAlgError:
        return 0.0, {f: 0.0 for f in factors}, pd.Series(dtype=float)

    f0 = float(beta[0])
    lambdas = {f: float(beta[i + 1]) for i, f in enumerate(factors)}
    resid = y - X @ beta
    return f0, lambdas, pd.Series(resid, index=syms)


def attribute_portfolio(
    prices: pd.DataFrame,
    daily_weights: pd.DataFrame,
    *,
    factors: list[str] | None = None,
    calibration_end: date | None = None,
) -> AttributionResult:
    """Attribute portfolio returns over the weight history.

    Parameters
    ----------
    prices : long-form OHLCV with date, symbol, open, high, low, close, volume
    daily_weights : DataFrame with columns date, symbol, weight
        Weights are pre-return (t-1) weights applied to return on day t.
    calibration_end : last date usable for VIF screen (before evaluation)
    """
    prices = prices.copy()
    prices["date"] = pd.to_datetime(prices["date"]).dt.date
    daily_weights = daily_weights.copy()
    daily_weights["date"] = pd.to_datetime(daily_weights["date"]).dt.date

    # Close-to-close returns
    wide = prices.pivot(index="date", columns="symbol", values="close").sort_index()
    rets = wide.pct_change()

    eval_dates = sorted(daily_weights["date"].unique())
    if not eval_dates:
        return AttributionResult(0, 0, 0, 0, [], {})

    # VIF screen on pre-evaluation window
    if factors is None:
        cal_end = calibration_end or (eval_dates[0] - timedelta(days=1))
        cal_exp = compute_style_exposures(prices, cal_end)
        factors = vif_screen(cal_exp) if not cal_exp.empty else list(FACTOR_NAMES)

    daily_rows = []
    sum_common = 0.0
    sum_style = 0.0
    sum_alpha = 0.0
    factor_sum: dict[str, float] = {f: 0.0 for f in factors}

    for t in eval_dates:
        # Need previous trading day for exposures and returns on t
        if t not in rets.index:
            continue
        r_t = rets.loc[t].dropna()
        if r_t.empty:
            continue

        # Exposures from info before t
        exp = compute_style_exposures(prices, t)
        if exp.empty:
            continue
        use_factors = [f for f in factors if f in exp.columns]
        f0, lambdas, resid = daily_cross_section_wls(r_t, exp, use_factors)

        w = daily_weights[daily_weights["date"] == t].set_index("symbol")["weight"]
        # Portfolio return
        common_syms = w.index.intersection(r_t.index)
        if len(common_syms) == 0:
            continue
        w = w.loc[common_syms]
        w = w / w.sum() if w.sum() > 0 else w
        r_p = float((w * r_t.loc[common_syms]).sum())

        # Style contribution: sum_k (sum_i w_i x_ik) * lambda_k
        style = 0.0
        for f in use_factors:
            if f not in exp.columns:
                continue
            port_exp = float((w * exp.reindex(common_syms)[f].fillna(0)).sum())
            contrib = port_exp * lambdas.get(f, 0.0)
            style += contrib
            factor_sum[f] = factor_sum.get(f, 0.0) + contrib

        # Selection alpha
        if len(resid):
            alpha = float((w.reindex(resid.index).fillna(0) * resid).sum())
        else:
            alpha = r_p - f0 - style

        common = f0  # unit exposure
        # Renormalize so Common + Style + Selection = R^p (numerical cleanup)
        # Spec: by construction they sum; use residual for alpha if drift
        residual_gap = r_p - (common + style + alpha)
        alpha += residual_gap

        sum_common += common
        sum_style += style
        sum_alpha += alpha
        daily_rows.append({
            "date": t.isoformat(),
            "portfolio_return": r_p,
            "common": common,
            "style": style,
            "selection_alpha": alpha,
        })

    total = sum_common + sum_style + sum_alpha
    return AttributionResult(
        common=sum_common,
        style=sum_style,
        selection_alpha=sum_alpha,
        total=total,
        daily=daily_rows,
        factor_contrib=factor_sum,
    )


def weights_from_step_artifacts(
    step_artifacts: list[dict],
    ledger_daily: pd.DataFrame,
) -> pd.DataFrame:
    """Extract approximate daily weights from episode artifacts for attribution.

    Prefer `ledger_after` (always real symbols) over observation positions,
    which may be aliased under stock_blind / blinded masks.
    """
    rows = []
    if ledger_daily is None or ledger_daily.empty:
        return pd.DataFrame(columns=["date", "symbol", "weight"])

    # Build per-step weight maps from ledger_after (real tickers)
    step_weights: dict[int, list[tuple[str, float]]] = {}
    for art in step_artifacts or []:
        step = int(art.get("step", 0))
        ledger = art.get("ledger_after") or {}
        nav = float(ledger.get("nav") or 0)
        positions = ledger.get("positions") or {}
        wlist: list[tuple[str, float]] = []
        if isinstance(positions, dict) and nav > 0:
            for sym, p in positions.items():
                if not isinstance(p, dict):
                    continue
                qty = float(p.get("qty") or 0)
                # mark at avg_cost as fallback; better: use observation if unmasked
                avg = float(p.get("avg_cost") or 0)
                # Prefer observation market_value when symbols match (bright)
                mv = qty * avg
                if mv > 0:
                    wlist.append((str(sym), mv / nav))
        # Fallback / refine with observation if symbols look real (not ASSET_)
        obs_pos = (art.get("observation") or {}).get("portfolio", {}).get("positions") or []
        if obs_pos and not any(str(p.get("symbol", "")).startswith("ASSET_") for p in obs_pos):
            wlist = [
                (str(p["symbol"]), float(p.get("weight") or 0))
                for p in obs_pos
                if float(p.get("weight") or 0) > 0
            ]
        elif obs_pos and wlist:
            # Observation is masked; keep ledger-derived weights (real symbols)
            pass
        step_weights[step] = [(s, w) for s, w in wlist if w > 0]

    for _, row in ledger_daily.iterrows():
        d = row["date"]
        if not hasattr(d, "isoformat"):
            d = pd.Timestamp(d).date()
        step = int(row.get("step", 0))
        wlist = step_weights.get(step, [])
        if not wlist:
            continue
        # Renormalize to sum of equity weights (cash is residual)
        total_w = sum(w for _, w in wlist) or 1.0
        for sym, w in wlist:
            rows.append({"date": d, "symbol": sym, "weight": w / total_w})

    return pd.DataFrame(rows)
