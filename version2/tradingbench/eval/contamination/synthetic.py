"""Block-bootstrap synthetic price paths (contamination §9.3)."""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd


def block_bootstrap_prices(
    prices: pd.DataFrame,
    seed: int = 0,
    block_len: int = 10,
) -> pd.DataFrame:
    """Stationary block bootstrap of returns; resample date index jointly across symbols.

    Names and structure preserved; future path is counterfactual.
    """
    rng = np.random.default_rng(seed)
    df = prices.copy()
    df["date"] = pd.to_datetime(df["date"]).dt.date
    dates = sorted(df["date"].unique())
    symbols = sorted(df["symbol"].unique())

    # Pivot closes
    wide = df.pivot(index="date", columns="symbol", values="close").sort_index()
    wide = wide.reindex(dates)
    rets = wide.pct_change().fillna(0.0)
    n = len(dates)
    if n < block_len + 2:
        return prices.copy()

    # Build bootstrapped return index sequence
    indices = []
    while len(indices) < n - 1:
        start = int(rng.integers(0, max(1, n - block_len)))
        indices.extend(range(start, min(start + block_len, n - 1)))
    indices = indices[: n - 1]

    boot_rets = rets.iloc[indices].reset_index(drop=True)
    # Reconstruct prices from first date levels
    levels = wide.iloc[0].values.astype(float)
    path = [levels.copy()]
    for i in range(len(boot_rets)):
        levels = levels * (1.0 + boot_rets.iloc[i].values.astype(float))
        levels = np.maximum(levels, 0.01)
        path.append(levels.copy())
    path_arr = np.array(path[:n])

    rows = []
    for i, d in enumerate(dates):
        for j, sym in enumerate(wide.columns):
            close = float(path_arr[i, j])
            # Synthetic OHLC around close
            open_p = close * (1 + float(rng.normal(0, 0.002)))
            high_p = max(open_p, close) * 1.005
            low_p = min(open_p, close) * 0.995
            rows.append({
                "date": d,
                "symbol": sym,
                "open": round(open_p, 4),
                "high": round(high_p, 4),
                "low": round(low_p, 4),
                "close": round(close, 4),
                "volume": 1e6,
                "currency": "USD",
            })
    return pd.DataFrame(rows)
