"""THE ONLY path from data to a prompt.

Agent code must import market data only through this module.
Masking is applied via AliasMap middleware (eval.contamination.alias_map).
"""

from __future__ import annotations

from datetime import date
from typing import Any

import numpy as np
import pandas as pd

from tradingbench.data.store import PointInTimeStore
from tradingbench.eval.contamination.alias_map import (
    AliasMap,
    memory_only_observation,
)
from tradingbench.sim.ledger import Ledger
from tradingbench.sim.validate import Violation


def _returns(closes: pd.Series, days: int) -> float | None:
    if len(closes) < 2:
        return None
    if len(closes) <= days:
        base = closes.iloc[0]
    else:
        base = closes.iloc[-(days + 1)]
    last = closes.iloc[-1]
    if base == 0:
        return None
    return float(last / base - 1.0)


def _vol_20d(closes: pd.Series) -> float | None:
    if len(closes) < 5:
        return None
    rets = closes.pct_change().dropna()
    window = rets.iloc[-20:] if len(rets) >= 20 else rets
    if window.empty:
        return None
    return float(window.std() * np.sqrt(252))


def _drawdown_from_high(closes: pd.Series, lookback: int = 252) -> float | None:
    if closes.empty:
        return None
    window = closes.iloc[-lookback:] if len(closes) >= lookback else closes
    peak = window.max()
    if peak <= 0:
        return None
    return float(window.iloc[-1] / peak - 1.0)


def build_observation(
    store: PointInTimeStore,
    ledger: Ledger,
    step: int,
    total_steps: int,
    prior_decisions: list[dict],
    last_step_violations: list[Violation] | list[dict],
    *,
    mask: str = "bright",
    decision_mode: str = "standard",
    mode: str | None = None,  # legacy alias: standard/named_control/blind/synthetic
    max_position_weight: float = 0.25,
    min_order_usd: float = 10.0,
    news_limit: int = 40,
    lookback_days: int = 100,
    alias_map: AliasMap | None = None,
    episode_seed: int = 0,
    episode_start: date | None = None,
) -> dict[str, Any]:
    """Build the structured observation dict (JSON-serializable).

    Masks (2×2 factorial, §9.1):
      bright       — real tickers, real dates, news allowed
      stock_blind  — alias tickers, real dates, no news
      date_blind   — real tickers, relative dates, no news
      blinded      — alias tickers, relative dates, no news

    Decision modes:
      standard     — full factor data
      memory_only  — identifiers only (null-channel control)

    Legacy `mode` values still accepted and mapped:
      standard → bright+standard, named_control → bright (no news),
      blind → blinded, synthetic → bright (caller supplies synthetic snapshot).
    """
    # Legacy mode mapping
    if mode is not None:
        legacy = {
            "standard": ("bright", "standard"),
            "named_control": ("bright", "standard"),  # news stripped below
            "blind": ("blinded", "standard"),
            "synthetic": ("bright", "standard"),
            "memory_only": ("bright", "memory_only"),
        }
        if mode in legacy:
            mask, decision_mode = legacy[mode]
            if mode == "named_control":
                # force no news via non-bright-like handling
                pass

    as_of = store.as_of
    universe = store.universe()
    symbols = universe["symbol"].tolist()
    asset_class = {r["symbol"]: r["asset_class"] for _, r in universe.iterrows()}
    sector = {r["symbol"]: r["sector"] for _, r in universe.iterrows()}

    prices = store.prices(symbols=symbols, lookback_days=lookback_days)
    last_map: dict[str, float] = {}
    market_rows = []

    for sym in symbols:
        sp = prices[prices["symbol"] == sym].sort_values("date")
        if sp.empty:
            continue
        closes = sp["close"].astype(float)
        last = float(closes.iloc[-1])
        last_map[sym] = last

        ret_1w = _returns(closes, 5)
        ret_1m = _returns(closes, 21)
        ret_3m = _returns(closes, 63)
        vol = _vol_20d(closes)
        dd = _drawdown_from_high(closes)

        market_rows.append({
            "symbol": sym,
            "asset_class": asset_class.get(sym, "equity"),
            "sector": sector.get(sym, "Unknown"),
            "last": round(last, 4),
            "ret_1w": None if ret_1w is None else round(ret_1w, 4),
            "ret_1m": None if ret_1m is None else round(ret_1m, 4),
            "ret_3m": None if ret_3m is None else round(ret_3m, 4),
            "vol_20d": None if vol is None else round(vol, 4),
            "drawdown_from_52w_high": None if dd is None else round(dd, 4),
        })

    # Portfolio (always real symbols internally)
    state = ledger.state(last_map, as_of)
    positions_out = []
    for sym, pos in state.positions.items():
        px = last_map.get(sym, pos.avg_cost)
        mv = pos.market_value(px)
        weight = mv / state.nav if state.nav else 0.0
        positions_out.append({
            "symbol": sym,
            "qty": round(pos.qty, 6),
            "avg_cost": round(pos.avg_cost, 4),
            "last": round(px, 4),
            "market_value": round(mv, 4),
            "weight": round(weight, 4),
            "unrealized_pnl_pct": round(pos.unrealized_pnl_pct(px), 4),
            "held_steps": ledger.held_steps(sym),
        })

    # News: only for bright mask (and not named_control legacy)
    news_out: list[dict] = []
    include_news = mask == "bright" and mode != "named_control"
    if include_news:
        held = set(state.positions.keys())
        news_df = store.news(symbols=symbols, lookback_days=14, limit=news_limit * 2)
        if not news_df.empty:
            def rank_key(row):
                rel = 0 if (pd.notna(row["symbol"]) and row["symbol"] in held) else 1
                return (rel, -pd.Timestamp(row["published_at"]).timestamp())

            news_df = news_df.copy()
            news_df["_rank"] = news_df.apply(rank_key, axis=1)
            news_df = news_df.sort_values("_rank").head(news_limit)
            for _, row in news_df.iterrows():
                sym = row["symbol"] if pd.notna(row["symbol"]) else None
                news_out.append({
                    "published_at": pd.Timestamp(row["published_at"]).strftime("%Y-%m-%dT%H:%M:%SZ"),
                    "symbol": sym,
                    "headline": str(row["headline"]),
                    "summary": str(row["summary"]),
                })

    viols = []
    for v in last_step_violations:
        if isinstance(v, Violation):
            d = v.to_dict()
        else:
            d = dict(v)
        viols.append(d)

    obs: dict[str, Any] = {
        "as_of": as_of.isoformat(),
        "episode": {"step": step, "total_steps": total_steps, "cadence": "weekly"},
        "portfolio": {
            "nav": round(state.nav, 4),
            "cash": round(state.cash, 4),
            "positions": positions_out,
        },
        "market": market_rows,
        "news": news_out,
        "prior_decisions": prior_decisions[-2:],
        "last_step_violations": viols,
        "rules": {
            "max_position_weight": max_position_weight,
            "min_order_usd": min_order_usd,
            "shorting": False,
            "leverage": False,
            "fees_bps": 5,
            "slippage_bps": {"equity": 10, "crypto": 25},
            "fill": "next session open",
            "crypto_aggregate_cap": 0.40,
        },
        "mask": mask,
        "decision_mode": decision_mode,
        "mode": mode or f"{mask}:{decision_mode}",  # legacy field for reports
    }

    # Apply alias map for non-bright masks
    if alias_map is None and mask != "bright":
        alias_map = AliasMap.build(
            symbols,
            seed=episode_seed,
            episode_start=episode_start or as_of,
            mask=mask,
        )
    if alias_map is not None and mask != "bright":
        obs = alias_map.apply_to_observation(obs)
    elif mask == "bright":
        obs["_blind_map"] = {}
        obs["_blind_map_inv"] = {}

    # Memory-only null channel (after masking so identifiers follow mask)
    if decision_mode == "memory_only":
        obs = memory_only_observation(obs)

    return obs


def observation_for_prompt(obs: dict) -> dict:
    """Strip internal keys before sending to the model."""
    return {k: v for k, v in obs.items() if not k.startswith("_")}
