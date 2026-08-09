"""THE ONLY path from data to a prompt.

Agent code must import market data only through this module.
"""

from __future__ import annotations

from datetime import date, timedelta
from typing import Any

import numpy as np
import pandas as pd

from tradingbench.data.store import PointInTimeStore
from tradingbench.sim.ledger import Ledger
from tradingbench.sim.validate import Violation


def _returns(closes: pd.Series, days: int) -> float | None:
    if len(closes) < 2:
        return None
    # trading-day approximate lookback
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
    mode: str = "standard",
    max_position_weight: float = 0.25,
    min_order_usd: float = 10.0,
    news_limit: int = 40,
    lookback_days: int = 100,
) -> dict[str, Any]:
    """Build the structured observation dict (JSON-serializable).

    mode:
      standard       — real symbols, news, prices
      named_control  — real symbols, no news, real prices
      blind          — pseudonymous symbols, no news, prices rebased to 100
      synthetic      — same shape as standard (caller supplies synthetic snapshot)
    """
    as_of = store.as_of
    universe = store.universe()
    symbols = universe["symbol"].tolist()
    asset_class = {r["symbol"]: r["asset_class"] for _, r in universe.iterrows()}
    sector = {r["symbol"]: r["sector"] for _, r in universe.iterrows()}

    prices = store.prices(symbols=symbols, lookback_days=lookback_days)
    last_map: dict[str, float] = {}
    market_rows = []

    # Blind symbol map (stable hash-based)
    blind_map: dict[str, str] = {}
    if mode == "blind":
        for i, s in enumerate(sorted(symbols)):
            blind_map[s] = f"ASSET_{i:02d}"

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

        display_sym = blind_map.get(sym, sym)
        display_last = last
        if mode == "blind":
            # rebase so last = 100; returns/vol unchanged in relative terms
            display_last = 100.0

        market_rows.append({
            "symbol": display_sym,
            "asset_class": asset_class.get(sym, "equity"),
            "sector": "Unknown" if mode == "blind" else sector.get(sym, "Unknown"),
            "last": round(display_last, 4),
            "ret_1w": None if ret_1w is None else round(ret_1w, 4),
            "ret_1m": None if ret_1m is None else round(ret_1m, 4),
            "ret_3m": None if ret_3m is None else round(ret_3m, 4),
            "vol_20d": None if vol is None else round(vol, 4),
            "drawdown_from_52w_high": None if dd is None else round(dd, 4),
        })

    # Portfolio
    state = ledger.state(last_map, as_of)
    positions_out = []
    for sym, pos in state.positions.items():
        px = last_map.get(sym, pos.avg_cost)
        mv = pos.market_value(px)
        weight = mv / state.nav if state.nav else 0.0
        display_sym = blind_map.get(sym, sym)
        display_px = 100.0 * (px / last_map[sym]) if mode == "blind" and sym in last_map and last_map[sym] else px
        display_avg = (
            100.0 * (pos.avg_cost / last_map[sym])
            if mode == "blind" and sym in last_map and last_map[sym]
            else pos.avg_cost
        )
        if mode == "blind":
            display_px = 100.0
            # keep avg_cost relative
            display_avg = 100.0 * (pos.avg_cost / px) if px else pos.avg_cost
        positions_out.append({
            "symbol": display_sym,
            "qty": round(pos.qty, 6),
            "avg_cost": round(display_avg, 4),
            "last": round(display_px, 4),
            "market_value": round(mv, 4),
            "weight": round(weight, 4),
            "unrealized_pnl_pct": round(pos.unrealized_pnl_pct(px), 4),
            "held_steps": ledger.held_steps(sym),
        })

    # News
    news_out: list[dict] = []
    if mode in ("standard", "synthetic"):
        held = set(state.positions.keys())
        news_df = store.news(symbols=symbols, lookback_days=14, limit=news_limit * 2)
        if not news_df.empty:
            # rank: portfolio relevance then recency
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
                    "symbol": blind_map.get(sym, sym) if sym else None,
                    "headline": str(row["headline"]),
                    "summary": str(row["summary"]),
                })

    # Prior decisions (last 2)
    prior = prior_decisions[-2:]
    if mode == "blind":
        prior = [_blind_decision(d, blind_map) for d in prior]

    viols = []
    for v in last_step_violations:
        if isinstance(v, Violation):
            d = v.to_dict()
        else:
            d = dict(v)
        if mode == "blind" and d.get("symbol"):
            d["symbol"] = blind_map.get(d["symbol"], d["symbol"])
        viols.append(d)

    obs = {
        "as_of": as_of.isoformat(),
        "episode": {"step": step, "total_steps": total_steps, "cadence": "weekly"},
        "portfolio": {
            "nav": round(state.nav, 4),
            "cash": round(state.cash, 4),
            "positions": positions_out,
        },
        "market": market_rows,
        "news": news_out,
        "prior_decisions": prior,
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
        "mode": mode,
    }
    # Attach reverse map for runner (not shown to model in prompt)
    if mode == "blind":
        obs["_blind_map"] = blind_map
        obs["_blind_map_inv"] = {v: k for k, v in blind_map.items()}
    return obs


def _blind_decision(d: dict, blind_map: dict[str, str]) -> dict:
    out = dict(d)
    orders = []
    for o in d.get("orders", []):
        oo = dict(o)
        if "symbol" in oo:
            oo["symbol"] = blind_map.get(oo["symbol"], oo["symbol"])
        orders.append(oo)
    out["orders"] = orders
    return out


def observation_for_prompt(obs: dict) -> dict:
    """Strip internal keys before sending to the model."""
    return {k: v for k, v in obs.items() if not k.startswith("_")}
