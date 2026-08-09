"""Blind-mode transforms — thin re-export of the AliasMap primitive.

Primary masking is done via `alias_map.AliasMap` at the observation boundary
(agent/observation.py). This module keeps the historical helpers used by
analysis scripts and synthetic-control post-processing.
"""

from __future__ import annotations

from typing import Any

from tradingbench.eval.contamination.alias_map import (
    MASKS,
    AliasMap,
    memory_only_observation,
)


def make_blind_map(symbols: list[str], seed: int = 0) -> dict[str, str]:
    """Stable-within-call alias map (prefer AliasMap.build for episodes)."""
    am = AliasMap.build(symbols, seed=seed, mask="blinded")
    return dict(am.ticker_real_to_alias)


def anonymize_observation(
    obs: dict[str, Any],
    blind_map: dict[str, str] | None = None,
    *,
    mask: str = "blinded",
    seed: int = 0,
) -> dict[str, Any]:
    """Apply a mask transform to an already-built bright observation."""
    symbols = [m["symbol"] for m in obs.get("market", [])]
    if blind_map is not None:
        am = AliasMap(seed=seed, mask=mask)
        am.ticker_real_to_alias = dict(blind_map)
        am.ticker_alias_to_real = {v: k for k, v in blind_map.items()}
    else:
        from datetime import date

        as_of = obs.get("as_of")
        try:
            start = date.fromisoformat(str(as_of)[:10]) if as_of else None
        except ValueError:
            start = None
        am = AliasMap.build(symbols, seed=seed, episode_start=start, mask=mask)
    return am.apply_to_observation(obs)


__all__ = [
    "MASKS",
    "AliasMap",
    "make_blind_map",
    "anonymize_observation",
    "memory_only_observation",
]
