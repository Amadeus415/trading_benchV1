"""Blind-mode transforms (symbols → ASSET_XX, prices rebased).

Primary blinding is done in agent/observation.py when mode='blind'.
This module exposes helpers for post-hoc analysis and synthetic controls.
"""

from __future__ import annotations

from typing import Any


def make_blind_map(symbols: list[str]) -> dict[str, str]:
    return {s: f"ASSET_{i:02d}" for i, s in enumerate(sorted(symbols))}


def anonymize_observation(obs: dict[str, Any], blind_map: dict[str, str] | None = None) -> dict[str, Any]:
    """Apply blind transform to an already-built standard observation."""
    symbols = [m["symbol"] for m in obs.get("market", [])]
    blind_map = blind_map or make_blind_map(symbols)
    inv = {v: k for k, v in blind_map.items()}

    out = dict(obs)
    out["mode"] = "blind"
    out["news"] = []
    out["market"] = []
    for m in obs.get("market", []):
        mm = dict(m)
        mm["symbol"] = blind_map.get(m["symbol"], m["symbol"])
        mm["sector"] = "Unknown"
        mm["last"] = 100.0
        out["market"].append(mm)

    positions = []
    for p in obs.get("portfolio", {}).get("positions", []):
        pp = dict(p)
        pp["symbol"] = blind_map.get(p["symbol"], p["symbol"])
        last = p.get("last") or 1.0
        pp["last"] = 100.0
        pp["avg_cost"] = 100.0 * (p.get("avg_cost", last) / last) if last else p.get("avg_cost")
        positions.append(pp)
    out["portfolio"] = dict(obs.get("portfolio", {}))
    out["portfolio"]["positions"] = positions
    out["_blind_map"] = blind_map
    out["_blind_map_inv"] = inv
    return out
