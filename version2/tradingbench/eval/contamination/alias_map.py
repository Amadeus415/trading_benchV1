"""Alias map middleware (KTD-Fin primitive, MVP_SPEC §9.1a).

Masking lives at the *tool boundary*, not in the data pipeline:

    model  ──alias──▶  harness  ──un-mask──▶  data layer
    model  ◀──re-mask──  harness  ◀──real──────  data layer

In v1.0 there are no tools, but the map is built now so v1.1 tool use
inherits it. Aliases are stable within an episode and reshuffled across
episodes (via seed).

Exception (documented, not hidden): `asset_class` stays unmasked because
fee/slippage tiers differ and the agent must obey rules.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any


# 2×2 mask factorial (MVP_SPEC §9.1)
MASKS = ("bright", "stock_blind", "date_blind", "blinded")

# Decision modes
DECISION_MODES = ("standard", "memory_only")


def _stable_shuffle(items: list[str], seed: int) -> list[str]:
    """Deterministic permutation of items using seed."""
    keyed = []
    for i, x in enumerate(items):
        h = hashlib.sha256(f"{seed}:{i}:{x}".encode()).hexdigest()
        keyed.append((h, x))
    keyed.sort()
    return [x for _, x in keyed]


@dataclass
class AliasMap:
    """Bidirectional ticker + calendar alias map for one episode."""

    seed: int
    ticker_real_to_alias: dict[str, str] = field(default_factory=dict)
    ticker_alias_to_real: dict[str, str] = field(default_factory=dict)
    episode_start: date | None = None
    date_offset_days: dict[str, int] = field(default_factory=dict)  # iso -> day index
    mask: str = "bright"

    @classmethod
    def build(
        cls,
        symbols: list[str],
        *,
        seed: int,
        episode_start: date | None = None,
        mask: str = "bright",
    ) -> "AliasMap":
        if mask not in MASKS:
            raise ValueError(f"Unknown mask {mask!r}; choose from {MASKS}")

        am = cls(seed=seed, episode_start=episode_start, mask=mask)
        sorted_syms = sorted(set(symbols))
        # Reshuffle alias indices across episodes via seed
        order = _stable_shuffle(sorted_syms, seed)
        for i, real in enumerate(order):
            # ASSET_0417-style opaque handles (4-digit, seed-mixed)
            h = int(hashlib.sha256(f"{seed}:{real}".encode()).hexdigest()[:4], 16)
            alias = f"ASSET_{h:04d}"
            # collision guard
            while alias in am.ticker_alias_to_real:
                h = (h + 1) % 10000
                alias = f"ASSET_{h:04d}"
            am.ticker_real_to_alias[real] = alias
            am.ticker_alias_to_real[alias] = real
            am.ticker_alias_to_real[alias.upper()] = real
        return am

    # --- ticker ---

    def mask_ticker(self, symbol: str) -> str:
        if self.mask in ("bright", "date_blind"):
            return symbol
        return self.ticker_real_to_alias.get(symbol, symbol)

    def unmask_ticker(self, symbol: str) -> str:
        if self.mask in ("bright", "date_blind"):
            return symbol
        return self.ticker_alias_to_real.get(symbol, self.ticker_alias_to_real.get(symbol.upper(), symbol))

    # --- dates ---

    def mask_date(self, d: date | str) -> str:
        """Return ISO date or relative day_+N depending on mask."""
        if isinstance(d, str):
            try:
                d = date.fromisoformat(d[:10])
            except ValueError:
                return d
        if self.mask in ("bright", "stock_blind"):
            return d.isoformat()
        if self.episode_start is None:
            return d.isoformat()
        delta = (d - self.episode_start).days
        return f"day_{delta:+d}"

    def mask_timestamp(self, ts: str | datetime) -> str:
        if self.mask in ("bright", "stock_blind"):
            if isinstance(ts, datetime):
                return ts.strftime("%Y-%m-%dT%H:%M:%SZ")
            return str(ts)
        # date_blind / blinded: strip calendar, keep relative day only
        if isinstance(ts, datetime):
            d = ts.date()
        else:
            try:
                d = date.fromisoformat(str(ts)[:10])
            except ValueError:
                return "day_?"
        return self.mask_date(d)

    # --- observation transform ---

    def apply_to_observation(self, obs: dict[str, Any]) -> dict[str, Any]:
        """Re-mask a fully-built *bright* observation for the agent."""
        out = dict(obs)
        out["mask"] = self.mask
        out["as_of"] = self.mask_date(obs.get("as_of", ""))

        # Market rows: alias tickers; keep numeric features + asset_class
        market = []
        for m in obs.get("market", []):
            mm = dict(m)
            real = m["symbol"]
            mm["symbol"] = self.mask_ticker(real)
            if self.mask in ("stock_blind", "blinded"):
                mm["sector"] = "Unknown"
            market.append(mm)
        out["market"] = market

        # Portfolio positions
        port = dict(obs.get("portfolio", {}))
        positions = []
        for p in port.get("positions", []):
            pp = dict(p)
            pp["symbol"] = self.mask_ticker(p["symbol"])
            positions.append(pp)
        port["positions"] = positions
        out["portfolio"] = port

        # Prior decisions
        prior = []
        for d in obs.get("prior_decisions", []):
            dd = dict(d)
            orders = []
            for o in d.get("orders", []):
                if isinstance(o, str):
                    # "buy NVDA 100" style summary — rewrite ticker tokens
                    s = o
                    for real, alias in self.ticker_real_to_alias.items():
                        if self.mask in ("stock_blind", "blinded"):
                            s = re.sub(rf"\b{re.escape(real)}\b", alias, s, flags=re.I)
                    orders.append(s)
                else:
                    oo = dict(o)
                    if "symbol" in oo:
                        oo["symbol"] = self.mask_ticker(oo["symbol"])
                    orders.append(oo)
            dd["orders"] = orders
            prior.append(dd)
        out["prior_decisions"] = prior

        # Violations
        viols = []
        for v in obs.get("last_step_violations", []):
            vv = dict(v)
            if vv.get("symbol"):
                vv["symbol"] = self.mask_ticker(vv["symbol"])
            viols.append(vv)
        out["last_step_violations"] = viols

        # News: withheld for all contamination masks (spec §9.1)
        if self.mask != "bright":
            out["news"] = []
        else:
            news = []
            for n in obs.get("news", []):
                nn = dict(n)
                if nn.get("symbol"):
                    nn["symbol"] = self.mask_ticker(nn["symbol"])
                if "published_at" in nn:
                    nn["published_at"] = self.mask_timestamp(nn["published_at"])
                news.append(nn)
            out["news"] = news

        out["_alias_map"] = {
            "mask": self.mask,
            "ticker_real_to_alias": dict(self.ticker_real_to_alias),
            "ticker_alias_to_real": {
                k: v for k, v in self.ticker_alias_to_real.items() if k.startswith("ASSET_")
            },
            "seed": self.seed,
            "episode_start": self.episode_start.isoformat() if self.episode_start else None,
        }
        out["_blind_map"] = dict(self.ticker_real_to_alias)
        out["_blind_map_inv"] = {
            k: v for k, v in self.ticker_alias_to_real.items() if k.startswith("ASSET_")
        }
        return out

    def unmask_orders(self, orders: list[dict]) -> list[dict]:
        """Un-mask order symbols before validation/execution."""
        out = []
        for o in orders:
            oo = dict(o)
            if "symbol" in oo:
                oo["symbol"] = self.unmask_ticker(str(oo["symbol"]))
            out.append(oo)
        return out


def memory_only_observation(obs: dict[str, Any]) -> dict[str, Any]:
    """Null-channel control (§9.2a): identifiers only, no factor data.

    Any trade is memorization by construction; abstention rate is the metric.
    """
    out = dict(obs)
    out["decision_mode"] = "memory_only"
    market = []
    for m in obs.get("market", []):
        market.append({
            "symbol": m["symbol"],
            "asset_class": m.get("asset_class", "equity"),
            # deliberately strip all returns/vol/drawdown
            "last": None,
            "ret_1w": None,
            "ret_1m": None,
            "ret_3m": None,
            "vol_20d": None,
            "drawdown_from_52w_high": None,
            "note": "memory_only: no price or factor data provided",
        })
    out["market"] = market
    out["news"] = []
    # Keep portfolio structure (cash/nav/weights) so rules are enforceable,
    # but strip last prices to identifiers + weights only.
    port = dict(obs.get("portfolio", {}))
    positions = []
    for p in port.get("positions", []):
        positions.append({
            "symbol": p["symbol"],
            "qty": p.get("qty"),
            "weight": p.get("weight"),
            "market_value": p.get("market_value"),
            "held_steps": p.get("held_steps"),
            "avg_cost": None,
            "last": None,
            "unrealized_pnl_pct": None,
        })
    port["positions"] = positions
    out["portfolio"] = port
    return out
