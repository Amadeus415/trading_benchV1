"""Non-LLM baselines (MVP_SPEC §8.2).

All long-only, fractional shares, same costs, rebalanced at decision cadence.

| Baseline            | Definition                                              |
|---------------------|---------------------------------------------------------|
| buy_and_hold        | Equal weight at t=0, never rebalanced                   |
| equal_weight_rebal  | Equal weight, rebalanced each step                      |
| momentum_3m         | Rank by trailing 3m return, equal-weight top half       |
| mean_reversion_3m   | Same, bottom half                                       |
| ma_crossover        | Hold when 50d SMA > 100d SMA proxy (ret_3m vs ret proxy)|
| random_agent        | Random weights within rules                             |
| sixty_forty         | 60% equity basket / 40% cash                            |
| momentum_lite       | Legacy 1m top-3 (kept for continuity with v1 runs)      |

`momentum_3m` is the bar that matters. If the best model cannot beat it
out-of-sample, say so plainly.
"""

from __future__ import annotations

import hashlib
from typing import Any, Callable, Protocol

import numpy as np


class Baseline(Protocol):
    def decide(self, observation: dict[str, Any], rng: np.random.Generator) -> dict:
        ...


def _symbols_by_class(obs: dict) -> tuple[list[str], list[str]]:
    equities, crypto = [], []
    for m in obs.get("market", []):
        if m.get("asset_class") == "crypto":
            crypto.append(m["symbol"])
        else:
            equities.append(m["symbol"])
    return equities, crypto


def _held(obs: dict) -> dict[str, float]:
    return {p["symbol"]: float(p.get("market_value") or 0) for p in obs["portfolio"]["positions"]}


def _core_equities(equities: list[str], n: int = 10) -> list[str]:
    preferred = [
        "AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "META", "JPM", "V", "XOM", "JNJ",
        "UNH", "WMT", "PG", "MA", "HD", "CVX", "MRK", "ABBV", "KO", "PEP",
    ]
    core = [s for s in preferred if s in equities]
    if not core:
        # Masked / aliased universe: stable sort so blind arms are comparable
        core = sorted(equities)[:n]
    return core[:n]


def _empty(view: str, risk: str = "") -> dict:
    return {
        "portfolio_view": view,
        "orders": [],
        "changed_view_because": None,
        "risk_note": risk,
    }


def _rebalance_to_targets(
    observation: dict[str, Any],
    targets: dict[str, float],
    *,
    thesis_buy: str,
    thesis_sell: str,
    view: str,
    band: float = 0.02,
) -> dict:
    """Trade toward target weights (fraction of NAV). targets need not sum to 1."""
    nav = max(observation["portfolio"]["nav"], 1e-9)
    cash = observation["portfolio"]["cash"]
    held = _held(observation)
    max_w = float(observation["rules"]["max_position_weight"])
    min_usd = float(observation["rules"].get("min_order_usd", 10.0))

    # Cap individual targets
    targets = {s: min(w, max_w * 0.98) for s, w in targets.items() if w > 0}
    if not targets:
        return _empty(view)

    orders: list[dict] = []
    # Sells first
    for sym, mv in held.items():
        target_mv = targets.get(sym, 0.0) * nav
        delta = mv - target_mv
        if delta >= min_usd and (sym not in targets or delta / nav > band):
            orders.append({
                "symbol": sym,
                "side": "sell",
                "notional_usd": round(min(delta, mv), 2),
                "thesis": thesis_sell,
                "confidence": 0.55,
                "horizon_steps": 4,
            })
            cash += min(delta, mv)

    # Buys
    for sym, tw in targets.items():
        target_mv = tw * nav
        cur = held.get(sym, 0.0)
        need = target_mv - cur
        if need >= min_usd and need / nav > band and cash >= min_usd:
            notional = min(need, cash * 0.98, nav * max_w * 0.98)
            if notional >= min_usd:
                orders.append({
                    "symbol": sym,
                    "side": "buy",
                    "notional_usd": round(notional, 2),
                    "thesis": thesis_buy,
                    "confidence": 0.55,
                    "horizon_steps": 6,
                })
                cash -= notional

    changed = "Rebalance to target weights." if any(o["side"] == "sell" for o in orders) else None
    return {
        "portfolio_view": view,
        "orders": orders,
        "changed_view_because": changed,
        "risk_note": f"{len(targets)} target names.",
    }


def buy_and_hold(observation: dict[str, Any], rng: np.random.Generator | None = None) -> dict:
    """Equal weight at t=0, never rebalanced."""
    step = observation["episode"]["step"]
    cash = observation["portfolio"]["cash"]
    nav = observation["portfolio"]["nav"]
    if step > 0 or cash < 20:
        return _empty(
            "Buy-and-hold baseline: maintain initial equity basket.",
            "No rebalancing.",
        )

    equities, _ = _symbols_by_class(observation)
    core = _core_equities(equities, 10)
    if not core:
        return _empty("No equities available.")

    spend = cash * 0.98
    per = min(spend / len(core), nav * 0.24)
    orders = []
    for s in core:
        if per >= 10:
            orders.append({
                "symbol": s,
                "side": "buy",
                "notional_usd": round(per, 2),
                "thesis": "Buy-and-hold core equity allocation.",
                "confidence": 0.55,
                "horizon_steps": 12,
            })
    return {
        "portfolio_view": f"Deploy capital equally across {len(orders)} core equities; hold thereafter.",
        "orders": orders,
        "changed_view_because": None,
        "risk_note": "Diversified equity BH; no crypto.",
    }


def equal_weight_rebal(observation: dict[str, Any], rng: np.random.Generator | None = None) -> dict:
    """Equal weight across core equities, rebalanced each step."""
    equities, _ = _symbols_by_class(observation)
    core = _core_equities(equities, 10)
    if not core:
        return _empty("Equal-weight rebal: no equities.")
    # Leave ~2% cash
    w = 0.98 / len(core)
    targets = {s: w for s in core}
    return _rebalance_to_targets(
        observation,
        targets,
        thesis_buy="Equal-weight rebalance buy.",
        thesis_sell="Equal-weight rebalance sell.",
        view=f"Equal-weight rebalance across {len(core)} names (2% cash buffer).",
        band=0.015,
    )


def momentum_3m(observation: dict[str, Any], rng: np.random.Generator | None = None) -> dict:
    """Rank by trailing 3-month return; equal-weight top half of equities."""
    equities, _ = _symbols_by_class(observation)
    market = {m["symbol"]: m for m in observation.get("market", [])}
    ranked = []
    for s in equities:
        m = market.get(s, {})
        r = m.get("ret_3m")
        if r is not None:
            ranked.append((s, float(r)))
    if not ranked:
        return _empty("Momentum_3m: no ret_3m available.")
    ranked.sort(key=lambda x: x[1], reverse=True)
    k = max(1, len(ranked) // 2)
    top = [s for s, r in ranked[:k] if r > 0] or [s for s, _ in ranked[:k]]
    # Cap names for position limit
    top = top[:8]
    if not top:
        return _empty("Momentum_3m: no positive momentum names; stay in cash.")
    w = 0.96 / len(top)
    targets = {s: w for s in top}
    return _rebalance_to_targets(
        observation,
        targets,
        thesis_buy="Momentum_3m: top-half trailing 3m return.",
        thesis_sell="Momentum_3m: exit name outside top half.",
        view=f"Momentum_3m long top {len(top)} of {len(ranked)} equities.",
        band=0.02,
    )


def mean_reversion_3m(observation: dict[str, Any], rng: np.random.Generator | None = None) -> dict:
    """Equal-weight bottom half by trailing 3m return (long-only mean reversion)."""
    equities, _ = _symbols_by_class(observation)
    market = {m["symbol"]: m for m in observation.get("market", [])}
    ranked = []
    for s in equities:
        m = market.get(s, {})
        r = m.get("ret_3m")
        if r is not None:
            ranked.append((s, float(r)))
    if not ranked:
        return _empty("Mean_reversion_3m: no ret_3m available.")
    ranked.sort(key=lambda x: x[1])  # worst first
    k = max(1, len(ranked) // 2)
    bottom = [s for s, _ in ranked[:k]][:8]
    if not bottom:
        return _empty("Mean_reversion_3m: empty set.")
    w = 0.96 / len(bottom)
    targets = {s: w for s in bottom}
    return _rebalance_to_targets(
        observation,
        targets,
        thesis_buy="Mean_reversion_3m: bottom-half trailing 3m return.",
        thesis_sell="Mean_reversion_3m: exit name outside bottom half.",
        view=f"Mean_reversion_3m long bottom {len(bottom)} of {len(ranked)} equities.",
        band=0.02,
    )


def ma_crossover(observation: dict[str, Any], rng: np.random.Generator | None = None) -> dict:
    """Hold equities when intermediate trend is up; else cash.

    Proxy for 50d SMA > 100d SMA using available observation features:
    ret_1m > 0 and ret_3m > 0 (both windows positive ⇒ intermediate trend up).
    """
    equities, _ = _symbols_by_class(observation)
    market = {m["symbol"]: m for m in observation.get("market", [])}
    longable = []
    for s in equities:
        m = market.get(s, {})
        r1 = m.get("ret_1m")
        r3 = m.get("ret_3m")
        if r1 is not None and r3 is not None and r1 > 0 and r3 > 0:
            longable.append(s)
    # Prefer liquid core that also passes the filter
    core = [s for s in _core_equities(equities, 12) if s in longable]
    if not core:
        core = longable[:8]
    if not core:
        # All cash
        held = _held(observation)
        orders = []
        for sym, mv in held.items():
            if mv >= 10:
                orders.append({
                    "symbol": sym,
                    "side": "sell",
                    "notional_usd": round(mv, 2),
                    "thesis": "MA crossover: trend down, move to cash.",
                    "confidence": 0.55,
                    "horizon_steps": 2,
                })
        return {
            "portfolio_view": "MA crossover: no names pass trend filter; cash.",
            "orders": orders,
            "changed_view_because": "Trend filter flipped off." if orders else None,
            "risk_note": "100% cash when intermediate trend negative.",
        }
    w = 0.96 / len(core)
    targets = {s: w for s in core}
    return _rebalance_to_targets(
        observation,
        targets,
        thesis_buy="MA crossover: 1m and 3m returns positive.",
        thesis_sell="MA crossover: trend filter failed.",
        view=f"MA crossover long {len(core)} trend-up names.",
        band=0.02,
    )


def sixty_forty(observation: dict[str, Any], rng: np.random.Generator | None = None) -> dict:
    """60% equity basket / 40% cash (bond proxy). Rebalance when drift >5pp."""
    equities, _ = _symbols_by_class(observation)
    held = _held(observation)
    nav = max(observation["portfolio"]["nav"], 1e-9)
    cash = observation["portfolio"]["cash"]
    equity_mv = sum(held.get(s, 0.0) for s in equities)
    equity_w = equity_mv / nav
    target_w = 0.60

    core = _core_equities(equities, 8)
    if not core:
        return _empty("60/40: no equities.")

    if abs(equity_w - target_w) < 0.05 and equity_mv > 0:
        return _empty(
            f"60/40 on target (equity weight {equity_w:.1%}). Hold.",
            "Within 5pp band.",
        )

    target_equity = nav * target_w
    delta = target_equity - equity_mv
    orders: list[dict] = []

    if delta > 15 and cash > 15 and core:
        per = min(delta, cash * 0.98) / len(core)
        per = min(per, nav * 0.24)
        for s in core:
            if per >= 10:
                orders.append({
                    "symbol": s,
                    "side": "buy",
                    "notional_usd": round(per, 2),
                    "thesis": "60/40 rebalance buy into equities.",
                    "confidence": 0.5,
                    "horizon_steps": 8,
                })
    elif delta < -15 and held:
        sell_total = min(-delta, equity_mv * 0.5)
        held_eq = [(s, held[s]) for s in core if held.get(s, 0) >= 10]
        if not held_eq:
            held_eq = [(s, v) for s, v in held.items() if v >= 10]
        if held_eq:
            total = sum(v for _, v in held_eq)
            for s, v in held_eq:
                notional = sell_total * (v / total)
                if notional >= 10:
                    orders.append({
                        "symbol": s,
                        "side": "sell",
                        "notional_usd": round(notional, 2),
                        "thesis": "60/40 rebalance trim equities.",
                        "confidence": 0.5,
                        "horizon_steps": 4,
                    })

    changed = "Rebalance toward 60% equity / 40% cash." if orders else None
    return {
        "portfolio_view": f"60/40 target. Current equity weight {equity_w:.1%}.",
        "orders": orders,
        "changed_view_because": changed,
        "risk_note": "Bond leg proxied by cash in v1.",
    }


def random_agent(observation: dict[str, Any], rng: np.random.Generator) -> dict:
    """Noise trader: random small buys/sells within rules. Seeded for reproducibility."""
    market = observation.get("market", [])
    cash = observation["portfolio"]["cash"]
    nav = observation["portfolio"]["nav"]
    held = _held(observation)
    max_w = observation["rules"]["max_position_weight"]

    if rng.random() > 0.4:
        return _empty("Random agent: no trade this step.", "Idle draw.")

    orders: list[dict] = []
    n_orders = int(rng.integers(1, 4))
    symbols = [m["symbol"] for m in market]
    for _ in range(n_orders):
        if not symbols:
            break
        sym = str(rng.choice(symbols))
        side = "buy"
        if sym in held and rng.random() < 0.4:
            side = "sell"
        if side == "buy":
            budget = min(cash * 0.3, nav * max_w * 0.9)
            if budget < 10:
                continue
            notional = float(rng.uniform(10, max(11, budget)))
            notional = min(notional, cash * 0.95)
            if notional < 10:
                continue
            orders.append({
                "symbol": sym,
                "side": "buy",
                "notional_usd": round(notional, 2),
                "thesis": "Random exploration trade.",
                "confidence": float(rng.uniform(0.3, 0.7)),
                "horizon_steps": int(rng.integers(1, 8)),
            })
            cash -= notional
        else:
            mv = held.get(sym, 0)
            if mv < 10:
                continue
            notional = float(rng.uniform(10, mv))
            orders.append({
                "symbol": sym,
                "side": "sell",
                "notional_usd": round(notional, 2),
                "thesis": "Random exit.",
                "confidence": float(rng.uniform(0.3, 0.7)),
                "horizon_steps": 1,
            })

    return {
        "portfolio_view": f"Random agent placed {len(orders)} order(s).",
        "orders": orders,
        "changed_view_because": "Random rebalance." if any(o["side"] == "sell" for o in orders) else None,
        "risk_note": "Uninformed baseline.",
    }


def momentum_lite(observation: dict[str, Any], rng: np.random.Generator | None = None) -> dict:
    """Legacy: buy top-3 1m momentum names, sell holdings with negative 1m return."""
    market = observation.get("market", [])
    cash = observation["portfolio"]["cash"]
    nav = observation["portfolio"]["nav"]
    held = _held(observation)
    max_w = observation["rules"]["max_position_weight"]

    ranked = sorted(
        [m for m in market if m.get("ret_1m") is not None],
        key=lambda m: m["ret_1m"],
        reverse=True,
    )
    top = [m["symbol"] for m in ranked[:3] if m["ret_1m"] and m["ret_1m"] > 0]
    weak = {m["symbol"] for m in ranked if m.get("ret_1m") is not None and m["ret_1m"] < -0.05}

    orders: list[dict] = []
    for sym, mv in held.items():
        if sym in weak and mv >= 10:
            orders.append({
                "symbol": sym,
                "side": "sell",
                "notional_usd": round(mv, 2),
                "thesis": "Momentum exit: 1m return weak.",
                "confidence": 0.55,
                "horizon_steps": 2,
            })
            cash += mv

    if top and cash >= 30:
        per = min(cash * 0.9 / len(top), nav * max_w * 0.9)
        for sym in top:
            if per >= 10:
                orders.append({
                    "symbol": sym,
                    "side": "buy",
                    "notional_usd": round(per, 2),
                    "thesis": "Momentum entry: strong 1m return.",
                    "confidence": 0.58,
                    "horizon_steps": 4,
                })

    return {
        "portfolio_view": f"Momentum-lite: long top {top}, exit weak names.",
        "orders": orders,
        "changed_view_because": "Momentum signal change." if any(o["side"] == "sell" for o in orders) else None,
        "risk_note": "High turnover rule baseline.",
    }


BASELINES: dict[str, Callable] = {
    "buy_and_hold": buy_and_hold,
    "equal_weight_rebal": equal_weight_rebal,
    "momentum_3m": momentum_3m,
    "mean_reversion_3m": mean_reversion_3m,
    "ma_crossover": ma_crossover,
    "sixty_forty": sixty_forty,
    "random_agent": random_agent,
    "momentum_lite": momentum_lite,
}


def get_baseline(name: str) -> Callable:
    if name not in BASELINES:
        raise KeyError(f"Unknown baseline {name}. Choose from {list(BASELINES)}")
    return BASELINES[name]


def seed_rng(seed: int, step: int = 0) -> np.random.Generator:
    """Deterministic per (episode_seed, step) RNG."""
    h = hashlib.sha256(f"{seed}:{step}".encode()).digest()
    s = int.from_bytes(h[:8], "little")
    return np.random.default_rng(s)
