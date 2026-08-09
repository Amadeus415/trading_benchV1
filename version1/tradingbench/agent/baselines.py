"""Non-LLM baselines: buy-and-hold, 60/40, random agent."""

from __future__ import annotations

import hashlib
from typing import Any, Callable, Protocol

import numpy as np

from tradingbench.sim.validate import Order


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
    return {p["symbol"]: p["market_value"] for p in obs["portfolio"]["positions"]}


def buy_and_hold(observation: dict[str, Any], rng: np.random.Generator | None = None) -> dict:
    """On step 0: equal-weight equities with available cash (leave ~2% cash buffer).
    Thereafter: hold (empty orders).
    """
    step = observation["episode"]["step"]
    cash = observation["portfolio"]["cash"]
    nav = observation["portfolio"]["nav"]
    if step > 0 or cash < 20:
        return {
            "portfolio_view": "Buy-and-hold baseline: maintain initial equity basket.",
            "orders": [],
            "changed_view_because": None,
            "risk_note": "No rebalancing.",
        }

    equities, _ = _symbols_by_class(observation)
    # Prefer a liquid core set
    core = [s for s in ["AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "META", "JPM", "V", "XOM", "JNJ"] if s in equities]
    if not core:
        core = equities[:10]
    if not core:
        return {
            "portfolio_view": "No equities available.",
            "orders": [],
            "changed_view_because": None,
            "risk_note": "",
        }

    spend = cash * 0.98
    per = spend / len(core)
    # respect 25% cap
    per = min(per, nav * 0.24)
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


def sixty_forty(observation: dict[str, Any], rng: np.random.Generator | None = None) -> dict:
    """60% equity basket / 40% cash-like (we use cash as bond proxy in v1).

    Rebalance when equity weight drifts >5pp from 60%.
    """
    equities, _ = _symbols_by_class(observation)
    held = _held(observation)
    nav = max(observation["portfolio"]["nav"], 1e-9)
    cash = observation["portfolio"]["cash"]
    equity_mv = sum(held.get(s, 0.0) for s in equities)
    equity_w = equity_mv / nav
    target_w = 0.60

    core = [s for s in ["AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "JPM", "V", "XOM"] if s in equities]
    if not core:
        core = equities[:8]

    orders = []
    if abs(equity_w - target_w) < 0.05 and equity_mv > 0:
        return {
            "portfolio_view": f"60/40 on target (equity weight {equity_w:.1%}). Hold.",
            "orders": [],
            "changed_view_because": None,
            "risk_note": "Within 5pp band.",
        }

    target_equity = nav * target_w
    delta = target_equity - equity_mv

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
        # sell pro-rata from holdings
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

    orders = []
    # 40% chance to trade at all
    if rng.random() > 0.4:
        return {
            "portfolio_view": "Random agent: no trade this step.",
            "orders": [],
            "changed_view_because": None,
            "risk_note": "Idle draw.",
        }

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
    """Simple rule: buy top-3 1m momentum names, sell holdings with negative 1m return."""
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

    orders = []
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
