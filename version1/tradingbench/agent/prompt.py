"""Versioned prompt templates."""

from __future__ import annotations

import hashlib
import json
from typing import Any

from tradingbench.agent.observation import observation_for_prompt

PROMPT_VERSION = "p1"

SYSTEM_PROMPT = """You are a portfolio manager running a $1,000 paper-trading account.
You receive a weekly observation of your portfolio and markets. You must return a single JSON decision.

Rules (enforced by the exchange, not optional):
- Long only. No shorting. No leverage.
- Max 25% of NAV in any single position after a buy.
- Minimum order size $10 notional.
- Orders are notional-denominated (USD), filled at next session open with fees/slippage.
- Empty orders is a valid action (holding is first-class).

confidence is defined exactly as: your stated probability that the position is up
(positive total return) at horizon_steps. Use the full [0, 1] range honestly.

If you close any position opened in a prior step, set changed_view_because to a non-null string.

Respond with JSON only matching the schema. No markdown fences."""


DECISION_SCHEMA = {
    "type": "object",
    "required": ["portfolio_view", "orders"],
    "properties": {
        "portfolio_view": {"type": "string", "maxLength": 600},
        "orders": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["symbol", "side", "notional_usd"],
                "properties": {
                    "symbol": {"type": "string"},
                    "side": {"type": "string", "enum": ["buy", "sell"]},
                    "notional_usd": {"type": "number", "minimum": 0},
                    "thesis": {"type": "string", "maxLength": 300},
                    "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                    "horizon_steps": {"type": "integer", "minimum": 1, "maximum": 12},
                },
            },
        },
        "changed_view_because": {"type": ["string", "null"]},
        "risk_note": {"type": "string", "maxLength": 300},
    },
}


def prompt_hash() -> str:
    payload = SYSTEM_PROMPT + PROMPT_VERSION + json.dumps(DECISION_SCHEMA, sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()[:12]


def render_user_prompt(observation: dict[str, Any]) -> str:
    """Compact JSON observation + short markdown table of market snapshot."""
    obs = observation_for_prompt(observation)
    market = obs.get("market", [])
    # Small table for readability
    lines = [
        f"as_of: {obs['as_of']} | step {obs['episode']['step'] + 1}/{obs['episode']['total_steps']}",
        f"NAV: ${obs['portfolio']['nav']:.2f} | cash: ${obs['portfolio']['cash']:.2f}",
        "",
        "Market snapshot (derived stats, not raw bars):",
        "| symbol | class | last | 1w | 1m | 3m | vol20 | dd52 |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for m in market[:30]:
        def fmt(x):
            if x is None:
                return "—"
            if isinstance(x, float):
                return f"{x:.2%}" if abs(x) < 5 else f"{x:.2f}"
            return str(x)

        lines.append(
            f"| {m['symbol']} | {m['asset_class'][:3]} | {m['last']:.2f} | "
            f"{fmt(m['ret_1w'])} | {fmt(m['ret_1m'])} | {fmt(m['ret_3m'])} | "
            f"{fmt(m['vol_20d'])} | {fmt(m['drawdown_from_52w_high'])} |"
        )

    lines.append("")
    lines.append("Full observation JSON:")
    lines.append(json.dumps(obs, indent=2, default=str))
    lines.append("")
    lines.append(
        "Return JSON: {portfolio_view, orders:[{symbol,side,notional_usd,thesis,confidence,horizon_steps}], "
        "changed_view_because, risk_note}"
    )
    return "\n".join(lines)


def render_repair_prompt(schema_error: str, raw_response: str) -> str:
    return (
        "Your previous response failed schema validation.\n"
        f"Error: {schema_error}\n\n"
        "Previous response:\n"
        f"{raw_response[:2000]}\n\n"
        "Return corrected JSON only. No new market data is provided."
    )
