"""Parse model response → Decision. Strict schema; one repair path; never silent coerce."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

from jsonschema import Draft202012Validator

from tradingbench.agent.prompt import DECISION_SCHEMA
from tradingbench.sim.validate import Order, Violation


@dataclass
class Decision:
    portfolio_view: str
    orders: list[Order]
    changed_view_because: str | None = None
    risk_note: str = ""
    raw: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "portfolio_view": self.portfolio_view,
            "orders": [o.to_dict() for o in self.orders],
            "changed_view_because": self.changed_view_because,
            "risk_note": self.risk_note,
        }

    def thesis_summary(self) -> str:
        return (self.portfolio_view or "")[:200]


def _strip_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    return text.strip()


def _extract_json(text: str) -> Any:
    text = _strip_fences(text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # try to find first {...} block
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            return json.loads(text[start : end + 1])
        raise


_validator = Draft202012Validator(DECISION_SCHEMA)


def validate_schema(data: dict) -> str | None:
    errors = sorted(_validator.iter_errors(data), key=lambda e: list(e.path))
    if not errors:
        return None
    e = errors[0]
    path = ".".join(str(p) for p in e.path) or "<root>"
    return f"{path}: {e.message}"


def parse_decision(
    raw_text: str,
    *,
    prior_open_symbols: set[str] | None = None,
    blind_map_inv: dict[str, str] | None = None,
) -> tuple[Decision | None, list[Violation]]:
    """Parse and validate. Returns (Decision|None, violations).

    On failure returns (None, [MALFORMED ...]).
    """
    violations: list[Violation] = []
    try:
        data = _extract_json(raw_text)
    except (json.JSONDecodeError, ValueError) as e:
        violations.append(
            Violation(code="MALFORMED", symbol=None, detail=f"JSON parse error: {e}")
        )
        return None, violations

    if not isinstance(data, dict):
        violations.append(
            Violation(code="MALFORMED", symbol=None, detail="response is not a JSON object")
        )
        return None, violations

    err = validate_schema(data)
    if err:
        violations.append(Violation(code="MALFORMED", symbol=None, detail=f"schema: {err}"))
        return None, violations

    orders: list[Order] = []
    for o in data.get("orders") or []:
        try:
            order = Order.from_dict(o)
        except (KeyError, TypeError, ValueError) as e:
            violations.append(
                Violation(code="MALFORMED", symbol=o.get("symbol"), detail=f"order: {e}")
            )
            return None, violations
        # unblind symbols if needed
        if blind_map_inv and order.symbol in blind_map_inv:
            order = Order(
                symbol=blind_map_inv[order.symbol],
                side=order.side,
                notional_usd=order.notional_usd,
                thesis=order.thesis,
                confidence=order.confidence,
                horizon_steps=order.horizon_steps,
            )
        elif blind_map_inv and order.symbol.upper() in blind_map_inv:
            real = blind_map_inv[order.symbol.upper()]
            order = Order(
                symbol=real,
                side=order.side,
                notional_usd=order.notional_usd,
                thesis=order.thesis,
                confidence=order.confidence,
                horizon_steps=order.horizon_steps,
            )
        orders.append(order)

    # Enforce changed_view_because when closing prior positions
    prior_open_symbols = prior_open_symbols or set()
    closing = {
        o.symbol for o in orders if o.side == "sell" and o.symbol in prior_open_symbols
    }
    changed = data.get("changed_view_because")
    if closing and (changed is None or (isinstance(changed, str) and not changed.strip())):
        violations.append(
            Violation(
                code="MALFORMED",
                symbol=next(iter(closing)),
                detail="changed_view_because required when closing a prior position",
            )
        )
        return None, violations

    decision = Decision(
        portfolio_view=str(data.get("portfolio_view") or "")[:600],
        orders=orders,
        changed_view_because=changed,
        risk_note=str(data.get("risk_note") or "")[:300],
        raw=data,
    )
    return decision, violations
