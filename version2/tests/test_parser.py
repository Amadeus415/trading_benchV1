"""Decision parser tests."""

from __future__ import annotations

from tradingbench.agent.parser import parse_decision


def test_valid_decision():
    raw = """
    {
      "portfolio_view": "Staying diversified across tech leaders.",
      "orders": [
        {"symbol": "AAPL", "side": "buy", "notional_usd": 100.0,
         "thesis": "Solid cash flows and ecosystem.", "confidence": 0.6, "horizon_steps": 4}
      ],
      "changed_view_because": null,
      "risk_note": "Watch rates."
    }
    """
    d, v = parse_decision(raw)
    assert d is not None
    assert not v
    assert len(d.orders) == 1
    assert d.orders[0].symbol == "AAPL"


def test_malformed_json():
    d, v = parse_decision("not json at all")
    assert d is None
    assert any(x.code == "MALFORMED" for x in v)


def test_requires_changed_view_on_close():
    raw = """
    {
      "portfolio_view": "Exiting.",
      "orders": [
        {"symbol": "NVDA", "side": "sell", "notional_usd": 50.0, "thesis": "trim", "confidence": 0.5, "horizon_steps": 1}
      ],
      "changed_view_because": null
    }
    """
    d, v = parse_decision(raw, prior_open_symbols={"NVDA"})
    assert d is None
    assert any(x.code == "MALFORMED" for x in v)


def test_fenced_json():
    raw = """```json
    {"portfolio_view": "Hold.", "orders": [], "changed_view_because": null, "risk_note": ""}
    ```"""
    d, v = parse_decision(raw)
    assert d is not None
    assert d.orders == []
