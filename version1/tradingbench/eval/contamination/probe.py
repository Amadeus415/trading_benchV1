"""Memory probe battery (contamination harness §9.2).

In MVP, probes run only when a real LLM client is configured.
Baselines score 0 by construction (no world knowledge).
"""

from __future__ import annotations

from datetime import date
from typing import Any


PROBE_RUBRIC = """
Score 0–3 against ground truth from the snapshot after as_of:
0 = no recall of post-as_of events
1 = vague directional guess
2 = specific direction with rough magnitude
3 = specific dated events that match history
"""


def probe_prompt(as_of: date, symbol: str) -> str:
    return (
        f"As of {as_of.isoformat()}, describe what you expect for {symbol} "
        f"over the following 90 days. If you recall actual events after {as_of.isoformat()}, "
        f"state them explicitly."
    )


def run_memory_probe(
    client,
    as_of: date,
    symbols: list[str],
    *,
    is_baseline: bool = False,
) -> dict[str, Any]:
    """Run probe battery. Baselines get score 0 without calling a model."""
    if is_baseline or client is None:
        return {
            "as_of": as_of.isoformat(),
            "scores": {s: 0 for s in symbols},
            "mean_score": 0.0,
            "method": "baseline_zero",
            "rubric": PROBE_RUBRIC.strip(),
        }

    scores = {}
    raw = {}
    for sym in symbols[:5]:  # cap cost
        user = probe_prompt(as_of, sym)
        try:
            comp = client.complete(
                system="You are being tested for knowledge of future market events. Answer honestly.",
                user=user,
                temperature=0.0,
                max_tokens=400,
            )
            text = comp.text
            # Heuristic scoring without a judge model in MVP:
            # look for year/date patterns after as_of — crude but free.
            score = _heuristic_score(text, as_of)
            scores[sym] = score
            raw[sym] = text[:500]
        except Exception as e:
            scores[sym] = None
            raw[sym] = f"error: {e}"

    valid = [v for v in scores.values() if isinstance(v, (int, float))]
    return {
        "as_of": as_of.isoformat(),
        "scores": scores,
        "mean_score": float(sum(valid) / len(valid)) if valid else None,
        "raw": raw,
        "method": "heuristic_v1",
        "rubric": PROBE_RUBRIC.strip(),
    }


def _heuristic_score(text: str, as_of: date) -> int:
    t = text.lower()
    # If model says it doesn't know / no recall
    if any(p in t for p in ("i don't recall", "i do not recall", "no specific", "cannot recall", "don't know")):
        return 0
    # Dated years after as_of
    years = [y for y in range(as_of.year, as_of.year + 3)]
    mentions_future_year = any(str(y) in text for y in years if y > as_of.year or (
        y == as_of.year and any(m in t for m in ("q3", "q4", "october", "november", "december"))
    ))
    specific = any(w in t for w in ("earnings", "split", "acquisition", "lawsuit", "fda", "halving"))
    if mentions_future_year and specific:
        return 3
    if specific:
        return 2
    if any(w in t for w in ("up", "down", "rally", "decline", "bull", "bear")):
        return 1
    return 0
