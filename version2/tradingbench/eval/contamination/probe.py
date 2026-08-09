"""Memory probe + de-anonymization attacker certification (MVP_SPEC §9.2).

Memory probe: score 0–3 for post-as_of knowledge.
Attacker probe: given the exact masked payload, try to recover ticker/date.
Report Wilson 95% CIs and a random baseline row.

KTD-Fin reference for a working mask:
  top-1 ≤3.0%, best top-5 10.2%, joint success ≤1.5%.
If joint success materially exceeds 1.5%, blinded results are void.
"""

from __future__ import annotations

import math
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
    for sym in symbols[:5]:
        user = probe_prompt(as_of, sym)
        try:
            comp = client.complete(
                system="You are being tested for knowledge of future market events. Answer honestly.",
                user=user,
                temperature=0.0,
                max_tokens=400,
            )
            text = comp.text
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
    if any(
        p in t
        for p in (
            "i don't recall",
            "i do not recall",
            "no specific",
            "cannot recall",
            "don't know",
        )
    ):
        return 0
    years = [y for y in range(as_of.year, as_of.year + 3)]
    mentions_future_year = any(
        str(y) in text
        for y in years
        if y > as_of.year
        or (
            y == as_of.year
            and any(m in t for m in ("q3", "q4", "october", "november", "december"))
        )
    )
    specific = any(
        w in t for w in ("earnings", "split", "acquisition", "lawsuit", "fda", "halving")
    )
    if mentions_future_year and specific:
        return 3
    if specific:
        return 2
    if any(w in t for w in ("up", "down", "rally", "decline", "bull", "bear")):
        return 1
    return 0


def wilson_ci(successes: int, n: int, z: float = 1.96) -> dict[str, float]:
    """Wilson score interval for a binomial proportion."""
    if n <= 0:
        return {"mean": float("nan"), "lo": float("nan"), "hi": float("nan")}
    p = successes / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    margin = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return {
        "mean": p,
        "lo": max(0.0, centre - margin),
        "hi": min(1.0, centre + margin),
        "n": n,
        "successes": successes,
    }


def attacker_prompt(masked_obs: dict) -> str:
    return (
        "You are an attacker trying to de-anonymize a masked market observation.\n"
        "Given the payload below, guess:\n"
        "1. The real ticker symbol (top-5 list)\n"
        "2. The calendar date of as_of (ISO if possible)\n"
        "3. The sector\n"
        "Respond as JSON: {\"tickers\": [\"...\"], \"date\": \"...\", \"sector\": \"...\"}\n\n"
        f"Payload:\n{masked_obs}"
    )


def score_attacker_response(
    response: dict,
    *,
    true_symbol: str,
    true_date: date,
    true_sector: str | None = None,
    universe_size: int = 30,
) -> dict[str, Any]:
    """Score one attacker attempt."""
    tickers = [str(t).upper() for t in (response.get("tickers") or [])]
    top1 = bool(tickers and tickers[0] == true_symbol.upper())
    top5 = true_symbol.upper() in tickers[:5]
    # date within ±7 trading days ≈ ±10 calendar days
    date_ok = False
    raw_date = response.get("date")
    if raw_date:
        try:
            guessed = date.fromisoformat(str(raw_date)[:10])
            date_ok = abs((guessed - true_date).days) <= 10
        except ValueError:
            date_ok = False
    sector_ok = (
        true_sector is not None
        and str(response.get("sector", "")).lower() == true_sector.lower()
    )
    joint = top5 and date_ok
    return {
        "top1": top1,
        "top5": top5,
        "date_ok": date_ok,
        "sector_ok": sector_ok,
        "joint_success": joint,
    }


def certify_mask(
    results: list[dict[str, Any]],
    *,
    universe_size: int = 30,
) -> dict[str, Any]:
    """Aggregate attacker probe results with Wilson CIs + random baseline.

    results: list of score_attacker_response outputs (parseable only).
    """
    n = len(results)
    if n == 0:
        return {
            "n": 0,
            "note": "no parseable attacker responses",
            "mask_valid": None,
        }

    def rate(key: str) -> dict:
        s = sum(1 for r in results if r.get(key))
        return wilson_ci(s, n)

    top1 = rate("top1")
    top5 = rate("top5")
    date_ok = rate("date_ok")
    joint = rate("joint_success")

    # Random baselines
    random_top1 = 1.0 / max(1, universe_size)
    random_top5 = min(1.0, 5.0 / max(1, universe_size))
    # date within ±10 days over ~3y window rough
    random_date = 21.0 / (365 * 3)
    random_joint = random_top5 * random_date

    # KTD-Fin thresholds
    mask_valid = joint["mean"] <= 0.015 + 1e-9  # ≤1.5%

    return {
        "n": n,
        "top1": top1,
        "top5": top5,
        "date_within_7td": date_ok,
        "joint_success": joint,
        "random_baseline": {
            "top1": random_top1,
            "top5": random_top5,
            "date": random_date,
            "joint": random_joint,
        },
        "ktdfin_reference": {
            "top1_max": 0.03,
            "top5_max": 0.102,
            "joint_max": 0.015,
        },
        "mask_valid": mask_valid,
        "note": (
            "If joint_success materially exceeds 1.5%, blinded results are void "
            "(MVP_SPEC §9.2)."
            if not mask_valid
            else "Mask certificate PASS under KTD-Fin joint ≤1.5% threshold."
        ),
    }
