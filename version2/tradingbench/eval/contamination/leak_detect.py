"""Scan reasoning text for post-as_of entity/date mentions."""

from __future__ import annotations

import re
from datetime import date
from typing import Any


# Crude date patterns
_DATE_RE = re.compile(
    r"\b(?:20\d{2})[-/](?:0?[1-9]|1[0-2])[-/](?:0?[1-9]|[12]\d|3[01])\b"
    r"|\b(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|"
    r"Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)"
    r"\s+\d{1,2},?\s+20\d{2}\b",
    re.I,
)


def scan_for_leaks(
    texts: list[str],
    as_of: date,
    known_future_events: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Return LOOKAHEAD_MENTION flags. Does not invalidate a run."""
    flags = []
    known_future_events = known_future_events or []
    for text in texts:
        if not text:
            continue
        for m in _DATE_RE.finditer(text):
            snippet = m.group(0)
            flags.append({
                "code": "LOOKAHEAD_MENTION",
                "detail": f"date-like mention: {snippet}",
                "snippet": text[max(0, m.start() - 40) : m.end() + 40],
            })
        for ev in known_future_events:
            if ev.lower() in text.lower():
                flags.append({
                    "code": "LOOKAHEAD_MENTION",
                    "detail": f"known future event phrase: {ev}",
                    "snippet": ev,
                })
    return flags
