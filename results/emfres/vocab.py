"""The one duration grammar shared by the extractor and the plotters.

Three parsers used to live in this tree, over two grammars: the logger writes
``1hr 5min 2s`` / ``43min 10s`` / ``22s``, while the rendered tables carry the
``1h 44m 16s`` spelling ``format_seconds`` emits. Reading a table therefore
needed a different parser from reading a log, and a third copy had drifted into
plot_L_ladder.

One regex now accepts both spellings. It is a strict superset of what each of
the three accepted before, so nothing that parsed stops parsing.
"""

from __future__ import annotations

import re
from typing import Optional

MISSING = "-"

# Longest-first alternation: 'hrs' before 'hr' before 'h', 'min' before 'm'.
_UNIT_ALT = r"hrs|hr|h|min|m|s"
_DURATION_UNITS = {"hrs": 3600.0, "hr": 3600.0, "h": 3600.0,
                   "min": 60.0, "m": 60.0, "s": 1.0}

_DURATION_FULL = re.compile(rf"^\s*(?:\d+(?:\.\d+)?\s*(?:{_UNIT_ALT})\s*)+$", re.I)
_DURATION_PART = re.compile(rf"(\d+(?:\.\d+)?)\s*({_UNIT_ALT})", re.I)


def is_duration(text: str) -> bool:
    """True if ``text`` is entirely a duration, in either spelling."""
    return bool(text) and bool(_DURATION_FULL.match(text))


def parse_duration(raw: Optional[str], *, strict: bool = False) -> Optional[float]:
    """Seconds from a duration string; None for a blank one.

    ``strict`` raises on anything that is not wholly a duration -- the
    extractor wants to hear about a log whose running_time it cannot read,
    where a plotter reading a table cell is happy with None.
    """
    if not raw:
        return None
    text = raw.strip()
    if not _DURATION_FULL.match(text):
        if strict:
            raise ValueError(f"unrecognized running_time format: {raw!r}")
        return None
    return sum(float(v) * _DURATION_UNITS[u.lower()]
               for v, u in _DURATION_PART.findall(text))


def format_seconds(sec: Optional[float]) -> str:
    if sec is None:
        return MISSING
    sec = round(sec)
    h, rem = divmod(sec, 3600)
    m, s = divmod(rem, 60)
    parts = []
    if h:
        parts.append(f"{h}h")
    if h or m:
        parts.append(f"{m}m")
    parts.append(f"{s}s")
    return " ".join(parts)


def format_seconds_fine(sec: Optional[float]) -> str:
    if sec is None:
        return MISSING
    if sec < 60:
        return f"{sec:.1f}s"
    return format_seconds(sec)
