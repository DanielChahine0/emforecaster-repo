#!/usr/bin/env python3
"""Install IBM Plex Sans, the typeface every figure script sets, for matplotlib.

Google Fonts now publishes IBM Plex Sans only as a variable font, and
matplotlib reaches just its default instance - every bold title would quietly
fall back to another family. IBM's own package still ships one file per weight,
as woff, so this fetches those and has fontTools rewrite the container to plain
TTF; the outlines are untouched.

Fonts land in ``~/.local/share/fonts/ibm-plex-sans`` (no root needed). Run once
per machine, then the plot_*.py scripts pick the family up on their own.

Usage
-----
    python3 install_plex_sans.py
"""

from __future__ import annotations

import subprocess
import sys
import urllib.request
from pathlib import Path

from fontTools.ttLib import TTFont

BASE = "https://cdn.jsdelivr.net/npm/@ibm/plex-sans@1.1.0/fonts/complete/woff"
# Regular / Italic / Bold carry the figures; the rest round the family out so a
# future weight change does not need another download.
FACES = [
    "Light", "LightItalic", "Regular", "Italic", "Medium", "MediumItalic",
    "SemiBold", "SemiBoldItalic", "Bold", "BoldItalic",
]
DEST = Path.home() / ".local/share/fonts/ibm-plex-sans"


def main() -> int:
    DEST.mkdir(parents=True, exist_ok=True)
    for face in FACES:
        name = f"IBMPlexSans-{face}"
        source = DEST / f"{name}.woff"
        try:
            with urllib.request.urlopen(f"{BASE}/{name}.woff", timeout=60) as response:
                source.write_bytes(response.read())
        except OSError as error:
            print(f"  {name}: download failed ({error})", file=sys.stderr)
            return 1
        font = TTFont(source)
        font.flavor = None  # woff -> plain TTF
        font.save(DEST / f"{name}.ttf")
        font.close()
        source.unlink()
        print(f"  {name}.ttf")

    subprocess.run(["fc-cache", "-f", str(DEST)], check=False,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    # matplotlib caches the font list; drop it so the new family is seen.
    for stale in (Path.home() / ".cache/matplotlib").glob("fontlist-*.json"):
        stale.unlink()

    from matplotlib import font_manager

    font_manager._load_fontmanager(try_read_cache=False)
    resolved = font_manager.findfont(
        font_manager.FontProperties(family="IBM Plex Sans", weight="bold")
    )
    if "IBMPlexSans" not in resolved:
        print(f"installed, but matplotlib still resolves bold to {resolved}",
              file=sys.stderr)
        return 1
    print(f"\nIBM Plex Sans ready in {DEST}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
