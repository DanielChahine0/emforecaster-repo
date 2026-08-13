#!/usr/bin/env python3
"""Entry point for the results/ pipeline.

    python results/make.py extract            # logs/ -> frames/*.csv

No install step: this puts results/ on sys.path itself, the way the plotters
already rely on.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from emfres.cli import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
