"""Put results/ on the import path, the way make.py and the plotters do.

The results package is imported as ``emfres`` from a script run inside results/,
not installed, so the tests need the same path to import it.

    venv/bin/python -m pytest results/tests
"""

import sys
from pathlib import Path

RESULTS = Path(__file__).resolve().parents[1]
if str(RESULTS) not in sys.path:
    sys.path.insert(0, str(RESULTS))
