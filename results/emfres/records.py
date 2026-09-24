"""frames/cells.csv → the Record list the plotters used to parse out of markdown.

This is what ends markdown-as-API. ``plot_metrics.parse_tables`` matched
regexes like ``^1\\.1\\b.*\\bMSE\\b`` against headings that extract_results
numbers automatically, so inserting one table silently emptied every figure.
The records are now read from the CSVs instead.

Deliberately a drop-in: it produces the same ``Record`` objects, in the same
order, with the same display names and column ranks, so nothing downstream of
``parse_tables`` had to change. ``records_match_tables`` is the equivalence
test that proves it.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path
from typing import Optional

# setup slug -> the section name the plotters key on.
SETUP_SECTIONS = {
    "multivariate": "Multivariate",
    "series": "Univariate Series",
    "perchannel": "Univariate (Per Channel)",
    "cluster": "Cluster (k-Grouped Channels)",
}

# (metric name in Record, column in cells.csv). The metric name is what
# plot_metrics.METRICS keys on, so these are the strings the panels title with.
METRIC_COLUMNS = [("MSE", "mse"), ("MAE", "mae"), ("Training time", "running_time")]

# plot_trustworthiness' three conformal panels.
TRUST_COLUMNS = [("Coverage", "ic"), ("Joint coverage", "jc"),
                 ("Interval width", "interval_width")]

# plot_complexity needs parameter counts beside accuracy and cost.
COMPLEXITY_COLUMNS = METRIC_COLUMNS + [("Parameters", "num_params")]

# A cluster section leads with the fixed train-mean membership, the primary
# cluster protocol (extract_results.render_setup), so a cluster cell's record
# reads the fixed twin of each column that has one. Time and size have no
# protocol. Frames written before the fixed columns existed have no twin to
# read, and their cluster records stay the logged batch protocol, as the
# section then does.
FIXED_TWINS = {"mse": "mse_fixed", "mae": "mae_fixed", "ic": "ic_fixed", "jc": "jc_fixed",
               "interval_width": "interval_width_fixed"}


def _f(v: str) -> Optional[float]:
    return float(v) if v not in ("", None) else None


def records_from_cells(cells_csv: Path, results: Path, metrics=None,
                       family: str = "emf"):
    """Rebuild plot_metrics' Record list from frames/cells.csv.

    ``family`` defaults to the EMF corpus, so every existing figure command
    renders exactly what it did before the VoIP datasets joined the same log
    trees; pass "voip" for those, or None for both at once (which no figure
    currently wants -- the two corpora share no axis).
    """
    metrics = METRIC_COLUMNS if metrics is None else metrics
    sys.path.insert(0, str(results))
    import extract_results as er
    import plot_metrics as pm

    from .frames import family_of

    rows = []
    with open(cells_csv) as f:
        for row in csv.DictReader(f):
            if row["setup"] not in SETUP_SECTIONS:
                continue
            if family is not None and (row.get("family")
                                       or family_of(row["dataset"])) != family:
                continue
            rows.append(row)

    records = []
    for setup, section in SETUP_SECTIONS.items():
        mine = [r for r in rows if r["setup"] == setup]
        if not mine:
            continue
        twins = (FIXED_TWINS if setup == "cluster" and any(r.get("mse_fixed") for r in mine)
                 else {})

        # The tables render one section per conformal flag; the figures only
        # ever read the section that actually got published, which is the one
        # extract_results renders last -- conf=True where it exists.
        for conf in ("1", "0"):
            block = [r for r in mine if r["conf"] == conf]
            if not block:
                continue

            models = er.sort_models({r["model"] for r in block})
            labels = list(dict.fromkeys(er.model_label(m) for m in models))
            # Two model ids can share a display label (RF_EMF_LSTM and LSTM),
            # and the table gives them one column, so index by label.
            row_keys = sorted(
                {(r["dataset"], int(r["seq_len"]), int(r["pred_len"]), r["k"])
                 for r in block},
                key=lambda t: (er.dataset_rank(t[0]), t[1], t[2],
                               -1 if t[3] == "" else int(t[3])),
            )
            by_key = {}
            for r in block:
                key = (r["dataset"], int(r["seq_len"]), int(r["pred_len"]), r["k"],
                       er.model_label(r["model"]))
                by_key[key] = r

            for metric, column in metrics:
                for ds, sl, pl, k in row_keys:
                    for col, label in enumerate(labels):
                        r = by_key.get((ds, sl, pl, k, label))
                        value = _f(r.get(twins.get(column, column), "")) if r else None
                        records.append(pm.Record(
                            section=section,
                            metric=metric,
                            dataset=er.dataset_label(ds),
                            lookback=sl,
                            horizon=pl,
                            model=label,
                            model_order=(pm.model_rank(label), col),
                            value=float("nan") if value is None else value,
                            cluster=None if k == "" else int(k),
                        ))
            break     # only the published conformal block feeds the figures
    return records


def _key(r):
    return (r.section, r.metric, r.dataset, r.lookback, r.horizon, r.model,
            r.cluster, r.model_order)


def records_match_tables(cells_csv: Path, tables_md: Path, results: Path):
    """Compare CSV-derived records against the markdown-derived ones.

    Returns (n_compared, list of human-readable differences).
    """
    import math

    sys.path.insert(0, str(results))
    import plot_metrics as pm

    want = {_key(r): r.value for r in pm.parse_tables(tables_md)}
    got = {_key(r): r.value for r in records_from_cells(cells_csv, results)}

    diffs = []
    for k in sorted(set(want) | set(got), key=lambda t: [str(x) for x in t]):
        a, b = want.get(k), got.get(k)
        if a is None:
            diffs.append(f"only in frames: {k[:6]}")
        elif b is None:
            diffs.append(f"only in tables.md: {k[:6]}")
        elif math.isnan(a) and math.isnan(b):
            continue
        elif math.isnan(a) != math.isnan(b):
            diffs.append(f"{k[:6]}: tables.md {a} vs frames {b}")
        elif round(a, 6) != round(b, 6):
            diffs.append(f"{k[:6]}: tables.md {a} vs frames {b}")
    return len(want), diffs
