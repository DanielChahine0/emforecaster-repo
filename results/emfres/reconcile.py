"""Check that every published cell in tables.md is recoverable from frames/.

This is the gate that lets markdown stop being an API: if the CSVs reproduce
the numbers the tables already publish, the renderers can be repointed at the
CSVs without anyone having to trust that they were.

    python results/make.py reconcile
"""

from __future__ import annotations

import csv
import math
import sys
from pathlib import Path

# Section heading in tables.md -> setup slug in frames/.
SECTION_SETUP = {
    "Multivariate": "multivariate",
    "Univariate Series": "series",
    "Univariate (Per Channel)": "perchannel",
    "Cluster (k-Grouped Channels)": "cluster",
}

# Display name in the tables -> the model_id(s) that render to it. Two ids
# share a display name, so a lookup has to accept either.
DISPLAY_IDS = {
    "EMForecaster": ["EMForecaster"],
    "DLinear": ["DLinear"],
    "TSMixer": ["TSMixer"],
    "PatchTST": ["PatchTST"],
    "Transformer": ["Transformer", "RF_EMF_Transformer"],
    "LSTM": ["LSTM", "RF_EMF_LSTM"],
    "ARIMA": ["ARIMA"],
}

# Dataset display -> slug.
DATASET_SLUG = {
    "EMF2023 (University-5 w/ PTV)": "emf_2023_university_5_with_ptv",
    "EMF2023": "emf_2023_university_5_with_ptv",
}


def load_archived(spec: Path) -> list[dict]:
    """Deliberately-unpublished cells declared in specs/runs.yaml.

    A rule matches on (setup, dataset, lookback, horizon); every field it
    omits is a wildcard. Returned as-is so an unreadable spec degrades to "no
    allowances" -- the gate then over-reports rather than under-reports, which
    is the safe direction for a tripwire.
    """
    if not spec.exists():
        return []
    try:
        import yaml
        return list((yaml.safe_load(spec.read_text()) or {}).get("archived", []))
    except Exception as exc:
        print(f"WARNING: could not read {spec}: {exc}", file=sys.stderr)
        return []


def is_archived(rules: list[dict], setup: str, dataset: str,
                lookback: int, horizon: int) -> bool:
    for rule in rules:
        if "setups" in rule and setup not in rule["setups"]:
            continue
        if "dataset" in rule and dataset.upper() != str(rule["dataset"]).upper():
            continue
        if "lookbacks" in rule and lookback not in rule["lookbacks"]:
            continue
        if "horizon" in rule and horizon != rule["horizon"]:
            continue
        return True
    return False


def load_cells(path: Path) -> dict:
    out = {}
    with open(path) as f:
        for row in csv.DictReader(f):
            key = (row["setup"], row["model"], row["dataset"],
                   int(row["seq_len"]), int(row["pred_len"]),
                   row["k"], int(row["conf"]))
            out[key] = row
    return out


def reconcile(tables: Path, cells_csv: Path, spec: Path | None = None) -> int:
    sys.path.insert(0, str(tables.parent))
    import plot_metrics as pm

    cells = load_cells(cells_csv)
    records = pm.parse_tables(tables)
    if spec is None:
        spec = Path(__file__).resolve().parent.parent / "specs" / "runs.yaml"
    archived_rules = load_archived(spec)

    checked = missing = mismatched = archived = 0
    problems: list[str] = []

    skipped_absent = 0
    for r in records:
        setup = SECTION_SETUP.get(r.section)
        if setup is None or r.metric not in ("MSE", "MAE"):
            continue          # training time is a formatted duration, checked separately
        if math.isnan(r.value):
            # A '-' in the table: that grid cell was never run. There should be
            # no frames row either, and it is checked as such.
            skipped_absent += 1
            if any(cells.get((setup, mid, slug, r.lookback, r.horizon,
                              "" if r.cluster is None else str(r.cluster), conf))
                   for mid in DISPLAY_IDS.get(pm.short_model(r.model), [pm.short_model(r.model)])
                   for slug in [DATASET_SLUG.get(r.dataset,
                                DATASET_SLUG.get(pm.short_dataset(r.dataset),
                                                 pm.short_dataset(r.dataset).lower()))]
                   for conf in (1, 0)):
                problems.append(
                    f"  tables.md says '-' but frames has a row: {setup}/"
                    f"{pm.short_model(r.model)}/{pm.short_dataset(r.dataset)} "
                    f"L{r.lookback}_O{r.horizon} [{r.metric}]")
            continue
        model_disp = pm.short_model(r.model)
        ids = DISPLAY_IDS.get(model_disp, [model_disp])
        ds = pm.short_dataset(r.dataset)
        slugs = [DATASET_SLUG.get(r.dataset, DATASET_SLUG.get(ds, ds.lower()))]
        k = "" if r.cluster is None else str(r.cluster)

        row = None
        for mid in ids:
            for slug in slugs:
                for conf in (1, 0):
                    row = cells.get((setup, mid, slug, r.lookback, r.horizon, k, conf))
                    if row:
                        break
                if row:
                    break
            if row:
                break

        if row is None and is_archived(archived_rules, setup, ds,
                                       r.lookback, r.horizon):
            # Published once, deliberately withdrawn since. Not counted as
            # checked: there is nothing left to check it against.
            archived += 1
            continue

        checked += 1
        if row is None:
            missing += 1
            if len(problems) < 15:
                problems.append(
                    f"  no cell for {setup}/{model_disp}/{ds} "
                    f"L{r.lookback}_O{r.horizon}{'' if not k else ' k'+k} [{r.metric}]")
            continue

        got = row["mse"] if r.metric == "MSE" else row["mae"]
        if got == "":
            missing += 1
            continue
        if round(float(got), 4) != round(r.value, 4):
            mismatched += 1
            if len(problems) < 15:
                problems.append(
                    f"  {setup}/{model_disp}/{ds} L{r.lookback}_O{r.horizon}"
                    f"{'' if not k else ' k'+k} [{r.metric}]: "
                    f"tables.md {r.value:.4f} vs frames {float(got):.4f}")

    print(f"checked   {checked} published MSE/MAE cells against frames/cells.csv")
    print(f"absent    {skipped_absent} cells are '-' in tables.md (never run) "
          f"and have no frames row -- as expected")
    if archived or archived_rules:
        print(f"archived  {archived} cells withdrawn by specs/runs.yaml "
              f"'archived' rules -- not checked")
    print(f"matched   {checked - missing - mismatched}")
    print(f"missing   {missing}")
    print(f"mismatch  {mismatched}")
    for p in problems:
        print(p)
    if len(problems) == 15:
        print("  ... (truncated)")
    return 0 if (missing == 0 and mismatched == 0) else 1
