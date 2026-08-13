#!/usr/bin/env python3
"""Build results/clustering/tables.md: plain vs. CCM, one table, one dataset.

    python3 results/clustering/make_tables.py
    python3 results/clustering/make_tables.py --dataset uh2022 --metric mse

By default it reports ED2022 at L336/O96 over MAE as a single table: one row per
cluster count k = 1..9 and a (plain, + CCM) column pair per model, so each row
reads as DLinear vs. DLinear + CCM at that k, and so on for PatchTST and TSMixer.

The plain columns come from logs/multivariate -- the un-clustered joint model,
which has no k, so its value repeats down the column as the fixed reference each
row is compared against. The + CCM columns come from logs/cluster. Note that the
k = 1 CCM cell is NOT the plain model: it is one cluster over all channels, but
the run still carries the CCM module's parameters.

Cells average the seed replicates of a run directory; where a cell has more than
one seed the sample spread is printed next to the mean, because on this grid the
k-to-k differences are small enough that seed noise is the thing to compare them
against.
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CLUSTER_ROOT = REPO_ROOT / "logs" / "cluster"
DEFAULT_PLAIN_ROOT = REPO_ROOT / "logs" / "multivariate"
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "tables.md"

# The models this report compares, in column order.
MODELS = ["DLinear", "PatchTST", "TSMixer"]
DATASET_DISPLAY = {"ed2022": "ED2022", "uh2022": "UH2022",
                   "emf2023": "EMF2023", "emf_2023_university_5_with_ptv": "EMF2023"}
METRIC_KEY = {"mae": "sl_test/mae", "mse": "sl_test/loss"}
METRIC_DISPLAY = {"mae": "MAE", "mse": "MSE"}

MISSING = "-"
# The run-dir tag forecasting/grid.py stamps on a cluster cell (exp.id = "k3").
_K_RE = re.compile(r"^k(\d+)$")


@dataclass
class Run:
    model: str
    k: Optional[int]  # None for a plain (un-clustered) run
    seed: Optional[str]
    value: Optional[float]
    path: Path


@dataclass
class Cell:
    mean: float
    spread: Optional[float]  # sample stdev, None for a single-seed cell
    n_seeds: int


def load_runs(logs_root: Path, dataset: str, seq_len: int, pred_len: int,
              models: list[str], metric: str, clustered: bool) -> list[Run]:
    """Every matching run under logs_root, tagged with its k (None when plain)."""
    key = METRIC_KEY[metric]
    runs = []
    for log_path in sorted(logs_root.rglob("log.json")):
        with open(log_path) as f:
            d = json.load(f)

        model = d.get("parameters/exp/model_id")
        dataset_raw = d.get("parameters/data/dataset")
        if model not in models or dataset_raw is None or dataset_raw.lower() != dataset:
            continue
        if d.get("parameters/data/seq_len") != seq_len or d.get("parameters/data/pred_len") != pred_len:
            continue

        # The logged k wins over the path tag; k_clusters is only meaningful when
        # cluster.enabled, since it defaults to 1 on every ordinary run.
        k = next((int(m.group(1)) for part in log_path.parts
                  if (m := _K_RE.fullmatch(part))), None)
        enabled = bool(d.get("parameters/cluster/enabled"))
        if enabled:
            k = d.get("parameters/cluster/k_clusters", k)

        if clustered and not enabled and k is None:
            print(f"WARNING: skipping {log_path} (no cluster config and no 'k<N>' path tag)",
                  file=sys.stderr)
            continue
        if not clustered:
            if enabled or k is not None:
                print(f"WARNING: skipping clustered run {log_path} found under the plain tree",
                      file=sys.stderr)
                continue
            k = None

        v = d.get(key)
        value = v[0] if v else None
        if value is None:
            print(f"WARNING: {log_path} has no {key}", file=sys.stderr)

        seed = log_path.parent.name if log_path.parent.name.isdigit() else None
        runs.append(Run(model, k, seed, value, log_path))
    return runs


def aggregate(runs: list[Run]) -> dict[tuple[str, Optional[int]], Cell]:
    groups: dict[tuple[str, Optional[int]], list[float]] = defaultdict(list)
    for r in runs:
        if r.value is not None:
            groups[(r.model, r.k)].append(r.value)
    return {key: Cell(mean=statistics.mean(vals),
                      spread=statistics.stdev(vals) if len(vals) > 1 else None,
                      n_seeds=len(vals))
            for key, vals in groups.items()}


def fmt_value(cell: Optional[Cell]) -> str:
    if cell is None:
        return MISSING
    if cell.spread is None:
        return f"{cell.mean:.4f}"
    return f"{cell.mean:.4f} ±{cell.spread:.4f}"


def fmt_delta(cell: Optional[Cell], base: Optional[Cell]) -> str:
    if cell is None or base is None or base.mean == 0:
        return MISSING
    d = cell.mean - base.mean
    return f"{d:+.4f} ({d / base.mean * 100:+.2f}%)"


def table(header: list[str], rows: list[list[str]]) -> str:
    align = ["|:---"] + ["|---:"] * (len(header) - 1)
    lines = ["| " + " | ".join(header) + " |", "".join(align) + "|"]
    lines += ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join(lines)


def seed_note(cells: dict[tuple[str, Optional[int]], Cell], model: str) -> str:
    counts = sorted({c.n_seeds for (m, _), c in cells.items() if m == model})
    return "/".join(str(n) for n in counts) if counts else MISSING


def render(plain: dict[tuple[str, Optional[int]], Cell],
           ccm: dict[tuple[str, Optional[int]], Cell],
           n_runs: int, dataset: str, seq_len: int, pred_len: int,
           models: list[str], metric: str) -> str:
    ks = sorted({k for _, k in ccm if k is not None})
    metric_name = METRIC_DISPLAY[metric]
    dataset_name = DATASET_DISPLAY.get(dataset, dataset.upper())

    out = [f"# Channel Clustering (CCM) vs. plain · {dataset_name} · {metric_name}",
           "",
           f"- Dataset: {dataset_name}, $L$ = {seq_len}, $O$ = {pred_len}",
           f"- Metric: {metric_name} (`{METRIC_KEY[metric]}`); `±` is the sample stdev across seeds",
           f"- Runs: {n_runs} logs. Seed replicates per cell: "
           + ", ".join(f"{m} {seed_note(plain, m)} plain / {seed_note(ccm, m)} + CCM"
                       for m in models),
           "",
           "Each row compares a model against itself with CCM at that $k$; **bold** = the lower "
           "of the pair, <u>underline</u> = that model's best $k$. The plain columns have no $k$, "
           "so their single value repeats down the column as the fixed reference.",
           ""]

    best_k = {m: min((k for k in ks if (m, k) in ccm),
                     key=lambda k: ccm[(m, k)].mean, default=None)
              for m in models}
    rows = []
    for k in ks:
        row = [f"{k}"]
        for m in models:
            base, cell = plain.get((m, None)), ccm.get((m, k))
            base_txt, cell_txt = fmt_value(base), fmt_value(cell)
            if best_k[m] == k:
                cell_txt = f"<u>{cell_txt}</u>"
            if base is not None and cell is not None:
                if cell.mean < base.mean:
                    cell_txt = f"**{cell_txt}**"
                else:
                    base_txt = f"**{base_txt}**"
            row += [base_txt, cell_txt]
        rows.append(row)

    header = ["$k$"]
    for m in models:
        header += [m, f"{m} + CCM"]
    out += [table(header, rows), ""]

    # The one thing the table cannot show: how each model's best k lands against
    # its plain run, and whether that margin clears the seed noise it sits in.
    out += ["**Best $k$ per model**, against its plain run:", ""]
    for m in models:
        k = best_k[m]
        if k is None:
            out.append(f"- {m}: no CCM cell.")
            continue
        cell, base = ccm[(m, k)], plain.get((m, None))
        spreads = [c.spread for (mm, kk), c in ccm.items()
                   if mm == m and kk is not None and c.spread is not None]
        noise = (f", mean seed stdev over the sweep {statistics.mean(spreads):.4f}"
                 if spreads else ", single seed per cell")
        out.append(f"- {m}: best at $k$ = {k}, {metric_name} {cell.mean:.4f} vs. plain "
                   f"{fmt_value(base)} ({fmt_delta(cell, base)}){noise}.")
    out += ["",
            f"$k$ = 1 is one cluster over all channels, but its run still carries the CCM "
            f"module's parameters, so it is not the same model as the plain column.",
            ""]
    return "\n".join(out)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cluster-root", type=Path, default=DEFAULT_CLUSTER_ROOT,
                   help=f"CCM logs tree (default: {DEFAULT_CLUSTER_ROOT})")
    p.add_argument("--plain-root", type=Path, default=DEFAULT_PLAIN_ROOT,
                   help=f"un-clustered logs tree (default: {DEFAULT_PLAIN_ROOT})")
    p.add_argument("--output", type=Path, default=DEFAULT_OUTPUT,
                   help=f"markdown file to write (default: {DEFAULT_OUTPUT})")
    p.add_argument("--dataset", default="ed2022", help="dataset to report (default: ed2022)")
    p.add_argument("--seq-len", type=int, default=336, help="lookback L (default: 336)")
    p.add_argument("--pred-len", type=int, default=96, help="horizon O (default: 96)")
    p.add_argument("--models", nargs="+", default=MODELS,
                   help=f"models, in column order (default: {' '.join(MODELS)})")
    p.add_argument("--metric", choices=sorted(METRIC_KEY), default="mae",
                   help="metric to tabulate (default: mae)")
    args = p.parse_args()

    for root in (args.cluster_root, args.plain_root):
        if not root.is_dir():
            sys.exit(f"no such logs tree: {root}")

    dataset = args.dataset.lower()
    window = (dataset, args.seq_len, args.pred_len, args.models, args.metric)
    ccm_runs = load_runs(args.cluster_root, *window, clustered=True)
    plain_runs = load_runs(args.plain_root, *window, clustered=False)
    if not ccm_runs:
        sys.exit(f"no CCM runs under {args.cluster_root} for {dataset} "
                 f"L{args.seq_len}/O{args.pred_len} and models {args.models}")

    ccm, plain = aggregate(ccm_runs), aggregate(plain_runs)
    ks = sorted({k for _, k in ccm if k is not None})
    for m in args.models:
        missing = [k for k in ks if (m, k) not in ccm]
        if missing:
            print(f"WARNING: {m} has no CCM cell for k = {', '.join(str(k) for k in missing)}",
                  file=sys.stderr)
        if (m, None) not in plain:
            print(f"WARNING: no plain {m} run for {dataset} L{args.seq_len}/O{args.pred_len}; "
                  f"its column will be '{MISSING}'", file=sys.stderr)

    args.output.write_text(render(plain, ccm, len(ccm_runs) + len(plain_runs), dataset,
                                  args.seq_len, args.pred_len, args.models, args.metric))
    print(f"Wrote {args.output} ({len(ccm_runs)} CCM + {len(plain_runs)} plain runs).",
          file=sys.stderr)


if __name__ == "__main__":
    main()
