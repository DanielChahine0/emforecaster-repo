"""Subcommand dispatch for results/make.py."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import frames

RESULTS = Path(__file__).resolve().parent.parent
REPO = RESULTS.parent
LOGS = REPO / "logs"
FRAMES = RESULTS / "frames"


def cmd_extract(args) -> int:
    logs = args.logs or LOGS
    if not logs.is_dir():
        print(f"not a directory: {logs}", file=sys.stderr)
        return 1

    runs = frames.build_runs(logs, REPO, args.exclude)
    if not runs:
        print(f"no usable log.json under {logs}", file=sys.stderr)
        return 1
    cells = frames.build_cells(runs)

    counts = {"runs": len(runs), "cells": len(cells)}
    per_setup = {}
    for r in runs:
        per_setup[r["setup"]] = per_setup.get(r["setup"], 0) + 1
    counts["runs_by_setup"] = per_setup

    if args.dry_run:
        print(f"{len(runs)} runs, {len(cells)} cells")
        for k, v in sorted(per_setup.items()):
            print(f"  {k:<13} {v:>5}")
        return 0

    frames.write_csv(FRAMES / "runs.csv", runs, frames.RUN_FIELDS)
    frames.write_csv(FRAMES / "cells.csv", cells, frames.CELL_FIELDS)

    if args.epochs:
        epochs = frames.build_epochs(logs, REPO, args.exclude)
        frames.write_csv(FRAMES / "epochs.csv", epochs, frames.EPOCH_FIELDS)
        counts["epochs"] = len(epochs)

    frames.write_manifest(FRAMES / "MANIFEST.json", REPO, runs, counts, args.exclude)

    rel = FRAMES.relative_to(REPO)
    print(f"wrote {rel}/runs.csv    {len(runs)} runs")
    print(f"wrote {rel}/cells.csv   {len(cells)} cells")
    if args.epochs:
        print(f"wrote {rel}/epochs.csv  {counts['epochs']} rows")
    print(f"wrote {rel}/MANIFEST.json")
    for k, v in sorted(per_setup.items()):
        print(f"  {k:<13} {v:>5} runs")
    return 0


def cmd_reconcile(args) -> int:
    from . import reconcile
    cells = FRAMES / "cells.csv"
    if not cells.exists():
        print(f"{cells} not found -- run `make.py extract` first", file=sys.stderr)
        return 1
    return reconcile.reconcile(args.tables, cells)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="make.py", description="Build the results/ frames, tables and figures.")
    sub = parser.add_subparsers(dest="command", required=True)

    ex = sub.add_parser("extract", help="logs/ -> frames/*.csv")
    ex.add_argument("--logs", type=Path, default=None,
                    help=f"logs tree to scan (default: {LOGS})")
    ex.add_argument("--exclude", action="append", default=[], metavar="NAME",
                    help="skip runs with this path component, repeatable")
    ex.add_argument("--epochs", action="store_true",
                    help="also write epochs.csv (per-epoch losses; slow, rereads every log)")
    ex.add_argument("--dry-run", action="store_true", help="print counts, write nothing")
    ex.set_defaults(func=cmd_extract)

    rc = sub.add_parser(
        "reconcile", help="check tables.md's published cells against frames/cells.csv")
    rc.add_argument("--tables", type=Path, default=RESULTS / "tables.md")
    rc.set_defaults(func=cmd_reconcile)

    args = parser.parse_args(argv)
    return args.func(args)
