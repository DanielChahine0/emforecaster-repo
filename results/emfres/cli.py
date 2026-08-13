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


def cmd_tables(args) -> int:
    import json
    from difflib import unified_diff

    from . import tables as T

    runs_csv = FRAMES / "runs.csv"
    if not runs_csv.exists():
        print(f"{runs_csv} not found -- run `make.py extract` first", file=sys.stderr)
        return 1

    sections = (T.render_sections(runs_csv, REPO, RESULTS)
                + T.render_cross_sections(runs_csv, REPO, RESULTS))

    # Gate: what we render from the CSVs must match what is published today.
    published = RESULTS / "tables.md"
    if published.exists():
        import re
        have = {}
        for block in re.split(r"(?m)^(?=# )", published.read_text()):
            if block.startswith("# "):
                have[block.splitlines()[0][2:].strip()] = block.strip()
        drift = 0
        for _, title, md in sections:
            old = have.get(title)
            if old is None:
                print(f"NOTE: '{title}' is new -- not in tables.md", file=sys.stderr)
            elif old != md.strip():
                drift += 1
                print(f"DRIFT: '{title}' differs from tables.md", file=sys.stderr)
                for line in list(unified_diff(old.splitlines(), md.strip().splitlines(),
                                              lineterm="", n=0))[:12]:
                    print(f"  {line}", file=sys.stderr)
        if drift and not args.force:
            print(f"\n{drift} section(s) drifted; refusing to write. Pass --force to "
                  f"write anyway.", file=sys.stderr)
            return 1
    if args.check:
        print(f"checked {len(sections)} sections against tables.md -- no drift")
        return 0

    outdir = RESULTS / "tables"
    outdir.mkdir(parents=True, exist_ok=True)
    manifest = FRAMES / "MANIFEST.json"
    digest = "unknown"
    if manifest.exists():
        digest = json.loads(manifest.read_text()).get("runs_digest", "unknown")[:12]
    banner = T.BANNER.format(digest=digest)

    for slug, title, md in sections:
        (outdir / f"{slug}.md").write_text(f"{banner}\n\n{md.strip()}\n")
    (outdir / "all.md").write_text(
        banner + "\n\n" + "\n\n".join(md.strip() for _, _, md in sections) + "\n")

    twins = T.cells_csv_twins(FRAMES / "cells.csv", outdir / "csv")

    index = [banner, "", "# Results tables", "",
             f"Rendered from `frames/runs.csv` (digest `{digest}`).",
             "Every number here is reproducible with `python results/make.py extract`.",
             "", "| Section | Markdown |", "| :--- | :--- |"]
    index += [f"| {title} | [`{slug}.md`]({slug}.md) |" for slug, title, _ in sections]
    index += ["", "Per-setup cell values, one row per aggregated cell:", ""]
    index += [f"- [`csv/{p.name}`](csv/{p.name})" for p in twins]
    (outdir / "index.md").write_text("\n".join(index) + "\n")

    rel = outdir.relative_to(REPO)
    print(f"wrote {len(sections)} sections + all.md + index.md to {rel}/")
    for slug, title, _ in sections:
        print(f"  {slug + '.md':<22} {title}")
    print(f"wrote {len(twins)} csv twins to {rel}/csv/")
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

    tb = sub.add_parser("tables", help="frames/ -> tables/*.md + csv twins")
    tb.add_argument("--check", action="store_true",
                    help="rebuild and diff against tables.md, write nothing")
    tb.add_argument("--force", action="store_true",
                    help="write even if a section drifted from tables.md")
    tb.set_defaults(func=cmd_tables)

    rc = sub.add_parser(
        "reconcile", help="check tables.md's published cells against frames/cells.csv")
    rc.add_argument("--tables", type=Path, default=RESULTS / "tables.md")
    rc.set_defaults(func=cmd_reconcile)

    args = parser.parse_args(argv)
    return args.func(args)
