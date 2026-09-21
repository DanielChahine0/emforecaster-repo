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
SPECS = RESULTS / "specs"
# The frozen report every published value is gated against (make.py tables /
# reconcile). Re-frozen when a set of numbers is accepted as published; the
# previous snapshots stay in archive/ as provenance.
PUBLISHED = RESULTS / "archive" / "2026-09-19_tables.md"


def spec_excludes() -> list[str]:
    """Run-level exclusions declared in specs/runs.yaml, with their reasons."""
    path = SPECS / "runs.yaml"
    if not path.exists():
        return []
    try:
        import yaml
        return [str(v) for v in (yaml.safe_load(path.read_text()) or {}).get("exclude", [])]
    except Exception as exc:
        print(f"WARNING: could not read {path}: {exc}", file=sys.stderr)
        return []


def spec_inferred() -> list[dict]:
    """Commit-inference rules for unstamped runs, from specs/provenance.yaml."""
    path = SPECS / "provenance.yaml"
    if not path.exists():
        return []
    try:
        import yaml
        return list((yaml.safe_load(path.read_text()) or {}).get("inferred", []))
    except Exception as exc:
        print(f"WARNING: could not read {path}: {exc}", file=sys.stderr)
        return []


def cmd_extract(args) -> int:
    logs = args.logs or LOGS
    if not logs.is_dir():
        print(f"not a directory: {logs}", file=sys.stderr)
        return 1

    exclude = list(args.exclude)
    if not args.no_spec:
        spec = spec_excludes()
        if spec:
            print(f"specs/runs.yaml excludes: {', '.join(spec)}", file=sys.stderr)
        exclude += spec

    runs = frames.build_runs(logs, REPO, exclude, inferred=spec_inferred())
    if not runs:
        print(f"no usable log.json under {logs}", file=sys.stderr)
        return 1
    cells = frames.build_cells(runs)

    # Contract C1 (.personal/PLAN.md): a comparison is only as good as the
    # code behind it. Say how many runs predate provenance stamping and
    # whether the stamped ones span more than one commit.
    commits = sorted({r["git_commit"] for r in runs if r["git_commit"]})
    inferred = sum(1 for r in runs if r["commit_source"] == "inferred")
    unstamped = sum(1 for r in runs if not r["git_commit"])
    if inferred or unstamped:
        print(f"NOTE: {inferred} of {len(runs)} runs take their commit from "
              f"specs/provenance.yaml and {unstamped} have none (all logged before "
              f"provenance stamping, 2026-09-19)", file=sys.stderr)
    if len(commits) > 1:
        print(f"WARNING: the published runs span {len(commits)} code commits: "
              + ", ".join(c[:8] for c in commits), file=sys.stderr)

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
        epochs = frames.build_epochs(logs, REPO, exclude)
        frames.write_csv(FRAMES / "epochs.csv", epochs, frames.EPOCH_FIELDS)
        counts["epochs"] = len(epochs)

    frames.write_manifest(FRAMES / "MANIFEST.json", REPO, runs, counts, exclude)

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

    family = args.family
    # The benchmark window every cross-setup section is built for, per corpus.
    setting = (48, 12) if family == "voip" else (336, 96)
    sections = (T.render_sections(runs_csv, REPO, RESULTS, family)
                + T.render_cross_sections(runs_csv, REPO, RESULTS, family, setting))
    # The k-sweep section is per dataset; each family has its own default.
    sweep_dataset = "voip_mobile" if family == "voip" else "ed2022"
    sweep_window = setting
    sweep = T.render_ccm_sweep(FRAMES / "cells.csv", dataset=sweep_dataset,
                               seq_len=sweep_window[0], pred_len=sweep_window[1])
    if sweep:
        sections.append(("ccm-sweep", "Channel Clustering (CCM) vs. plain", sweep))
    if not sections:
        print(f"no runs in family {family!r}", file=sys.stderr)
        return 1

    # Gate: every value the frozen report publishes must still come out of the
    # CSVs. This is deliberately a *value* check, not a text comparison.
    #
    # It used to diff the rendered markdown against the archive and refuse on
    # any difference. That cannot survive a section legitimately gaining rows:
    # restoring the cluster L-ladder added nine lookbacks and reshaped the
    # parameter-count table from one column to ten, changing a great deal of
    # text while leaving every published number identical. A tripwire that
    # fires on correct work teaches you to pass --force, which is worse than
    # having no tripwire. reconcile tolerates growth by construction -- a cell
    # the archive never had is a cell it cannot contradict.
    from . import reconcile as R

    # The frozen snapshot covers the EMF corpus only, so it cannot speak for a
    # VoIP rebuild; those tables are ungated until a snapshot is frozen for them.
    published = PUBLISHED if family == "emf" else None
    if published is None:
        print(f"NOTE: no frozen snapshot for family {family!r} -- values are not "
              f"gated against one", file=sys.stderr)
    elif published.exists():
        if R.reconcile(published, FRAMES / "cells.csv") != 0 and not args.force:
            print(f"\nA published value no longer reproduces from frames/; refusing "
                  f"to write. Pass --force to write anyway.", file=sys.stderr)
            return 1

        # Text drift is reported, never fatal: it says how the report grew.
        import re
        have = {}
        for block in re.split(r"(?m)^(?=# )", published.read_text()):
            if block.startswith("# "):
                have[block.splitlines()[0][2:].strip()] = block.strip()
        for _, title, md in sections:
            old = have.get(title)
            if old is None:
                print(f"NOTE: '{title}' is new -- not in {published.name}", file=sys.stderr)
                continue
            diff = list(unified_diff(old.splitlines(), md.strip().splitlines(),
                                     lineterm="", n=0))
            add = sum(1 for line in diff if line.startswith("+") and not line.startswith("+++"))
            rem = sum(1 for line in diff if line.startswith("-") and not line.startswith("---"))
            if add or rem:
                print(f"NOTE: '{title}' has grown since {published.name} "
                      f"(+{add}/-{rem} lines; every published value still matches)",
                      file=sys.stderr)
    if args.check:
        print(f"\nchecked {len(sections)} sections -- every published value reproduces")
        return 0

    # One directory per corpus: the EMF tables keep their established paths,
    # anything else lands in a sibling named after its family.
    outdir = RESULTS / "tables" if family == "emf" else RESULTS / "tables" / family
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

    twins = T.cells_csv_twins(FRAMES / "cells.csv", outdir / "csv", family)

    index = [banner, "",
             "# Results tables" if family == "emf" else f"# Results tables ({family})",
             "",
             f"Rendered from `frames/runs.csv` (digest `{digest}`), "
             f"family `{family}`.",
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
    ex.add_argument("--no-spec", action="store_true",
                    help="ignore specs/runs.yaml (loads every run, including "
                         "groups excluded there for a documented reason)")
    ex.add_argument("--dry-run", action="store_true", help="print counts, write nothing")
    ex.set_defaults(func=cmd_extract)

    tb = sub.add_parser("tables", help="frames/ -> tables/*.md + csv twins")
    tb.add_argument("--family", default="emf", choices=["emf", "voip"],
                    help="which corpus to render (default: emf). The two share "
                         "every driver and log tree but no axis -- different "
                         "channels, windows and split -- so they are never "
                         "rendered into one table. 'voip' writes tables/voip/.")
    tb.add_argument("--check", action="store_true",
                    help="rebuild and diff against tables.md, write nothing")
    tb.add_argument("--force", action="store_true",
                    help="write even if a section drifted from tables.md")
    tb.set_defaults(func=cmd_tables)

    rc = sub.add_parser(
        "reconcile", help="check tables.md's published cells against frames/cells.csv")
    rc.add_argument("--tables", type=Path, default=PUBLISHED)
    rc.set_defaults(func=cmd_reconcile)

    args = parser.parse_args(argv)
    return args.func(args)
