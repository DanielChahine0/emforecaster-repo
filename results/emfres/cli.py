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
# The (L, O) windows each family's paradigm comparison and cluster sweep are
# rendered at, the benchmark first. The EMF grids ran L336/O96 only (the other
# windows live in logs/_archive); the VoIP grids ran L48/O12 and L24/O6.
WINDOWS = {"emf": [(336, 96)], "voip": [(48, 12), (24, 6)]}

# The frozen report every published value is gated against (make.py tables /
# reconcile). Re-frozen when a set of numbers is accepted as published; the
# previous snapshots stay in archive/ as provenance.
PUBLISHED = RESULTS / "archive" / "2026-09-22_tables.md"


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


def _provenance_rules(section: str) -> list[dict]:
    path = SPECS / "provenance.yaml"
    if not path.exists():
        return []
    try:
        import yaml
        return list((yaml.safe_load(path.read_text()) or {}).get(section, []) or [])
    except Exception as exc:
        print(f"WARNING: could not read {path}: {exc}", file=sys.stderr)
        return []


def spec_inferred() -> list[dict]:
    """Commit-inference rules for unstamped runs, from specs/provenance.yaml."""
    return _provenance_rules("inferred")


def spec_attested() -> list[dict]:
    """Code attestations for runs stamped on a dirty tree, from specs/provenance.yaml."""
    return _provenance_rules("attested")


def spec_equivalent() -> list[dict]:
    """Code-version equivalences, from specs/provenance.yaml.

    Refused rules -- no evidence, a version mapped to two others, a cycle
    (runlogs.equivalence_table) -- are warned about and left out.
    """
    from .runlogs import equivalence_table, valid_equivalence

    rules = _provenance_rules("equivalent_code")
    table, problems = equivalence_table(rules)
    for problem in problems:
        print(f"WARNING: specs/provenance.yaml equivalent_code: {problem}; ignoring it",
              file=sys.stderr)
    return [r for r in rules if valid_equivalence(r)
            and table.get(str(r["from"]).strip()) == str(r["to"]).strip()]


def spec_timing() -> dict:
    """The solo per-channel timing pass, from specs/timing.yaml ({} if absent)."""
    path = SPECS / "timing.yaml"
    if not path.exists():
        return {}
    try:
        import yaml
        return (yaml.safe_load(path.read_text()) or {}).get("solo_pass") or {}
    except Exception as exc:
        print(f"WARNING: could not read {path}: {exc}", file=sys.stderr)
        return {}


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

    loaded = frames.load_all(logs, exclude, inferred=spec_inferred(),
                             attested=spec_attested())
    runs = frames.build_runs(logs, REPO, exclude, equivalent=spec_equivalent(),
                             loaded=loaded)
    if not runs:
        print(f"no usable log.json under {logs}", file=sys.stderr)
        return 1
    cells = frames.build_cells(runs)
    bands = frames.build_bands(loaded)
    timing = frames.build_timing(runs, spec_timing())

    # Contract C1 (.personal/PLAN.md): a comparison is only as good as the
    # code behind it. Say how many runs predate provenance stamping and
    # whether the stamped ones span more than one commit.
    # Judged on the training code, not the commit id: commits that differ only
    # in results/, logs/ or docs share a code version (frames.code_fingerprint),
    # so committing results work while a sweep runs does not split it. And
    # judged after specs/provenance.yaml's equivalent_code rules: a version
    # declared equivalent, with evidence, to another is not a split -- it is
    # listed, and MANIFEST.json records it, but it is not warned about.
    versions: dict = {}
    for r in runs:
        if r["code_version"]:
            versions.setdefault(r["code_equivalent"], {}).setdefault(
                r["code_version"], set()).add(r["git_commit"][:8])
    unknown = [r for r in runs if not r["code_version"]]
    by_source: dict = {}
    for r in runs:
        src = r["commit_source"] or "unstamped"
        by_source[src] = by_source.get(src, 0) + 1
    print("provenance: " + ", ".join(f"{n} {s}" for s, n in sorted(by_source.items())),
          file=sys.stderr)

    def describe(target: str, raw: dict) -> None:
        for fp, cs in sorted(raw.items(), key=lambda kv: kv[0] != target):
            n = sum(1 for r in runs if r["code_version"] == fp)
            via = "" if fp == target else f"  (equivalent to {target})"
            print(f"    {fp}  {n:>5} runs  commits {', '.join(sorted(cs))}{via}",
                  file=sys.stderr)

    if len(versions) > 1:
        print(f"WARNING: the published runs span {len(versions)} training-code versions:",
              file=sys.stderr)
        for target, raw in sorted(versions.items()):
            describe(target, raw)
    elif any(len(raw) > 1 or target not in raw for target, raw in versions.items()):
        print("NOTE: every run's training code counts as one version under "
              "specs/provenance.yaml equivalent_code:", file=sys.stderr)
        for target, raw in versions.items():
            describe(target, raw)
    if unknown:
        print(f"NOTE: {len(unknown)} runs have no establishable code version "
              f"(unstamped with no rule, or stamped dirty with no attestation)",
              file=sys.stderr)

    # The fixed-membership metrics are the primary cluster protocol. Say where
    # they came from, and hold the k = 1 runs to what k = 1 means: one cluster,
    # so a logged or re-scored fixed value is the batch one up to device noise
    # (a CPU re-score sits at most 9.2e-5 from a GPU run's logged test metrics,
    # forecasting/ccm_rescore.py).
    fixed: dict = {}
    scored: dict = {}
    for r in runs:
        if r["fixed_source"]:
            fixed[r["fixed_source"]] = fixed.get(r["fixed_source"], 0) + 1
        if r["fixed_source"] not in ("", "none"):
            fp = r["fixed_code_version"] or "?"
            scored[fp] = scored.get(fp, 0) + 1
    if fixed:
        print("cluster fixed-membership metrics: "
              + ", ".join(f"{n} {s}" for s, n in sorted(fixed.items()))
              + ("; computed by code " + ", ".join(f"{fp} ({n})"
                                                   for fp, n in sorted(scored.items()))
                 if scored else ""), file=sys.stderr)
    off = [r for r in runs if r["k"] == 1 and r["fixed_source"] in ("log", "sidecar")
           and r["mse"] and r["mse_fixed"]
           and abs(float(r["mse_fixed"]) - float(r["mse"])) > 1e-3 * abs(float(r["mse"]))]
    if off:
        print(f"WARNING: {len(off)} k=1 cluster runs have a fixed-membership MSE more "
              f"than 1e-3 (relative) from their batch MSE, e.g. {off[0]['path']}",
              file=sys.stderr)

    counts = {"runs": len(runs), "cells": len(cells), "bands": len(bands),
              "timing": len(timing)}
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
    frames.write_csv(FRAMES / "bands.csv", bands, frames.BAND_FIELDS)
    frames.write_csv(FRAMES / "timing.csv", timing, frames.TIMING_FIELDS)

    if args.epochs:
        epochs = frames.build_epochs(logs, REPO, exclude)
        frames.write_csv(FRAMES / "epochs.csv", epochs, frames.EPOCH_FIELDS)
        counts["epochs"] = len(epochs)

    frames.write_manifest(FRAMES / "MANIFEST.json", REPO, runs, counts, exclude)

    rel = FRAMES.relative_to(REPO) if FRAMES.is_relative_to(REPO) else FRAMES
    print(f"wrote {rel}/runs.csv    {len(runs)} runs")
    print(f"wrote {rel}/cells.csv   {len(cells)} cells")
    print(f"wrote {rel}/bands.csv   {len(bands)} run x band rows")
    print(f"wrote {rel}/timing.csv  {len(timing)} per-channel seeds")
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
    setting = WINDOWS[family][0]
    sections = (T.render_sections(runs_csv, REPO, RESULTS, family)
                + T.render_cross_sections(runs_csv, REPO, RESULTS, family, setting))
    # extract_results is importable here: render_sections put results/ on the path.
    import extract_results as er
    labels = dict(dataset_label=er.dataset_label, model_label=er.model_label,
                  dataset_rank=er.dataset_rank, model_rank=er.model_rank)
    # Every cluster sweep of the family, at every window it ran, with k chosen
    # on validation exactly as the paradigm comparison chooses it.
    sweep = T.render_ccm_sweep(runs_csv, family, WINDOWS[family], **labels)
    if sweep:
        sections.append(("ccm-sweep", T.SWEEP_TITLE, sweep))
    # The seed-aware answer to "which paradigm is best": matched seeds, a tie
    # band at seed noise, k chosen on validation (see emfres/paradigms.py).
    from . import paradigms as PG
    paradigm_md = PG.render(runs_csv, family, setting, **labels)
    if paradigm_md:
        sections.append(("paradigms", "Paradigm Comparison", paradigm_md))
    # The family's other windows, each its own section: the same comparison as a
    # robustness check, never pooled with the benchmark's cells.
    for seq_len, pred_len in WINDOWS[family][1:]:
        title = f"Paradigm Comparison · L{seq_len}/O{pred_len}"
        md = PG.render(runs_csv, family, (seq_len, pred_len), title=title, **labels)
        if md:
            sections.append((f"paradigms-l{seq_len}", title, md))
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
        values_match = R.reconcile(published, FRAMES / "cells.csv") == 0
        if not values_match and not args.force:
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
                # Under --force the values may be exactly what changed, so the
                # reassurance is only printed when the gate actually passed.
                verdict = ("every published value still matches" if values_match
                           else "published values changed -- written under --force")
                print(f"NOTE: '{title}' changed since {published.name} "
                      f"(+{add}/-{rem} lines; {verdict})", file=sys.stderr)
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
