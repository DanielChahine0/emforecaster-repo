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

# The frozen report each family's published values are gated against
# (make.py tables / reconcile), None until one is frozen: a family without one
# is rendered ungated, and reconcile says so. Re-frozen when a set of numbers
# is accepted as published; the previous snapshots stay in archive/ as
# provenance. The changes expected against a family's snapshot are declared
# in its block of specs/gate.yaml, which names the snapshot its rules were
# written for, so a re-freeze retires them.
#
# Freezing a family F: render it (make.py tables --family F), copy its all.md
# (tables/all.md, tables/voip/all.md) into archive/ as <date>_tables.md for
# emf or <date>_voip_tables.md for voip, point SNAPSHOTS[F] at the copy, and
# set F's `snapshot` in specs/gate.yaml to its name, with empty lists. Only
# once no cell prints `(n=…)`: one still training is held at the mean it
# printed, and fails the gate as its seeds finish.
PUBLISHED = RESULTS / "archive" / "2026-09-22_tables.md"
SNAPSHOTS: dict[str, Path | None] = {"emf": PUBLISHED, "voip": None}


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


def render_family(family: str) -> list[tuple[str, str, str]]:
    """Every section `make.py tables` writes for a family: (slug, title, markdown)."""
    from . import tables as T

    runs_csv = FRAMES / "runs.csv"
    # The paradigm comparison's band variants read bands.csv beside runs.csv.
    # Without it they are left out (the section's note says so) -- which the
    # tracked frames alone cannot tell apart from bands that changed nothing.
    if not (FRAMES / "bands.csv").exists():
        print(f"WARNING: {FRAMES / 'bands.csv'} not found: the paradigm comparison is "
              f"rendered WITHOUT its band-level sensitivity rows and per-cell band "
              f"table. Run `make.py extract` to rebuild it; it has to be committed "
              f"with runs.csv for the tracked frames to reproduce those rows.",
              file=sys.stderr)

    # The benchmark window every cross-setup section is built for, per corpus.
    setting = WINDOWS[family][0]
    sections = (T.render_sections(runs_csv, REPO, RESULTS, family)
                + T.render_cross_sections(runs_csv, REPO, RESULTS, family, setting))
    # Every cluster sweep of the family, at every window it ran, with k chosen
    # on validation exactly as the paradigm comparison chooses it.
    sweep = T.render_ccm_sweep(runs_csv, family, WINDOWS[family], **_labels())
    if sweep:
        sections.append(("ccm-sweep", T.SWEEP_TITLE, sweep))
    return sections + render_paradigms(family)


def _labels() -> dict:
    """The tables' dataset and model labels and orders (extract_results)."""
    code = str(Path(__file__).resolve().parent.parent)
    if code not in sys.path:
        sys.path.insert(0, code)
    import extract_results as er
    return dict(dataset_label=er.dataset_label, model_label=er.model_label,
                dataset_rank=er.dataset_rank, model_rank=er.model_rank)


def render_paradigms(family: str, models=None) -> list[tuple[str, str, str]]:
    """The family's paradigm comparison, one section per window, benchmark first.

    The seed-aware answer to "which paradigm is best": matched seeds, a tie
    band at seed noise, k chosen on validation (see emfres/paradigms.py). The
    family's other windows are each their own section: the same comparison as
    a robustness check, never pooled with the benchmark's cells. ``models``
    re-counts it over those model ids only, for the value gate's
    held_elsewhere rows (specs/gate.yaml).
    """
    from . import paradigms as PG
    from .reconcile import PARADIGM_TITLE

    out = []
    for i, (seq_len, pred_len) in enumerate(WINDOWS[family]):
        title = PARADIGM_TITLE if i == 0 else f"{PARADIGM_TITLE} · L{seq_len}/O{pred_len}"
        md = PG.render(FRAMES / "runs.csv", family, (seq_len, pred_len), title=title,
                       models=models, **_labels())
        if md:
            out.append(("paradigms" if i == 0 else f"paradigms-l{seq_len}", title, md))
    return out


def run_gate(family: str, sections, snapshot: Path | None = None, verbose: bool = False):
    """The value gate for a family's rendered sections, or None when it has no snapshot."""
    from . import reconcile as R

    snapshot = SNAPSHOTS.get(family) if snapshot is None else snapshot
    if snapshot is None:
        print(f"NOTE: family {family!r} has no frozen snapshot yet (SNAPSHOTS in "
              f"emfres/cli.py) -- its values are not gated", file=sys.stderr)
        return None
    if not snapshot.exists():
        print(f"WARNING: the frozen snapshot {snapshot} is missing -- values are not "
              f"gated", file=sys.stderr)
        return None

    def render_models(models) -> str:
        return "\n\n".join(md.strip() for _, _, md in render_paradigms(family, models))

    return R.gate(snapshot, FRAMES / "cells.csv", sections, family=family,
                  runs_spec=SPECS / "runs.yaml", gate_spec=SPECS / "gate.yaml",
                  render_models=render_models, verbose=verbose)


def cmd_tables(args) -> int:
    import json

    from . import reconcile as R
    from . import tables as T

    runs_csv = FRAMES / "runs.csv"
    if not runs_csv.exists():
        print(f"{runs_csv} not found -- run `make.py extract` first", file=sys.stderr)
        return 1
    family = args.family
    sections = render_family(family)
    if not sections:
        print(f"no runs in family {family!r}", file=sys.stderr)
        return 1

    # Gate: every value the frozen report publishes must still come out of the
    # CSVs, or be a change specs/gate.yaml declares. This is deliberately a
    # *value* check, not a text comparison.
    #
    # It used to diff the rendered markdown against the archive and refuse on
    # any difference. That cannot survive a section legitimately gaining rows:
    # restoring the cluster L-ladder added nine lookbacks and reshaped the
    # parameter-count table from one column to ten, changing a great deal of
    # text while leaving every published number identical. A tripwire that
    # fires on correct work teaches you to pass --force, which is worse than
    # having no tripwire. The gate tolerates growth by construction -- a value
    # the archive never had is a value it cannot contradict.
    gate = run_gate(family, sections)
    # No snapshot yet (SNAPSHOTS[family] is None) renders ungated; a frozen one
    # whose file is gone is a failure, as in `reconcile`: its values would go
    # ungated with nothing but a warning to say so.
    snapshot = SNAPSHOTS.get(family)
    if snapshot is not None and not snapshot.exists():
        if args.check:
            print(f"\nrendered {len(sections)} sections -- family {family!r}'s frozen "
                  f"snapshot {snapshot.name} is missing, so no value was gated")
            return 1
        if not args.force:
            print(f"\nfamily {family!r}'s frozen snapshot {snapshot} is missing, so no "
                  f"value was gated; refusing to write. Restore it, or pass --force "
                  f"to write anyway.", file=sys.stderr)
            return 1
    if gate is not None and not gate.ok and not args.force:
        print(f"\nA published value changed and specs/gate.yaml does not expect it; "
              f"refusing to write. Pass --force to write anyway.", file=sys.stderr)
        return 1

    # Text drift is reported, never fatal: it says how the report grew, and
    # what of each changed section the gate did (or did not) hold.
    if gate is not None:
        from difflib import unified_diff

        published = SNAPSHOTS[family]      # there is one: the gate ran against it
        blocks = R.split_blocks(published.read_text())
        titles = frozenset(title for _, title, _ in sections)
        for _, title, md in sections:
            old = R.find_blocks(blocks, title, claimed=titles - {title})
            if not old:
                print(f"NOTE: '{title}' is new -- not in {published.name}", file=sys.stderr)
                continue
            was = "\n\n".join(block for _, block in old)
            diff = list(unified_diff(was.splitlines(), md.strip().splitlines(),
                                     lineterm="", n=0))
            add = sum(1 for line in diff if line.startswith("+") and not line.startswith("+++"))
            rem = sum(1 for line in diff if line.startswith("-") and not line.startswith("---"))
            if add or rem:
                heading = (f"'{title}'" if [h for h, _ in old] == [title] else
                           f"'{title}' (was {', '.join(repr(h) for h, _ in old)})")
                print(f"NOTE: {heading} changed since {published.name} "
                      f"(+{add}/-{rem} lines; {gate.coverage(title)})", file=sys.stderr)
    if args.check:
        if gate is None:
            print(f"\nrendered {len(sections)} sections -- family {family!r} has no "
                  f"frozen snapshot, so no value was gated")
            return 0
        if not gate.ok:
            print(f"\nchecked {len(sections)} sections -- {gate.n_unexpected} "
                  f"published values changed unexpectedly (see above)")
            return 1
        print(f"\nchecked {len(sections)} sections -- every published value reproduces"
              + (f" or is one of the {gate.n_expected} changes specs/gate.yaml "
                 f"expects" if gate.n_expected else ""))
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
    index += [f"- [`csv/{p.name}`](csv/{p.name}){T.twin_note(p)}" for p in twins]
    (outdir / "index.md").write_text("\n".join(index) + "\n")

    # RESULTS can point outside the repo (a scratch render, the tests).
    rel = outdir.relative_to(REPO) if outdir.is_relative_to(REPO) else outdir
    print(f"wrote {len(sections)} sections + all.md + index.md to {rel}/")
    for slug, title, _ in sections:
        print(f"  {slug + '.md':<22} {title}")
    print(f"wrote {len(twins)} csv twins to {rel}/csv/")
    return 0


def cmd_reconcile(args) -> int:
    """The value gate of every family asked for that has a frozen snapshot.

    A family without one is named, not gated, and does not fail the run; a
    frozen snapshot that is missing does. With more than one family, a last
    line gives each one's verdict.
    """
    for name in ("cells.csv", "runs.csv"):
        if not (FRAMES / name).exists():
            print(f"{FRAMES / name} not found -- run `make.py extract` first",
                  file=sys.stderr)
            return 1
    families = list(SNAPSHOTS) if args.family == "all" else [args.family]
    if args.tables is not None and len(families) > 1:
        print("--tables is one family's snapshot: pass --family with it",
              file=sys.stderr)
        return 1
    if args.tables is None and len(families) == 1 and SNAPSHOTS.get(families[0]) is None:
        print(f"no frozen snapshot for family {families[0]!r} yet (SNAPSHOTS in "
              f"emfres/cli.py); pass --tables, or freeze one as specs/gate.yaml "
              f"describes", file=sys.stderr)
        return 1

    ok, gated, verdicts = True, 0, []
    for i, family in enumerate(families):
        if i:
            print()
        snapshot = args.tables if args.tables is not None else SNAPSHOTS.get(family)
        if snapshot is None:
            print(f"value gate ({family}): no frozen snapshot yet -- the {family} tables "
                  f"are not gated (SNAPSHOTS in emfres/cli.py; specs/gate.yaml describes "
                  f"how to freeze one)", flush=True)
            verdicts.append(f"{family} not gated (no frozen snapshot yet)")
            continue
        gated += 1
        gate = run_gate(family, render_family(family), snapshot=snapshot, verbose=args.all)
        if gate is None:              # frozen, but the file is gone: run_gate said so
            ok = False
            verdicts.append(f"{family} FAIL ({snapshot.name} is missing)")
        elif not gate.ok:
            ok = False
            verdicts.append(f"{family} FAIL ({gate.n_unexpected} unexpected changes)")
        else:
            verdicts.append(f"{family} PASS" + (f" ({gate.n_expected} declared changes)"
                                               if gate.n_expected else ""))
    if len(families) > 1:
        print(f"\nreconcile: {'; '.join(verdicts)}")
    if not gated:
        print("no family has a frozen snapshot: nothing was gated", file=sys.stderr)
        return 1
    return 0 if ok else 1


def cmd_figures(args) -> int:
    from . import figures

    frames_dir = args.frames or FRAMES
    if not (frames_dir / "runs.csv").exists():
        print(f"{frames_dir / 'runs.csv'} not found -- run `make.py extract` first",
              file=sys.stderr)
        return 1
    families = ("emf", "voip") if args.family == "all" else (args.family,)
    only = [f.strip().upper() for f in args.only.split(",") if f.strip()] if args.only else None
    formats = [f.strip() for f in args.formats.split(",") if f.strip()]
    try:
        written, problems = figures.render(frames_dir, args.outdir or RESULTS / "figures",
                                           families=families, only=only, formats=formats,
                                           style_name=args.style)
    except ValueError as exc:
        print(exc, file=sys.stderr)
        return 1
    for path in written:
        print(f"wrote {path.relative_to(REPO) if path.is_relative_to(REPO) else path}")
    if problems:
        for key, metric, got, want in problems[:10]:
            print(f"SPOT CHECK: {key} {metric}: plotted {got!r}, cells.csv {want!r}",
                  file=sys.stderr)
        print(f"{len(problems)} plotted values disagree with cells.csv", file=sys.stderr)
        return 1
    return 0


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
                    help="rebuild and gate against the frozen snapshot, write nothing")
    tb.add_argument("--force", action="store_true",
                    help="write even if a published value changed unexpectedly, or "
                         "the family's frozen snapshot is missing")
    tb.set_defaults(func=cmd_tables)

    rc = sub.add_parser(
        "reconcile", help="hold the frozen report's published values against frames/")
    rc.add_argument("--family", default="all", choices=[*SNAPSHOTS, "all"],
                    help="which corpus to gate (default: all, every family with a "
                         "frozen snapshot; each one without is named, not gated)")
    rc.add_argument("--tables", type=Path, default=None,
                    help="the snapshot to hold, with --family emf or voip (default: "
                         "each family's frozen one, SNAPSHOTS in emfres/cli.py)")
    rc.add_argument("--all", action="store_true",
                    help="list every expected change and every new value, uncapped "
                         "(of every family gated: all of them unless --family names one)")
    rc.set_defaults(func=cmd_reconcile)

    fg = sub.add_parser("figures", help="frames/ -> the paradigm-comparison figures")
    fg.add_argument("--family", default="all", choices=["emf", "voip", "all"],
                    help="which corpus to draw (default: both; each figure is one corpus, "
                         "except the scoreboard, which stacks them)")
    fg.add_argument("--only", default="",
                    help="comma-separated figure ids, e.g. P1,P5 (default: every one; "
                         "see emfres/figures.py)")
    fg.add_argument("--frames", type=Path, default=None,
                    help=f"frames directory to read (default: {FRAMES})")
    fg.add_argument("--outdir", type=Path, default=None,
                    help="figures root the generated paths hang off (default: results/figures)")
    fg.add_argument("--formats", default="png", help="comma-separated output formats")
    fg.add_argument("--style", default="paper", choices=["paper", "slide"],
                    help="type and mark sizes (default: paper, IEEEtran)")
    fg.set_defaults(func=cmd_figures)

    args = parser.parse_args(argv)
    return args.func(args)
