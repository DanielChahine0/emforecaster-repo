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


# The GPU logs the 2026-09-24 sweeps write, each beside the sweep log whose
# start/STOP/DONE lines bound it: the CCM sweep (.personal/run_ccm_all.sh), the
# tail rerun of the August per-channel bands (.personal/run_tail.sh), and its
# ARIMA bands, trained beside the CCM sweep (.personal/run_tail_arima.sh; where
# two of these ran at once, emfres.contention.combine gives a shared window).
# All are git-ignored: `make.py contention` copies what the frames need into
# specs/contention.yaml, skipping a pair whose GPU log does not exist yet.
LOG_PAIRS = [
    (REPO / ".personal" / "ccm_gpu_contention.log", REPO / ".personal" / "ccm_sweep.log"),
    (REPO / ".personal" / "tail_gpu_contention.log", REPO / ".personal" / "tail_sweep.log"),
    (REPO / ".personal" / "tail_arima_gpu.log", REPO / ".personal" / "tail_arima.log"),
]


def _rel(path: Path) -> str:
    """A path as the repo spells it, or as given when it lies outside the repo."""
    return path.relative_to(REPO).as_posix() if path.is_relative_to(REPO) else str(path)


def spec_contention():
    """The foreign GPU windows of specs/contention.yaml, or None (see emfres.contention)."""
    from . import contention

    path = SPECS / "contention.yaml"
    try:
        return contention.load(path)
    except Exception as exc:
        print(f"WARNING: could not read {path}: {exc}", file=sys.stderr)
        return None


def gpu_log_entries_after(until) -> int | None:
    """How many entries the local GPU logs (LOG_PAIRS) have after ``until``:
    a sweep that went on after specs/contention.yaml was derived, or a later
    sweep's log it has not read. None when no GPU log is readable (another
    machine: .personal is git-ignored)."""
    from . import contention as C

    found = None
    for g, _ in LOG_PAIRS:
        try:
            text = g.read_text()
        except OSError:
            continue
        found = (found or 0) + sum(1 for snap in C.parse_gpu_log(text) if snap.time > until)
    return found


def stale_contention(frames_dir: Path) -> str:
    """Why the frames' contended flags are not final, from the contention
    entry extract writes to MANIFEST.json; '' when they are (or it has none)."""
    import json

    path = frames_dir / "MANIFEST.json"
    try:
        c = json.loads(path.read_text()).get("contention") or {}
    except (OSError, ValueError):
        return ""
    why = []
    if c.get("past_end"):
        why.append(f"{c['past_end']} runs trained past specs/contention.yaml's "
                   f"observed_until {c.get('observed_until')} while it was open, so their "
                   f"contended flag is ? or a lower bound")
    if c.get("gpu_log_entries_after"):
        why.append(f"the GPU logs had {c['gpu_log_entries_after']} entries after it when "
                   f"the frames were extracted, so it is stale")
    return "; ".join(why)


# What to do about stale_contention, after a sweep (specs/contention.yaml's header).
STALE_FIX = ("rerun `make.py contention`, check it with `make.py contention --check`, "
             "then `make.py extract`, and commit the spec with the frames")


def cmd_contention(args) -> int:
    """GPU logs -> specs/contention.yaml: the foreign windows and a per-cell summary.

    --check derives it again to the committed file's observed_until and says
    whether the file is what the logs give, and whether a GPU log has entries
    after it (a sweep went on: rerun without --check).
    """
    from datetime import datetime, timedelta

    from . import contention as C

    if args.gpu_log:
        sweeps = list(args.sweep_log or [])
        pairs = [(g, sweeps[i] if i < len(sweeps) else None)
                 for i, g in enumerate(args.gpu_log)]
    else:
        pairs = [(g, w) for g, w in LOG_PAIRS if g.exists()]
    missing = [g for g, _ in pairs if not g.exists()]
    if not pairs or missing:
        print(f"no GPU log to derive from ({', '.join(map(str, missing)) or 'none found'})",
              file=sys.stderr)
        return 1
    spec = SPECS / "contention.yaml"
    if args.check:
        old = C.load(spec)
        if old is None:
            print(f"{spec} not found -- run `make.py contention`", file=sys.stderr)
            return 1
        until = old.observed_until
    elif args.until:
        until = datetime.strptime(args.until, C.TIME_FMT)
    else:
        until = datetime.now().replace(microsecond=0)

    derived = C.combine([
        C.derive(g.read_text(), w.read_text() if w is not None and w.exists() else None,
                 until=until, source=_rel(g))
        for g, w in pairs])
    exclude = [] if args.no_spec else spec_excludes()
    loaded = frames.load_all(args.logs or LOGS, exclude, contention=derived)
    text = C.dump(derived, C.summarise(derived, loaded))
    flagged = [r for _, r in loaded if r.foreign_overlap]
    watched = [r for _, r in loaded if r.foreign_overlap is not None]
    partial = [r for r in watched
               if derived.monitored_seconds(r.started, r.started + timedelta(
                   seconds=r.running_time)) < r.running_time - 1]
    fg = sum((w.end - w.start).total_seconds() for w in derived.windows)
    sh = sum((s.end - s.start).total_seconds() for s in derived.shared)
    mon = sum((e - s).total_seconds() for s, e in derived.monitored)
    print(f"monitored {mon / 3600:.2f} h, foreign processes on the GPU for {fg / 3600:.2f} h "
          f"in {len(derived.windows)} windows, two launches of this repo at once for "
          f"{sh / 3600:.2f} h in {len(derived.shared)} shared windows; {len(flagged)} of "
          f"{len(watched)} monitored runs overlap them"
          + (" (the last span is still open)" if derived.open else ""))
    for s in derived.shared:
        print(f"  shared {C._fmt(s.start)} to {C._fmt(s.end)}: {s.launches}")
    seen: dict = {}
    for g, _ in pairs:
        for snap in C.parse_gpu_log(g.read_text()):
            for _, exe, _ in snap.foreign:
                seen[exe] = seen.get(exe, 0) + 1
    for exe, n in sorted(seen.items(), key=lambda kv: -kv[1]):
        print(f"  foreign executable {exe}  ({n} entries)")
    if partial:
        print(f"WARNING: {len(partial)} runs are only partly inside the monitored time; "
              f"their foreign_overlap is a lower bound, e.g. {partial[0].path}",
              file=sys.stderr)
    if args.check:
        later = [s for g, _ in pairs for s in C.parse_gpu_log(g.read_text())
                 if until is not None and s.time > until]
        same = spec.read_text() == text
        print(f"{spec.name}: " + ("reproduces from the logs" if same else
                                  "does NOT reproduce from the logs (rerun `make.py contention`)")
              + (f"; the GPU logs have {len(later)} entries after its observed_until "
                 f"{C._fmt(until)}" if later else ""))
        return 0 if same and not later else 1
    spec.write_text(text)
    print(f"wrote {_rel(spec)}")
    return 0


def cmd_extract(args) -> int:
    from datetime import timedelta

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

    contention = spec_contention()
    loaded = frames.load_all(logs, exclude, inferred=spec_inferred(),
                             attested=spec_attested(), contention=contention)
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

    # Whose running_time is contended (specs/contention.yaml, emfres.contention).
    flags: dict = {}
    for r in runs:
        for token in (r["contended"] or "none").split(";"):
            flags[token] = flags.get(token, 0) + 1
    print("contended timings: " + ", ".join(f"{n} {t}" for t, n in sorted(flags.items()))
          + (f" (foreign windows: specs/contention.yaml, observed until "
             f"{contention.observed_until:%Y-%m-%d %H:%M:%S}"
             + (", its last span still open" if contention.open else "") + ")"
             if contention is not None and contention.observed_until else
             " (no specs/contention.yaml: nothing is flagged foreign)"), file=sys.stderr)
    manifest_contention = None
    if contention is not None and contention.observed_until:
        until = f"{contention.observed_until:%Y-%m-%d %H:%M:%S}"
        # An open spec: the sweep went on after it was derived. Such a run is
        # flagged ? (or foreign, on the overlap seen so far), never none.
        past = [r for _, r in loaded if contention.past_end(r.started, r.running_time)]
        if past:
            print(f"WARNING: {len(past)} runs trained past specs/contention.yaml's "
                  f"observed_until {until} while its sweep was still running, e.g. "
                  f"{past[0].path}: their contended flag is ? where no overlap was seen "
                  f"yet, a lower bound where one was. `make.py tables` will not write "
                  f"from these frames; {STALE_FIX}", file=sys.stderr)
        # A closed spec: a run after its sweep's DONE has no GPU record.
        late = [r for _, r in loaded if not contention.open and r.started is not None
                and r.running_time
                and r.started + timedelta(seconds=r.running_time) > contention.observed_until
                and r.started >= contention.monitored[0][0]] if contention.monitored else []
        if late:
            print(f"WARNING: {len(late)} runs finished after specs/contention.yaml's "
                  f"observed_until {until}, so part of their time was never held against "
                  f"the GPU log; if a GPU log covers them, {STALE_FIX}", file=sys.stderr)
        after = gpu_log_entries_after(contention.observed_until)
        if after:
            print(f"WARNING: the GPU logs have {after} entries after specs/contention.yaml's "
                  f"observed_until {until}: it is stale. `make.py tables` will not write "
                  f"from these frames; {STALE_FIX}", file=sys.stderr)
        manifest_contention = {"observed_until": until, "open": contention.open,
                               "past_end": len(past), "gpu_log_entries_after": after}

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

    frames.write_manifest(FRAMES / "MANIFEST.json", REPO, runs, counts, exclude,
                          contention=manifest_contention)

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
    # Frames whose contention flags are not final: a timing table would print
    # an unmarked time for a run nobody held against the GPU log.
    stale = stale_contention(FRAMES)
    if stale:
        if args.check:
            print(f"NOTE: the frames' contention flags are not final: {stale}",
                  file=sys.stderr)
        elif not args.force:
            print(f"\nThe frames' contention flags are not final: {stale}. Refusing to "
                  f"write: {STALE_FIX}. Pass --force to write anyway.", file=sys.stderr)
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
    stale = stale_contention(frames_dir)
    if stale:
        print(f"WARNING: the frames' contention flags are not final: {stale}; P7 draws "
              f"such a cost as if clean. {STALE_FIX[0].upper()}{STALE_FIX[1:]}.",
              file=sys.stderr)
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
                    help="write even if a published value changed unexpectedly, the "
                         "family's frozen snapshot is missing, or the frames' contention "
                         "flags are not final (MANIFEST.json)")
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

    ct = sub.add_parser(
        "contention", help="the sweep's GPU log -> specs/contention.yaml (foreign GPU windows)")
    ct.add_argument("--gpu-log", type=Path, action="append", default=None,
                    help="a GPU log to read, repeatable (default: every one of "
                         + ", ".join(_rel(g) for g, _ in LOG_PAIRS)
                         + " that exists)")
    ct.add_argument("--sweep-log", type=Path, action="append", default=None,
                    help="the sweep log beside each --gpu-log, in order, whose start/STOP/"
                         "DONE lines bound its monitored spans (default: "
                         + ", ".join(_rel(w) for _, w in LOG_PAIRS)
                         + ")")
    ct.add_argument("--until", default=None,
                    help="end of a span still running, 'YYYY-MM-DD HH:MM:SS' (default: now)")
    ct.add_argument("--logs", type=Path, default=None,
                    help=f"logs tree the per-cell summary reads (default: {LOGS})")
    ct.add_argument("--no-spec", action="store_true", help="ignore specs/runs.yaml")
    ct.add_argument("--check", action="store_true",
                    help="derive again to the committed observed_until, write nothing; "
                         "fails when the file does not reproduce or the GPU log went on")
    ct.set_defaults(func=cmd_contention)

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
