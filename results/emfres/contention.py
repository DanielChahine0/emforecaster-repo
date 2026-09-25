"""Another user's processes on the GPU while a sweep trained, and the runs they overlapped.

A logged ``parameters/running_time`` is a wall time. Two things inflate it
without changing what a run computes: our own trainings sharing the card by
design (the per-channel grid trains 8 to 12 bands at once, stamped as
``parameters/exp/concurrency``), and somebody else's jobs on the same GPU,
which no run log records. The 2026-09-24 CCM sweep (.personal/run_ccm_all.sh)
records the second: before each block, and whenever the set of compute apps
changes on a 60 s poll, it appends the GPU's compute apps to
.personal/ccm_gpu_contention.log, each marked ``own`` (a python whose cwd is
this repo) or ``FOREIGN``.

This module turns that log into the foreign windows (``derive``), which
``make.py contention`` writes to specs/contention.yaml so the frames never read
the git-ignored .personal/, and gives every run a flag from its stamps and
those windows (``flag``).

A third thing no log marks FOREIGN: another launch of this repo on the same
card. The ARIMA part of the tail rerun (.personal/run_tail_arima.sh) ran
beside the CCM sweep on 2026-09-24 from 20:45:27 to 20:52:18; each poller
marks the other's processes ``own`` (their cwd is this repo too), and neither
log records process groups. Each launcher's own log bounds its launches, so
where the launches of two GPU logs overlap in time (``combine``), that stretch
is a ``shared`` window, and a run's overlap counts it as it counts a foreign
one:

* ``concurrent``: its own grid trained several runs at once (concurrency > 1);
* ``foreign``: its training window overlapped a foreign window (another user's
  process) or a shared one (another launch of this repo);
* ``none``: neither is recorded -- concurrency 1 and no foreign process seen,
  or no GPU record at all (every run before 2026-09-24 10:38 but the solo
  timing pass of specs/timing.yaml, which found none);
* ``?``: not known -- concurrency not stamped (runs logged before
  2026-09-19), or, with no overlap seen, a run that trained past the end of
  a spec derived while its sweep was still running (``Contention.past_end``):
  the GPU log goes on, the spec does not, and ``none`` would read as clean.

A run's training window is [parameters/exp/time, + running_time]. The flag
says its time may be inflated; its accuracy is unaffected -- sharing a GPU
changes scheduling, not arithmetic.

What the log can and cannot say. The poller samples once a minute and writes
only when the set of (pid, name) changes, so a sampled state is taken to hold
until the next entry: a window's edges are known to about a minute, and a
foreign process that came and went between two polls is not seen. It records
memory, not utilisation, so an overlap says another user's process held the
card, not how hard it worked it.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Iterable, Optional

TIME_FMT = "%Y-%m-%d %H:%M:%S"

# run_ccm_all.sh poll_gpu's sleep between samples.
POLL_SECONDS = 60

FOREIGN = "foreign"
CONCURRENT = "concurrent"
NONE = "none"
UNKNOWN = "?"

# A snapshot header: "=== <time> before <block>" (gpu_apps before a block, and
# "=== <time> sweep done") or "--- <time> [<block>] compute apps changed".
_SNAPSHOT = re.compile(r"^(===|---) (\d{4}-\d\d-\d\d \d\d:\d\d:\d\d) (.*)$")
# One compute app under a header: "  FOREIGN 2243602, /path/python, 672 MiB".
# The tail rerun's log (.personal/run_tail.sh) also marks its MPS server
# `own-mps` and adds "(user <name>)" after a FOREIGN entry; the ARIMA tail's
# (.personal/run_tail_arima.sh) adds what an own process runs, "(--setup
# cluster)" or "(rf_emf_mv/arima)".
_APP = re.compile(r"^\s+(FOREIGN|own-mps|own)\s+(\d+),\s*(.*?),\s*(\d+)\s*MiB"
                  r"(?:\s*\([^()]*\))*\s*$")
# The ARIMA tail's line of card totals under each entry, "  gpu: used, free,
# util = ...": not an app, so neither read nor a sign of a failed sample.
_TOTALS = re.compile(r"^\s+gpu: ")
# nvidia-smi's name for a process that exited while it was listing: its /proc
# entry is gone, so the poller cannot see its cwd and marks it FOREIGN (the
# ARIMA tail's log at 20:51:20 on 2026-09-24: one of its own bands, "(user
# ?)"). It holds no memory for the interval the entry opens: not read.
_EXITED = "[No data]"
# The sweep log's own start and end lines (.personal/ccm_sweep.log). An ABORT
# line is not an end: every ABORT comes before the poller starts, so an
# aborted launch has no GPU entry to bound.
_SWEEP = re.compile(r"^=== (start|STOP|DONE) (\d{4}-\d\d-\d\d \d\d:\d\d:\d\d)")


def _time(text: str) -> datetime:
    return datetime.strptime(str(text).strip(), TIME_FMT)


def _fmt(t: datetime) -> str:
    return t.strftime(TIME_FMT)


@dataclass
class Snapshot:
    """The GPU's compute apps at one sample. ``unknown`` when a line under it
    did not parse (an nvidia-smi error, say): the state until the next sample
    is then not known."""
    time: datetime
    block: str
    foreign: list          # (pid, process, MiB)
    own: list
    unknown: bool = False


def parse_gpu_log(text: str) -> list[Snapshot]:
    """Every snapshot in a run_ccm_all.sh, run_tail.sh or run_tail_arima.sh GPU
    log, in time order."""
    snaps: list[Snapshot] = []
    for line in text.splitlines():
        m = _SNAPSHOT.match(line)
        if m:
            block = m.group(3)
            b = re.search(r"\[(.*)\]", block)
            block = b.group(1) if b else re.sub(r"^before ", "", block)
            snaps.append(Snapshot(_time(m.group(2)), block, [], []))
            continue
        if not line.strip() or not snaps or _TOTALS.match(line):
            continue
        a = _APP.match(line)
        if a is None:
            snaps[-1].unknown = True
            continue
        app = (int(a.group(2)), a.group(3), int(a.group(4)))
        if app[1] == _EXITED:
            continue
        (snaps[-1].foreign if a.group(1) == "FOREIGN" else snaps[-1].own).append(app)
    snaps.sort(key=lambda s: s.time)
    return snaps


def sweep_spans(text: str) -> list[tuple[datetime, Optional[datetime]]]:
    """[start, end) of every sweep launch in the sweep log; end None if unended.

    A start line opens a span and the next STOP or DONE line closes it. A span
    that the next start finds still open was killed without a line (or is
    the one running now): its end stays None, since when its poller stopped
    is not known.
    """
    spans: list = []
    for line in text.splitlines():
        m = _SWEEP.match(line)
        if not m:
            continue
        if m.group(1) == "start":
            spans.append([_time(m.group(2)), None])
        elif spans and spans[-1][1] is None:
            spans[-1][1] = _time(m.group(2))
    return [(s, e) for s, e in spans]


@dataclass(frozen=True)
class Window:
    """A stretch of wall time with at least one foreign process on the GPU."""
    start: datetime
    end: datetime
    peak_mib: int = 0      # the largest foreign process seen in it
    processes: int = 0     # distinct foreign pids seen in it
    # Those pids, so that two logs of one stretch (two launches side by side)
    # count a process once; not written to the spec, empty when loaded from it.
    pids: frozenset = field(default=frozenset(), compare=False, repr=False)


@dataclass(frozen=True)
class Shared:
    """A stretch of wall time when launches of two GPU logs (two of this
    repo's launchers) were running at once: each one's processes were on the
    card beside the other's, which its log marks ``own``."""
    start: datetime
    end: datetime
    launches: str = ""     # the two GPU logs, "a + b"


def _merge(spans: list) -> list:
    out: list = []
    for s, e in sorted(spans):
        if out and s <= out[-1][1]:
            out[-1][1] = max(out[-1][1], e)
        else:
            out.append([s, e])
    return [(s, e) for s, e in out]


@dataclass
class Contention:
    """The foreign windows of the monitored wall time (specs/contention.yaml)."""
    monitored: list          # merged (start, end) spans the GPU log covers
    windows: list            # Window, merged, inside monitored
    source: str = ""
    observed_until: Optional[datetime] = None
    open: bool = False       # the last span had not ended when this was derived
    shared: list = field(default_factory=list)   # Shared, merged (combine)
    # Each launch's (start, end) from its first GPU entry: combine's input for
    # `shared`; not written to the spec.
    launches: list = field(default_factory=list)

    def contended(self) -> list:
        """The merged (start, end) stretches with another user's process
        (``windows``) or another launch of this repo (``shared``) on the GPU."""
        return _merge([(w.start, w.end) for w in self.windows]
                      + [(s.start, s.end) for s in self.shared])

    def foreign_seconds(self, start: datetime, end: datetime) -> float:
        """Seconds of [start, end) inside ``contended``: a foreign or a shared
        window, each second counted once."""
        return sum(max(0.0, (min(end, e) - max(start, s)).total_seconds())
                   for s, e in self.contended())

    def monitored_seconds(self, start: datetime, end: datetime) -> float:
        return sum(max(0.0, (min(end, e) - max(start, s)).total_seconds())
                   for s, e in self.monitored)

    def overlap(self, started: Optional[datetime], seconds: Optional[float]) -> Optional[float]:
        """The fraction of a run's training window with a foreign process or
        another launch seen (``foreign_seconds``), or None when no part of it
        was monitored (or it has no window).

        A lower bound where part of the window fell outside the monitored time.
        """
        if started is None or seconds is None or seconds <= 0:
            return None
        end = started + timedelta(seconds=seconds)
        if self.monitored_seconds(started, end) <= 0:
            return None
        return min(1.0, self.foreign_seconds(started, end) / seconds)

    def past_end(self, started: Optional[datetime], seconds: Optional[float]) -> bool:
        """Whether a run's training window runs past observed_until while the
        last span is open: the sweep went on after the spec was derived, so
        the part after it was logged but never read (flag gives such a run
        ``?``, not ``none``). A closed spec's end is its sweep's DONE: a run
        after it has no GPU record, as every run before the sweep."""
        if not self.open or self.observed_until is None or started is None or not seconds:
            return False
        return started + timedelta(seconds=seconds) > self.observed_until


def derive(gpu_log: str, sweep_log: Optional[str] = None,
           until: Optional[datetime] = None, source: str = "") -> Contention:
    """The monitored spans and foreign windows a GPU log gives.

    Each snapshot's state holds from its time to the next snapshot. The
    monitored time is each sweep span (``sweep_log``; without one, the log's
    own first to last snapshot) from its first snapshot to its end: its STOP
    or DONE line; ``until`` for the last span when it has none, a sweep still
    running (default: its last snapshot); otherwise -- a sweep killed without
    a line -- its last snapshot. A snapshot whose apps did not parse leaves
    its interval out.

    ``until`` also ends everything: nothing after it is read, and a span still
    running at it (ended later, or not at all) is the open last span. So a
    derive to a spec's observed_until gives that spec again once the logs go
    on, a later launch in another log included.
    """
    snaps = parse_gpu_log(gpu_log)
    if until is not None:
        snaps = [s for s in snaps if s.time <= until]
    if not snaps:
        return Contention([], [], source)
    spans = sweep_spans(sweep_log) if sweep_log else [(snaps[0].time, None)]
    if until is not None:
        spans = [(s, e if e is not None and e <= until else None)
                 for s, e in spans if s <= until]
    is_open = False
    bounded = []
    for i, (start, end) in enumerate(spans):
        nxt = spans[i + 1][0] if i + 1 < len(spans) else None
        mine = [s for s in snaps if s.time >= start and (nxt is None or s.time < nxt)]
        if not mine:
            continue
        if end is None:
            if nxt is None:
                is_open = sweep_log is not None
                end = until or mine[-1].time
            else:
                end = mine[-1].time
        bounded.append((mine, max(end, mine[0].time)))
    monitored, foreign = [], []
    for mine, end in bounded:
        for i, snap in enumerate(mine):
            seg_end = min(mine[i + 1].time if i + 1 < len(mine) else end, end)
            if seg_end <= snap.time or snap.unknown:
                continue
            monitored.append((snap.time, seg_end))
            if snap.foreign:
                foreign.append((snap.time, seg_end, snap))
    windows = []
    for s, e in _merge([(s, e) for s, e, _ in foreign]):
        inside = [snap for fs, fe, snap in foreign if fs >= s and fe <= e]
        pids = frozenset(a[0] for x in inside for a in x.foreign)
        windows.append(Window(s, e, max((a[2] for x in inside for a in x.foreign), default=0),
                              len(pids), pids))
    last = bounded[-1][1] if bounded else None
    launches = [(mine[0].time, end) for mine, end in bounded if end > mine[0].time]
    return Contention(_merge(monitored), windows, source, last, is_open, launches=launches)


def _shared(parts: list) -> list:
    """Where launches of two different logs overlap in time, merged (Shared).

    One launcher's own launches never overlap (each takes its lock), but two
    launchers can: the CCM sweep and run_tail.sh cannot (the tail waits for
    the sweep's lock), the CCM sweep and run_tail_arima.sh did. A launch spans
    its first GPU entry to its end (derive), so the stretch is both launches
    running, not both training every second of it: a flag it gives says the
    time may be inflated, as a foreign window's does.
    """
    found = []
    for i, a in enumerate(parts):
        for b in parts[i + 1:]:
            for a0, a1 in a.launches:
                for b0, b1 in b.launches:
                    s, e = max(a0, b0), min(a1, b1)
                    if s < e:
                        found.append((s, e, {a.source or "?", b.source or "?"}))
    found += [(x.start, x.end, set(x.launches.split(" + "))) for p in parts for x in p.shared]
    out: list = []
    for s, e, names in sorted(found, key=lambda t: (t[0], t[1])):
        if out and s <= out[-1][1]:
            out[-1] = (out[-1][0], max(out[-1][1], e), out[-1][2] | names)
        else:
            out.append((s, e, set(names)))
    return [Shared(s, e, " + ".join(sorted(names))) for s, e, names in out]


def combine(parts: list) -> Contention:
    """Several GPU logs' Contentions as one (the CCM sweep's and the tails').

    Windows that touch are merged, a pid seen in two logs counted once; where
    two logs' launches ran at once, a ``shared`` window (_shared). ``source``
    names the logs that monitored any time (a log whose entries all come after
    ``until`` adds nothing).
    """
    parts = [p for p in parts if p is not None]
    if not parts:
        return Contention([], [])
    windows: list = []
    for w in sorted((w for p in parts for w in p.windows), key=lambda w: w.start):
        if windows and w.start <= windows[-1].end:
            last = windows[-1]
            pids = last.pids | w.pids
            n = (len(pids) if (last.pids or not last.processes) and (w.pids or not w.processes)
                 else last.processes + w.processes)
            windows[-1] = Window(last.start, max(last.end, w.end),
                                 max(last.peak_mib, w.peak_mib), n, pids)
        else:
            windows.append(w)
    ends = [p.observed_until for p in parts if p.observed_until is not None]
    return Contention(_merge([m for p in parts for m in p.monitored]), windows,
                      "; ".join(p.source for p in parts if p.source and p.monitored),
                      max(ends) if ends else None, any(p.open for p in parts),
                      shared=_shared(parts),
                      launches=[x for p in parts for x in p.launches])


def flag(concurrency: Optional[int], overlap: Optional[float],
         past_end: bool = False) -> str:
    """A run's contended value: its tokens ';'-joined, or "none".

    ``past_end`` (Contention.past_end): part of its window came after an
    open spec's end, so no overlap seen means not known, not none.
    """
    parts = []
    if concurrency is None:
        parts.append(UNKNOWN)
    elif concurrency > 1:
        parts.append(CONCURRENT)
    if overlap:
        parts.append(FOREIGN)
    elif past_end and UNKNOWN not in parts:
        parts.append(UNKNOWN)
    return ";".join(parts) or NONE


def tokens(values: Iterable[str]) -> str:
    """The distinct tokens of several contended values, ';'-joined in a fixed order."""
    order = (NONE, CONCURRENT, FOREIGN, UNKNOWN)
    seen = {t for v in values if v for t in str(v).split(";")}
    return ";".join(t for t in order if t in seen)


# --------------------------------------------------------------------------- #
# specs/contention.yaml
# --------------------------------------------------------------------------- #

HEADER = """\
# Another user's processes on the GPU while the 2026-09-24 sweeps trained.
#
# Written by `python results/make.py contention` (emfres/contention.py); do not
# edit by hand. Rerun it once a sweep has ended (a DONE line in its sweep log
# closes its last span) and commit the result with the frames it feeds.
#
# Where it came from. Three launchers append the GPU's compute apps (nvidia-smi
# pid, process, memory) to a git-ignored log before each block (the tails:
# before training) and whenever the set of processes changes on a poll (60 s;
# 15 s for the ARIMA tail): .personal/run_ccm_all.sh, the CCM sweep, to
# .personal/ccm_gpu_contention.log; .personal/run_tail.sh, the rerun of the
# August per-channel bands, to .personal/tail_gpu_contention.log; and
# .personal/run_tail_arima.sh, that rerun's ARIMA bands, to
# .personal/tail_arima_gpu.log (`source` names those with an entry by
# `observed_until`). Each entry is marked `own` (a python whose cwd is this
# repo; `own-mps` for the tail's MPS server) or FOREIGN (anything else on the
# card; `make.py contention` prints which executables those were). When this
# file was first derived every FOREIGN entry was another user's python job, a
# string of short jobs. This file copies what the frames need from those logs.
#
# How it was derived. A sampled state holds until the next entry. `monitored`
# is each launch of a launcher's own log (.personal/ccm_sweep.log,
# .personal/tail_sweep.log, .personal/tail_arima.log: a `=== start` line to
# its STOP/DONE line) from its first GPU entry on; a launch still running ends
# at `observed_until`, when this file was derived (`open: true`). `foreign` is
# the merged stretches whose state listed a FOREIGN process, with the largest
# such process (MiB) and how many distinct pids were seen in each. `shared` is
# where launches of two of those logs ran at once: another launch of this
# repo was on the card, which each log marks `own` (the ARIMA tail beside the
# CCM sweep, 2026-09-24 20:45 to 20:52). Edges are known to about a minute, a
# foreign job that came and went between two polls is not seen, and memory
# says nothing about how hard it used the card.
#
# How it is used. `make.py extract` gives every run its training window
# [parameters/exp/time, + running_time] and writes to runs.csv:
#   foreign_overlap  the fraction of that window inside `foreign` or `shared`
#                    (blank: no part of it was monitored)
#   contended        concurrent (its own grid trained several runs at once,
#                    concurrency > 1), foreign (foreign_overlap > 0), both
#                    ';'-joined, none (neither recorded), or ? (not known:
#                    concurrency not stamped, runs before 2026-09-19; or no
#                    overlap seen in a run that trained past `observed_until`
#                    while this file is `open`)
# and to cells.csv each cell's distinct values and n_foreign, its runs with a
# foreign overlap. The timing tables mark such a cell and P7 draws it hollow.
# The flag says the time may be inflated; the accuracy is unaffected. Frames
# extracted while runs trained past `observed_until` of an open file, or while
# the GPU logs go on after it, say so in MANIFEST.json, and `make.py tables`
# will not write from them: after a sweep, rerun `make.py contention`, check
# it with `--check`, extract, and commit this file with the frames.
#
# Earlier trees. No run before the CCM sweep has a GPU record, except the solo
# timing pass of 2026-09-22 (specs/timing.yaml: foreign processes none in all
# 54 of its runs, which are not in logs/), so every earlier run reads none,
# concurrent or ? by its stamps alone. The per-channel grid is concurrent by
# design (8 or 12 bands at a time; part of the 2026-09-21 sweep under CUDA
# MPS, .personal/PLAN.md E10).
#
# `cells` below is a summary for reading, not an input: per cell the runs
# whose window was monitored, how many overlapped `foreign` or `shared`, and
# the mean and largest overlap fraction.
"""


def dump(c: Contention, cells: Optional[list] = None) -> str:
    """specs/contention.yaml's text: HEADER, then the data in flow style."""
    def block(name, rows):
        return [f"{name}: []"] if not rows else [f"{name}:"] + [f"  - {r}" for r in rows]

    lines = [HEADER.rstrip("\n"), f"source: {c.source}",
             f"observed_until: '{_fmt(c.observed_until)}'" if c.observed_until
             else "observed_until: null",
             f"open: {'true' if c.open else 'false'}",
             f"poll_seconds: {POLL_SECONDS}"]
    lines += block("monitored", [f"{{from: '{_fmt(s)}', to: '{_fmt(e)}'}}"
                                 for s, e in c.monitored])
    lines += block("foreign", [f"{{from: '{_fmt(w.start)}', to: '{_fmt(w.end)}', "
                               f"peak_mib: {w.peak_mib}, processes: {w.processes}}}"
                               for w in c.windows])
    lines += block("shared", [f"{{from: '{_fmt(s.start)}', to: '{_fmt(s.end)}', "
                              f"launches: '{s.launches}'}}" for s in c.shared])
    if cells is not None:
        lines += block("cells", ["{" + ", ".join(f"{k}: {v}" for k, v in row.items()) + "}"
                                 for row in cells])
    return "\n".join(lines) + "\n"


def load(path: Path) -> Optional[Contention]:
    """specs/contention.yaml as a Contention, or None when it is absent."""
    if not Path(path).exists():
        return None
    return loads(Path(path).read_text())


def loads(text: str) -> Contention:
    """dump's text back as a Contention (its `cells` summary is not read)."""
    import yaml
    d = yaml.safe_load(text) or {}
    until = d.get("observed_until")
    return Contention(
        monitored=[(_time(m["from"]), _time(m["to"])) for m in d.get("monitored") or []],
        windows=[Window(_time(w["from"]), _time(w["to"]), int(w.get("peak_mib", 0)),
                        int(w.get("processes", 0))) for w in d.get("foreign") or []],
        source=str(d.get("source") or ""),
        observed_until=_time(until) if until else None,
        open=bool(d.get("open")),
        shared=[Shared(_time(s["from"]), _time(s["to"]), str(s.get("launches") or ""))
                for s in d.get("shared") or []])


def summarise(c: Contention, runs: Iterable) -> list[dict]:
    """The spec's `cells` rows: per cell with a monitored run, how its runs overlapped.

    ``runs`` are (setup, runlogs.Run) pairs (frames.load_all).
    """
    by: dict = {}
    for setup, run in runs:
        o = c.overlap(run.started, run.running_time)
        if o is None:
            continue
        key = (setup, run.model, run.dataset, f"L{run.seq_len}/O{run.pred_len}",
               "" if run.k is None else run.k)
        by.setdefault(key, []).append(o)
    rows = []
    for (setup, model, dataset, window, k), os_ in sorted(
            by.items(), key=lambda kv: [str(x) for x in kv[0]]):
        row = {"setup": setup, "model": model, "dataset": dataset, "window": window}
        if k != "":
            row["k"] = k
        row.update(runs=len(os_), foreign=sum(1 for o in os_ if o > 0),
                   mean_overlap=round(sum(os_) / len(os_), 3), max_overlap=round(max(os_), 3))
        rows.append(row)
    return rows
