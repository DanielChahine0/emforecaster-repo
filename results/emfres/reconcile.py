"""The value gate: every value the frozen report publishes, held against today's.

This is the gate that lets markdown stop being an API: if the CSVs reproduce
the numbers the tables already publish, the renderers can be repointed at the
CSVs without anyone having to trust that they were.

Three kinds of published value are held (KINDS):

- ``cells``: the MSE/MAE tables of the four per-setup sections, recomputed
  from frames/cells.csv -- a cluster cell from its fixed-protocol mean, the
  column its section leads with, and one the snapshot printed under the batch
  protocol also from the batch mean, where it has to reproduce;
- ``scoreboard``: the Paradigm Comparison's scoreboard and sensitivity rows
  (the challenger/tie/baseline counts, cells, median gap and Wilcoxon p),
  against today's render of the same section;
- ``sweep``: the CCM sweep section (each model's plain cell, its cell at every
  k, the k it picks), against today's render.

A published value that no longer reproduces is a *change*. Some changes are
the point of a deliberate step -- a retrained band, a new primary protocol --
and specs/gate.yaml declares them, each with its reason, in the block of the
family (emf, voip) whose snapshot it was written against: those print as
expected. Any other change is unexpected and fails the gate, and so is a value
that vanished or turned into '-', whatever a rule says: a rule declares a
value that moved, never one that went missing. A snapshot row whose quantity
today's report prints under another name -- the cluster comparisons, whose
scoreboard rows are now the fixed protocol over more models -- is held against
the row that carries it (HeldElsewhere), and its own row still has to print a
value wherever the snapshot did: what it prints is new, that it prints nothing
is a failure. A value today's report has and the snapshot does not is *new*:
counted, never fatal, since a cell the snapshot never had is a cell it cannot
contradict. Once the expected changes are reviewed, re-freezing the snapshot
is what retires them.

    python results/make.py reconcile
"""

from __future__ import annotations

import csv
import math
import re
import sys
from dataclasses import dataclass, field
from decimal import ROUND_HALF_UP, Decimal
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

# Dataset display -> slug, where the slug is not the display lower-cased
# (extract_results.DATASET_DISPLAY): without these a VoIP cell finds no
# cells.csv row and reads as no longer published.
DATASET_SLUG = {
    "EMF2023 (University-5 w/ PTV)": "emf_2023_university_5_with_ptv",
    "EMF2023": "emf_2023_university_5_with_ptv",
    "VoIP Mobile": "voip_mobile",
    "VoIP Fixed": "voip_fixed",
}


def dataset_slug(label: str) -> str:
    """The cells.csv slug of a dataset as the tables print it."""
    short = re.sub(r"\s*\(.*\)\s*$", "", label).strip() or label
    return DATASET_SLUG.get(label, DATASET_SLUG.get(short, short.lower()))


# The heading of the paradigm comparison; its other windows render as
# "Paradigm Comparison · L24/O6".
PARADIGM_TITLE = "Paradigm Comparison"

# What identifies one published value, per kind. A specs/gate.yaml rule
# matches on these names (and k_min), nothing else.
KINDS = {
    "cells": ("setup", "model", "dataset", "window", "k", "metric"),
    "scoreboard": ("section", "table", "comparison", "variant", "metric", "column"),
    "sweep": ("dataset", "window", "metric", "model", "column", "k"),
}

# Where each kind's values are published, for the per-section notes.
KIND_WHERE = {
    "cells": "MSE/MAE tables vs frames/cells.csv",
    "scoreboard": "scoreboard and sensitivity rows vs today's render",
    "sweep": "plain, per-k and chosen-k values vs today's render",
}

# What a section held by each kind has that no kind holds, for the same notes:
# a section partly gated says so, rather than read as held throughout.
KIND_TEXT_ONLY = {
    "cells": "its tables other than MSE/MAE are text only",
    "scoreboard": "its per-cell tables are text only",
}


def _blank(value: str | None) -> bool:
    """Not printed at all, or printed as '-' (with or without a note after it)."""
    return value is None or value == "-" or value.startswith("- ")


# What a row held elsewhere is still held to under its own name. Its values
# there are a quantity the snapshot never published, so they are not compared;
# but where the snapshot printed a value, today's row has to print one too.
OWN_ROW = "its own row, which still has to be printed"


@dataclass
class Change:
    """One published value that no longer reproduces."""

    kind: str
    key: dict
    published: str
    current: str | None          # None: today's report no longer publishes it
    via: str = ""                # the row it was held against, if not its own

    @property
    def vanished(self) -> bool:
        """Today's report no longer prints the value, or prints '-' for it.

        No rule forgives this (gate): every change specs/gate.yaml declares
        moves a value, none removes one, so a value that vanished under a rule
        is a failure the rule was not written for.
        """
        return _blank(self.current)

    @property
    def ident(self) -> tuple:
        """The published value this is a change of. A value held elsewhere can
        change twice: at the row it is held against, and by vanishing from its
        own."""
        return (self.kind,) + tuple(self.key.values())

    def where(self) -> str:
        k = self.key
        if self.kind == "cells":
            return (f"{k['setup']}/{k['model']}/{k['dataset']} {k['window']}"
                    f"{' k' + k['k'] if k['k'] else ''} [{k['metric']}]")
        if self.kind == "scoreboard":
            variant = f" / {k['variant']}" if k["variant"] else ""
            return (f"{k['section']}: {k['comparison']}{variant} [{k['metric']}] "
                    f"{k['column']}")
        col = f"k={k['k']}" if k["column"] == "k" else k["column"]
        return f"sweep {k['dataset']} {k['window']} [{k['metric']}] {k['model']} {col}"

    def label(self) -> str:
        now = "no longer published" if self.current is None else self.current
        via = f" (held against {self.via})" if self.via else ""
        return f"{self.where()}{via}: {self.published} -> {now}"


# --------------------------------------------------------------------------- #
# Comparing two printed values
# --------------------------------------------------------------------------- #

_NUM = re.compile(r"[+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?")


def _quantum(token: str) -> Decimal:
    """The step a printed number is known to.

    The smaller of its printed quantum and that of its third significant
    digit: a value printed with ``:.3g`` (a Wilcoxon p) drops trailing zeros,
    so "1" is 1.00, known to 0.01, and "0.5" is 0.500, known to 0.001 -- not
    a number known only to the unit or the tenth. "0.3765" is known to 1e-4,
    "7.6e-06" to 1e-8, and a zero to the places it is printed with.
    """
    d = Decimal(token)
    printed = Decimal(1).scaleb(d.as_tuple().exponent)
    return printed if d.is_zero() else min(printed, Decimal(1).scaleb(d.adjusted() - 2))


def _num_same(a: str, b: str) -> bool:
    """Two printed numbers agree at the coarser of the steps they are known to.

    So "1" does not match "0.5" (1.00 against 0.50), while "0.3765" matches
    "0.37649" and "1235" matches "1234.6": printed to more places, a value
    still agrees at the fewer places the other side printed. A signed zero is
    no change.
    """
    q = max(_quantum(a), _quantum(b))
    return (Decimal(a).quantize(q, rounding=ROUND_HALF_UP)
            == Decimal(b).quantize(q, rounding=ROUND_HALF_UP))


def same(published: str, current: str) -> bool:
    """Whether a value, as printed today, is the value the snapshot printed.

    The text around the numbers has to match exactly -- a tie mark, a sign
    of incompleteness, an sd that appeared or vanished is a change -- and the
    numbers have to agree at the precision the snapshot (or today, if it
    prints fewer digits) printed them with, where a number is known at least
    to its third significant digit (_num_same).
    """
    if published == current:
        return True
    if _NUM.sub("#", published) != _NUM.sub("#", current):
        return False
    return all(_num_same(a, b) for a, b in zip(_NUM.findall(published),
                                               _NUM.findall(current)))


# --------------------------------------------------------------------------- #
# Reading the markdown
# --------------------------------------------------------------------------- #

def split_blocks(text: str) -> list[tuple[str, str]]:
    """The report's top-level sections, as (heading, markdown) in order."""
    out = []
    for block in re.split(r"(?m)^(?=# )", text):
        if block.startswith("# "):
            out.append((block.splitlines()[0][2:].strip(), block.strip()))
    return out


def titled(heading: str, title: str) -> bool:
    """A heading that is ``title``, or ``title`` narrowed by a " · " suffix.

    The CCM sweep used to be rendered one dataset and metric at a time
    ("Channel Clustering (CCM) vs. plain · ED2022 · MAE"), and the paradigm
    comparison's other windows still are ("Paradigm Comparison · L24/O6").
    """
    return heading == title or heading.startswith(title + " · ")


def find_blocks(blocks: list[tuple[str, str]], title: str,
                claimed: frozenset = frozenset()) -> list[tuple[str, str]]:
    """The snapshot sections a rendered section corresponds to.

    An exact heading wins. Otherwise every heading that is ``title`` plus a
    " · " suffix and that no other rendered section claims exactly: so the
    old per-dataset sweep headings find today's one sweep, while
    "Paradigm Comparison · L24/O6" never stands in for "Paradigm Comparison"
    when both are rendered.
    """
    exact = [b for b in blocks if b[0] == title]
    if exact:
        return exact
    return [b for b in blocks if titled(b[0], title) and b[0] not in claimed]


def _cell(text: str, keep_bold: bool = False) -> str:
    text = re.sub(r"</?u>", "", text).replace("`", "").replace("$", "")
    if not keep_bold:
        text = text.replace("**", "")
    return " ".join(text.split())


def _row(line: str) -> list[str]:
    return [c.strip() for c in line.strip().strip("|").split("|")]


def _tables(block: str):
    """Each markdown table of a block: (the headings above it, header, rows).

    Cells are returned raw (emphasis intact); the caller decides what markup
    is part of a value.
    """
    heads: dict[int, str] = {}
    header = None
    rows: list[list[str]] = []
    out = []
    for line in block.splitlines() + [""]:
        if line.startswith("|"):
            cells = _row(line)
            if header is None:
                header = [_cell(c) for c in cells]
            elif not (set("".join(cells)) <= set("-: ")):
                rows.append(cells)
            continue
        if header is not None:
            out.append((dict(heads), header, rows))
            header, rows = None, []
        m = re.match(r"^(#{1,3}) (.*)$", line)
        if m:
            level = len(m.group(1))
            heads = {lv: h for lv, h in heads.items() if lv < level}
            heads[level] = m.group(2).strip()
    return out


def _short_model(name: str) -> str:
    """``DLinear (2023)`` -> ``DLinear`` (as plot_metrics.short_model)."""
    return re.sub(r"\s*\(\d{4}\)\s*$", "", _cell(name)).strip()


def _short_dataset(name: str) -> str:
    """``EMF2023 (University-5 w/ PTV)`` -> ``EMF2023`` (as plot_metrics.short_dataset)."""
    name = _cell(name)
    return re.sub(r"\s*\(.*\)\s*$", "", name).strip() or name


def scoreboard_values(text: str) -> dict[tuple, str]:
    """The paradigm comparison's scoreboard and sensitivity rows, one value per cell.

    A row is keyed by its comparison ("Per-channel vs Multivariate", the part of
    the scoreboard's question before the dash), its variant (sensitivity rows
    only) and its metric -- not by its wording, so rephrasing a question is not
    a change -- and each count, gap and p is its own value.
    """
    out: dict[tuple, str] = {}
    for heading, block in split_blocks(text):
        if not titled(heading, PARADIGM_TITLE):
            continue
        for _, header, rows in _tables(block):
            if header[:1] == ["Question"]:
                table, lead = "scoreboard", ("Question",)
            elif header[:2] == ["Comparison", "Variant"]:
                table, lead = "sensitivity", ("Comparison", "Variant")
            else:
                continue
            if "Metric" not in header:
                continue
            for raw in rows:
                row = [_cell(c) for c in raw]
                if len(row) != len(header):
                    continue
                cols = dict(zip(header, row))
                comparison = row[0].split(" — ")[0].strip()
                variant = cols.get("Variant", "")
                for col, value in cols.items():
                    if col in lead or col == "Metric":
                        continue
                    out.setdefault(("scoreboard", heading, table, comparison, variant,
                                    cols["Metric"], col), value)
    return out


def _k_of(text: str) -> str:
    m = re.search(r"\bk\s*=\s*(\d+)", text)
    return m.group(1) if m else "-"


def sweep_values(text: str) -> dict[tuple, str]:
    """The CCM sweep's values: each model's plain cell, its cell at every k, its k.

    Reads both layouts the sweep has had. Today's (one section; a table per
    metric, dataset and window; a row per model, a column per k) publishes the
    k chosen on validation and the oracle k, the one the test split picks.
    The one frozen before 2026-09-24 (a section per dataset and metric; a row
    per k, a plain and a "+ CCM" column per model) underlined the k with the
    best test value, which is that same oracle k.
    """
    from .tables import SWEEP_TITLE

    out: dict[tuple, str] = {}

    def put(dataset, window, metric, model, column, k, value):
        out.setdefault(("sweep", dataset, window, metric, model, column, k), value)

    for heading, block in split_blocks(text):
        if not titled(heading, SWEEP_TITLE):
            continue
        # The old layout names dataset and metric in the heading, and the
        # window in a bullet.
        parts = [p.strip() for p in heading.split(" · ")[1:]]
        m = re.search(r"L\s*=\s*(\d+),\s*O\s*=\s*(\d+)", _cell(block))
        window = f"L{m.group(1)}/O{m.group(2)}" if m else ""
        for heads, header, rows in _tables(block):
            if header[:2] == ["Model", "Plain"]:
                metric = heads.get(2, "").split("·")[-1].strip()
                where = heads.get(3, "").split("·", 1)[-1].strip()
                label, _, win = where.rpartition(", ")
                dataset = _short_dataset(label)
                for raw in rows:
                    cols = dict(zip(header, raw))
                    model = _short_model(cols["Model"])
                    put(dataset, win, metric, model, "plain", "", _cell(cols["Plain"]))
                    for col, value in cols.items():
                        if re.fullmatch(r"k=\d+", col):
                            put(dataset, win, metric, model, "k", col[2:], _cell(value))
                    val_k = _k_of(_cell(cols.get("Val k", "")))
                    put(dataset, win, metric, model, "val k", "", val_k)
                    oracle = _cell(cols.get("Oracle k", "-"))
                    put(dataset, win, metric, model, "oracle k", "",
                        val_k if oracle == "= val k" else (oracle.split() or ["-"])[0])
                    for col in ("Δ vs k=1", "Δ vs plain"):
                        if col in cols:
                            put(dataset, win, metric, model, col, "",
                                _cell(cols[col], keep_bold=True))
            elif header[:1] == ["k"] and len(parts) == 2:
                dataset, metric = _short_dataset(parts[0]), parts[1]
                models = [h for h in header[1:] if not h.endswith(" + CCM")]
                for raw in rows:
                    cols = dict(zip(header, raw))
                    k = _cell(cols["k"])
                    for name in models:
                        model = _short_model(name)
                        if name in cols:
                            put(dataset, window, metric, model, "plain", "",
                                _cell(cols[name]))
                        ccm = cols.get(f"{name} + CCM")
                        if ccm is None:
                            continue
                        put(dataset, window, metric, model, "k", k, _cell(ccm))
                        if "<u>" in ccm:
                            put(dataset, window, metric, model, "oracle k", "", k)
    return out


# --------------------------------------------------------------------------- #
# The per-setup cells, against frames/cells.csv
# --------------------------------------------------------------------------- #

def load_archived(spec: Path) -> list[dict]:
    """Deliberately-unpublished cells declared in specs/runs.yaml.

    A rule matches on (setup, dataset, lookback, horizon); every field it
    omits is a wildcard (is_archived). Returned as-is so an unreadable spec
    degrades to "no allowances" -- the gate then over-reports rather than
    under-reports, which is the safe direction for a tripwire.
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
    """Whether a specs/runs.yaml `archived` rule withdraws this cell.

    ``dataset`` is the label the tables print (ED2022, VoIP Mobile); a rule
    may name it by that label or by its cells.csv slug (ed2022, voip_mobile),
    in any case. The rules are not kept per family: a rule applies wherever
    its dataset is published, and one naming no dataset applies to every
    family's snapshot.
    """
    names = {dataset.casefold(), dataset_slug(dataset).casefold()}
    for rule in rules:
        if "setups" in rule and setup not in rule["setups"]:
            continue
        if "dataset" in rule and str(rule["dataset"]).casefold() not in names:
            continue
        if "lookbacks" in rule and lookback not in rule["lookbacks"]:
            continue
        if "horizon" in rule and horizon != rule["horizon"]:
            continue
        return True
    return False


# The cells.csv column each setup's section prints a metric from, where it is
# not the metric's own (mse, mae): a cluster section leads with the fixed
# train-mean membership, the primary cluster protocol, and prints the logged
# batch protocol only as its last group (extract_results.render_setup).
PRIMARY_COLUMN = {"cluster": {"MSE": "mse_fixed", "MAE": "mae_fixed"}}

# Where a cluster value the snapshot printed under the batch protocol is held
# besides today's primary: the batch columns, whose runs no protocol change
# touches. A change there is never one a rule forgives (gate).
BATCH_VIA = "cells.csv's batch-protocol mse/mae, the protocol the snapshot printed"


def primary_column(setup: str, metric: str) -> str:
    """The cells.csv column a setup's section publishes ``metric`` (MSE/MAE) from."""
    return PRIMARY_COLUMN.get(setup, {}).get(metric, metric.lower())


def cluster_printed_batch(text: str) -> bool:
    """Whether a report's cluster section led with the logged batch protocol.

    Its first MSE table names the log key it printed: `sl_test/loss` (every
    snapshot frozen before the fixed protocol became primary) or
    `sl_test_fixed/loss`.
    """
    for heading, block in split_blocks(text):
        if heading != "Cluster (k-Grouped Channels)":
            continue
        for line in block.splitlines():
            m = re.match(r"^### 1\.1\b(.*)$", line)
            if m and re.search(r"\bMSE\b", m.group(1)):
                return "`sl_test/loss`" in m.group(1)
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


@dataclass
class Held:
    """What one kind of check found: published values, changes, new values."""

    published: int = 0
    changes: list = field(default_factory=list)
    new: list = field(default_factory=list)
    notes: list = field(default_factory=list)
    by_section: dict = field(default_factory=dict)   # section -> [published, changed]
    elsewhere: dict = field(default_factory=dict)    # section -> {via: values held there}

    def tally(self, section: str, changed: bool) -> None:
        self.published += 1
        n = self.by_section.setdefault(section, [0, 0])
        n[0] += 1
        n[1] += changed

    @property
    def changed(self) -> int:
        """Published values that changed (a value can have two changes)."""
        return sum(c for _, c in self.by_section.values())


def hold_cells(tables: Path, cells_csv: Path, archived_rules: list[dict]) -> Held:
    """Every published MSE/MAE cell of the per-setup sections, from cells.csv.

    Each is held against the column its section prints today (primary_column):
    a cluster cell against its fixed-protocol mean. A cluster cell the snapshot
    printed under the logged batch protocol (cluster_printed_batch) is held
    against the batch column as well, where it has to reproduce: today's
    section prints it in its batch-protocol group, and no rule forgives a
    change there.
    """
    results = str(Path(__file__).resolve().parent.parent)
    if results not in sys.path:
        sys.path.insert(0, results)
    import plot_metrics as pm

    cells = load_cells(cells_csv)
    held = Held()
    absent = archived = 0
    batch_printed = cluster_printed_batch(tables.read_text())
    n_batch = 0

    def lookup(setup, model_disp, dataset, lookback, horizon, k):
        slug = dataset_slug(dataset)
        for mid in DISPLAY_IDS.get(model_disp, [model_disp]):
            for conf in (1, 0):
                row = cells.get((setup, mid, slug, lookback, horizon, k, conf))
                if row:
                    return row
        return None

    for r in pm.parse_tables(tables):
        setup = SECTION_SETUP.get(r.section)
        if setup is None or r.metric not in ("MSE", "MAE"):
            continue          # training time is a formatted duration, not held here
        model = pm.short_model(r.model)
        ds = pm.short_dataset(r.dataset)
        k = "" if r.cluster is None else str(r.cluster)
        key = dict(setup=setup, model=model, dataset=ds,
                   window=f"L{r.lookback}/O{r.horizon}", k=k, metric=r.metric)
        row = lookup(setup, model, r.dataset, r.lookback, r.horizon, k)
        column = primary_column(setup, r.metric)
        if math.isnan(r.value):
            # A '-' in the table: that grid cell was never run. A frames row
            # for it now is a value the snapshot never had -- new, not a change.
            absent += 1
            got = row and row.get(column, "")
            if got:
                held.new.append(Change("cells", key, "-", f"{float(got):.4f}"))
            continue
        if row is None and is_archived(archived_rules, setup, ds, r.lookback, r.horizon):
            # Published once, deliberately withdrawn since. Not held: there is
            # nothing left to hold it against.
            archived += 1
            continue
        published = f"{r.value:.4f}"
        # Against the column the section prints it from today, and a cluster
        # value the snapshot printed under the batch protocol also against
        # that protocol's column, where its runs still have to reproduce it.
        against = [(column, "")]
        if setup == "cluster" and batch_printed:
            against.append((r.metric.lower(), BATCH_VIA))
            n_batch += 1
        found = []
        for col, via in against:
            got = "" if row is None else (row.get(col) or "")
            if got == "" or round(float(got), 4) != round(r.value, 4):
                found.append(Change("cells", key, published,
                                    None if got == "" else f"{float(got):.4f}", via))
        held.tally(r.section, bool(found))
        held.changes += found
    if absent:
        held.notes.append(f"{absent} cells are '-' in the snapshot (never run); "
                          f"{len(held.new)} of them have a frames row now")
    if n_batch:
        held.notes.append(f"{n_batch} cluster cells the snapshot printed under the batch "
                          f"protocol: held against today's fixed-protocol cells, the "
                          f"section's primary, and against {BATCH_VIA}")
    if archived or archived_rules:
        held.notes.append(f"{archived} cells withdrawn by specs/runs.yaml "
                          f"'archived' rules -- not held")
    return held


def hold_rendered(kind: str, published_text: str, current_text: str, parse,
                  section_of, elsewhere=None) -> Held:
    """A kind whose values are read from the snapshot and today's render alike.

    ``elsewhere(key)`` is (today's value, where it was read) for a snapshot
    value held against another row than its own (specs/gate.yaml
    held_elsewhere, see gate), or None. Today's value under that snapshot key
    is then one the snapshot never published: what it is counts as new, but
    that it is there is still held (OWN_ROW). Where the snapshot printed a
    value, a key missing from today's row, or printed '-' there, is a change
    no rule forgives, however well the row it is held against reproduces.
    """
    names = KINDS[kind]
    before, after = parse(published_text), parse(current_text)
    held = Held()
    new = set(after.keys() - before.keys())
    for key, value in before.items():
        fields = dict(zip(names, key[1:]))
        other = elsewhere(key) if elsewhere else None
        now, via = (after.get(key), "") if other is None else other
        found = []
        if now is None or not same(value, now):
            found.append(Change(kind, fields, value, now, via))
        if other is not None:
            n = held.elsewhere.setdefault(section_of(key), {})
            n[via] = n.get(via, 0) + 1
            own = Change(kind, fields, value, after.get(key), OWN_ROW)
            if not own.vanished:
                new.add(key)
            elif not _blank(value):
                found.append(own)
        held.tally(section_of(key), bool(found))
        held.changes += found
    for key in sorted(new):
        held.new.append(Change(kind, dict(zip(names, key[1:])), "", after[key]))
    return held


# --------------------------------------------------------------------------- #
# Expected changes, from specs/gate.yaml
# --------------------------------------------------------------------------- #

# The keys of one family's block of specs/gate.yaml.
BLOCK_FIELDS = ("snapshot", "held_elsewhere", "expected")


def _spec_entries(spec: Path | None, snapshot: str, family: str, name: str,
                  what: str) -> tuple[list, list[str]]:
    """The list ``name`` of ``family``'s block of specs/gate.yaml, if that
    block was written for ``snapshot``.

    specs/gate.yaml holds one block per family (emf, voip), each naming the
    snapshot its lists were written for. A family with no block expects
    nothing. A list outside every block is not read, with a warning: which
    snapshot it was written for is anyone's guess, and dropping it can only
    make the gate stricter.
    """
    if spec is None or not spec.exists():
        return [], []
    try:
        import yaml
        doc = yaml.safe_load(spec.read_text()) or {}
    except Exception as exc:
        return [], [f"WARNING: could not read {spec}: {exc}; none of its {what} applies"]
    if not isinstance(doc, dict):
        return [], [f"WARNING: {spec.name} is not a mapping of family blocks; none of "
                    f"its {what} applies"]
    notes = []
    if name in doc:
        notes.append(f"WARNING: {spec.name} has `{name}` outside a family block; it is "
                     f"not read (put it under the family's key, e.g. `{family}:`)")
    block = doc.get(family)
    if block is None:
        return [], notes
    if not isinstance(block, dict):
        return [], notes + [f"WARNING: {spec.name}'s `{family}` block is not a mapping; "
                            f"none of its {what} applies"]
    written_for = str(block.get("snapshot") or "")
    entries = list(block.get(name) or [])
    if entries and written_for != snapshot:
        return [], notes + [f"NOTE: {spec.name}'s {len(entries)} {family} {what} were "
                            f"written against {written_for or 'no snapshot'}, not "
                            f"{snapshot}; none applies"]
    return entries, notes


def load_expected(spec: Path | None, snapshot: str,
                  family: str) -> tuple[list[dict], list[str]]:
    """The rules of ``family``'s block of specs/gate.yaml that apply to
    ``snapshot``, and notes.

    A rule written for another snapshot, a rule with no reason, a rule naming
    a field its kind does not have and a rule whose ``now`` is malformed
    (_pins) are all left out, with a note: each can only make the gate
    stricter by being dropped, never looser.
    """
    rules, notes = _spec_entries(spec, snapshot, family, "expected", "expected changes")
    kept = []
    for i, rule in enumerate(rules, 1):
        if not isinstance(rule, dict) or rule.get("kind") not in KINDS:
            notes.append(f"WARNING: {spec.name} rule {i}: kind must be one of "
                         f"{', '.join(KINDS)}; ignoring it")
            continue
        if not str(rule.get("reason") or "").strip():
            notes.append(f"WARNING: {spec.name} rule {i}: no reason; ignoring it")
            continue
        unknown = (set(rule) - set(KINDS[rule["kind"]])
                   - {"kind", "reason", "k_min", "now"})
        if unknown:
            notes.append(f"WARNING: {spec.name} rule {i}: {rule['kind']} values have no "
                         f"field {', '.join(sorted(unknown))}; ignoring it")
            continue
        if "now" in rule and _pins(rule) is None:
            notes.append(f"WARNING: {spec.name} rule {i}: `now` must be one printed "
                         f"value, or a list of entries each with a `value` and only "
                         f"{rule['kind']} fields; ignoring it")
            continue
        kept.append(dict(rule, _n=i))
    return kept, notes


def _pins(rule: dict) -> list[dict] | None:
    """What a rule's ``now`` pins, as entries of (fields..., value); None if malformed.

    ``now`` is one printed value, which every value the rule matches has to
    print today, or a list of entries, each naming some of the kind's fields
    and the ``value`` the values it matches print today: so a rule forgives
    the transition its reason declares, not any value in the columns it names.
    """
    now = rule.get("now")
    if isinstance(now, (str, int, float)) and not isinstance(now, bool):
        return [{"value": str(now)}]
    names = set(KINDS.get(rule.get("kind"), ()))
    if (not isinstance(now, list) or not now
            or not all(isinstance(e, dict) and "value" in e and set(e) - {"value"} <= names
                       for e in now)):
        return None
    return [dict(e, value=str(e["value"])) for e in now]


def _fields_match(spec: dict, key: dict) -> bool:
    """Every field ``spec`` names (a value or a list, any case) is the key's."""
    for name, want in spec.items():
        allowed = want if isinstance(want, list) else [want]
        if name not in key or (str(key[name]).casefold()
                               not in {str(a).casefold() for a in allowed}):
            return False
    return True


def matches(rule: dict, change: Change) -> bool:
    """Whether ``rule`` declares ``change``: its fields and k_min match the
    changed value, and, if it pins ``now``, today's value is the one pinned.
    No rule declares a cluster value that moved in the batch protocol the
    snapshot printed it under (BATCH_VIA): those runs did not change."""
    if rule["kind"] != change.kind or change.via == BATCH_VIA:
        return False
    if not _fields_match({n: rule[n] for n in KINDS[change.kind] if n in rule}, change.key):
        return False
    if "k_min" in rule:
        k = str(change.key.get("k", ""))
        if not k.isdigit() or int(k) < int(rule["k_min"]):
            return False
    if "now" in rule:
        pins = _pins(rule) or []
        return change.current is not None and any(
            _fields_match({n: v for n, v in pin.items() if n != "value"}, change.key)
            and same(pin["value"], change.current) for pin in pins)
    return True


@dataclass(frozen=True)
class HeldElsewhere:
    """Snapshot scoreboard rows held against another row of today's report.

    For a comparison whose quantity today's scoreboard no longer prints under
    its name. The snapshot's cluster rows are the logged batch protocol over
    the models it clustered; today's rows of the same name are the fixed
    protocol over more models, values the snapshot never published. The
    snapshot's quantity lives on as today's ``variant`` sensitivity row, and
    re-counted over ``models`` alone it has to reproduce: each value is held
    there. Today's scoreboard value under the old name counts as new, but it
    has to be there: missing or '-' where the snapshot printed a value is a
    change no rule forgives (hold_rendered).
    """

    comparisons: tuple         # casefolded, as the scoreboard key has them
    variant: str               # the sensitivity row's variant, as the table prints it
    models: tuple              # model ids, as runs.csv has them
    reason: str
    n: int

    def covers(self, key: tuple) -> bool:
        # ("scoreboard", section, table, comparison, variant, metric, column)
        return (key[0] == "scoreboard" and key[2] == "scoreboard" and not key[4]
                and key[3].casefold() in self.comparisons)

    def target(self, key: tuple) -> tuple:
        return key[:2] + ("sensitivity", key[3], self.variant) + key[5:]

    def via(self) -> str:
        return f"today's '{self.variant}' row over {', '.join(self.models)}"


HELD_FIELDS = ("comparison", "variant", "models", "reason")


def load_held_elsewhere(spec: Path | None, snapshot: str,
                        family: str) -> tuple[list[HeldElsewhere], list[str]]:
    """The held_elsewhere entries of ``family``'s block of specs/gate.yaml
    that apply to ``snapshot``.

    An entry without all of HELD_FIELDS, or with any other, is left out with a
    warning: its rows are then held under their own name, where they have to
    reproduce like any other value.
    """
    entries, notes = _spec_entries(spec, snapshot, family, "held_elsewhere",
                                   "held_elsewhere entries")

    def listed(v) -> tuple:
        return tuple(str(x) for x in (v if isinstance(v, list) else [v]) if str(x).strip())

    kept = []
    for i, e in enumerate(entries, 1):
        if (not isinstance(e, dict) or set(e) != set(HELD_FIELDS)
                or not all(listed(e[f]) for f in HELD_FIELDS)):
            notes.append(f"WARNING: {spec.name} held_elsewhere {i}: needs exactly "
                         f"{', '.join(HELD_FIELDS)}, each non-empty; ignoring it")
            continue
        kept.append(HeldElsewhere(tuple(c.casefold() for c in listed(e["comparison"])),
                                  _cell(str(e["variant"])), listed(e["models"]),
                                  str(e["reason"]), i))
    return kept, notes


def _headline(reason: str) -> str:
    text = " ".join(str(reason).split())
    first = re.split(r"(?<=\.)\s", text, maxsplit=1)[0]
    return first if len(first) <= 110 else first[:107] + "..."


# --------------------------------------------------------------------------- #
# The gate
# --------------------------------------------------------------------------- #

@dataclass
class Gate:
    family: str
    snapshot: str
    held: dict                       # kind -> Held
    expected: list                   # (rule, [Change])
    unexpected: list                 # [Change]
    notes: list

    @property
    def ok(self) -> bool:
        return not self.unexpected

    @property
    def n_expected(self) -> int:
        return sum(len(cs) for _, cs in self.expected)

    @property
    def n_unexpected(self) -> int:
        """Published values with a change no rule expects."""
        return len({c.ident for c in self.unexpected})

    def coverage(self, title: str) -> str:
        """What the gate held of one rendered section, for the text-drift note."""
        for kind, held in self.held.items():
            if title not in held.by_section:
                continue
            n, changed = held.by_section[title]
            where = "(" + KIND_WHERE[kind] + "".join(
                f"; {m} of them against {via}, and their own row only for being printed"
                for via, m in held.elsewhere.get(title, {}).items()) + ")"
            rest = f"; {KIND_TEXT_ONLY[kind]}" if kind in KIND_TEXT_ONLY else ""
            if not changed:
                return f"its {n} gated values {where} all reproduce{rest}"
            bad = len({c.ident for c in self.unexpected if _section(c) == title})
            return (f"{changed} of its {n} gated values {where} changed: "
                    f"{changed - bad} expected, {bad} unexpected{rest}")
        return "text only -- none of its values is gated"


def _section(change: Change) -> str:
    """The rendered section a change is published in (Held.by_section's key)."""
    from .tables import SWEEP_TITLE

    if change.kind == "cells":
        return next(t for t, s in SECTION_SETUP.items() if s == change.key["setup"])
    if change.kind == "scoreboard":
        return change.key["section"]
    return SWEEP_TITLE


def gate(published: Path, cells_csv: Path, sections: list[tuple[str, str, str]], *,
         family: str, runs_spec: Path | None = None, gate_spec: Path | None = None,
         render_models=None, verbose: bool = False, out=None) -> Gate:
    """Hold every value ``published`` prints against today's frames and render.

    ``published`` is ``family``'s snapshot, and only that family's block of
    specs/gate.yaml applies to it. ``sections`` is today's report, as make.py
    tables renders it: (slug, title, markdown). ``render_models(models)`` is
    today's paradigm comparison re-counted over only those model ids
    (cli.render_paradigms), what the block's held_elsewhere rows are held
    against. Prints the account (to ``out``, sys.stdout as it is at the call)
    and returns it; the gate passes when no change is unexpected.
    """
    from .tables import SWEEP_TITLE

    specs = Path(__file__).resolve().parent.parent / "specs"
    runs_spec = specs / "runs.yaml" if runs_spec is None else runs_spec
    gate_spec = specs / "gate.yaml" if gate_spec is None else gate_spec

    text = published.read_text()
    current = "\n\n".join(md.strip() for _, _, md in sections)
    stand_ins, notes = load_held_elsewhere(gate_spec, published.name, family)
    if stand_ins and render_models is None:
        notes.append("WARNING: nothing to re-count the held_elsewhere rows with: each "
                     "of their values reads as no longer published")
    recounted: dict = {}

    def elsewhere(key):
        s = next((s for s in stand_ins if s.covers(key)), None)
        if s is None:
            return None
        if s.models not in recounted:
            recounted[s.models] = (scoreboard_values(render_models(s.models))
                                   if render_models else {})
        return recounted[s.models].get(s.target(key)), s.via()

    held = {
        "cells": hold_cells(published, cells_csv, load_archived(runs_spec)),
        "scoreboard": hold_rendered("scoreboard", text, current, scoreboard_values,
                                    lambda key: key[1], elsewhere),
        "sweep": hold_rendered("sweep", text, current, sweep_values,
                               lambda key: SWEEP_TITLE),
    }
    for s in stand_ins:
        n = sum(v.get(s.via(), 0) for v in held["scoreboard"].elsewhere.values())
        held["scoreboard"].notes.append(
            f"{n} of them held against {s.via()}; their own row only has to print "
            f"them (specs/gate.yaml held_elsewhere {s.n}: {_headline(s.reason)})"
            + ("" if n else " -- matched nothing"))
    rules, more = load_expected(gate_spec, published.name, family)
    notes += more
    expected = [(rule, []) for rule in rules]
    unexpected = []
    for h in held.values():
        for change in h.changes:
            hit = None if change.vanished else next(
                (cs for rule, cs in expected if matches(rule, change)), None)
            (unexpected if hit is None else hit).append(change)
    result = Gate(family, published.name, held, expected, unexpected, notes)
    report(result, verbose=verbose, out=out)
    return result


def report(g: Gate, *, verbose: bool = False, out=None) -> None:
    out = sys.stdout if out is None else out
    print(f"value gate ({g.family}): {g.snapshot} against today's frames", file=out)
    for kind, h in g.held.items():
        changed = h.changed
        print(f"  {kind:<10} {h.published:>4} published, {h.published - changed} "
              f"reproduce, {changed} changed"
              + (f", {len(h.new)} new" if h.new else "")
              + f" -- {KIND_WHERE[kind]}", file=out)
        for note in h.notes:
            print(f"  {'':<10} {note}", file=out)
    for note in g.notes:
        print(note, file=out)

    cap = None if verbose else 8
    if g.expected:
        print(f"expected changes (specs/gate.yaml {g.family}, snapshot {g.snapshot}): "
              f"{g.n_expected}", file=out)
        for rule, cs in g.expected:
            print(f"  rule {rule['_n']}: {len(cs)} -- {_headline(rule['reason'])}"
                  + ("" if cs else " (matched nothing)"), file=out)
            for c in cs[:cap]:
                print(f"      {c.label()}", file=out)
            if cap is not None and len(cs) > cap:
                print(f"      ... and {len(cs) - cap} more (make.py reconcile --all)",
                      file=out)
    print(f"unexpected changes: {len(g.unexpected)}", file=out)
    for c in g.unexpected[:None if verbose else 40]:
        print(f"  {c.label()}", file=out)
    if not verbose and len(g.unexpected) > 40:
        print(f"  ... and {len(g.unexpected) - 40} more (make.py reconcile --all)", file=out)
    gone = sum(c.vanished for c in g.unexpected)
    if gone:
        print(f"  ({gone} of them vanished or turned into '-': no rule forgives that)",
              file=out)
    if verbose:
        for h in g.held.values():
            for c in h.new:
                print(f"  new: {c.where()}: {c.current}", file=out)
    if g.ok:
        print("PASS: every published value reproduces"
              + (f" or is one of the {g.n_expected} declared changes" if g.n_expected
                 else ""), file=out)
    else:
        print(f"FAIL: {g.n_unexpected} published values changed and no rule in "
              f"specs/gate.yaml expects it", file=out)
    out.flush()           # before make.py's own messages on stderr
