"""The value gate: every value the frozen report publishes, held against today's.

This is the gate that lets markdown stop being an API: if the CSVs reproduce
the numbers the tables already publish, the renderers can be repointed at the
CSVs without anyone having to trust that they were.

Three kinds of published value are held (KINDS):

- ``cells``: the MSE/MAE tables of the four per-setup sections, recomputed
  from frames/cells.csv;
- ``scoreboard``: the Paradigm Comparison's scoreboard and sensitivity rows
  (the challenger/tie/baseline counts, cells, median gap and Wilcoxon p),
  against today's render of the same section;
- ``sweep``: the CCM sweep section (each model's plain cell, its cell at every
  k, the k it picks), against today's render.

A published value that no longer reproduces is a *change*. Some changes are
the point of a deliberate step -- a retrained band, a new primary protocol --
and specs/gate.yaml declares them, each with its reason, for the snapshot it
was written against: those print as expected. Any other change is unexpected
and fails the gate, and so is a value that vanished or turned into '-',
whatever a rule says: a rule declares a value that moved, never one that went
missing. A value today's report has and the snapshot does not is
*new*: counted, never fatal, since a cell the snapshot never had is a cell it
cannot contradict. Once the expected changes are reviewed, re-freezing the
snapshot is what retires them.

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

# Dataset display -> slug.
DATASET_SLUG = {
    "EMF2023 (University-5 w/ PTV)": "emf_2023_university_5_with_ptv",
    "EMF2023": "emf_2023_university_5_with_ptv",
}

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


@dataclass
class Change:
    """One published value that no longer reproduces."""

    kind: str
    key: dict
    published: str
    current: str | None          # None: today's report no longer publishes it

    @property
    def vanished(self) -> bool:
        """Today's report no longer prints the value, or prints '-' for it.

        No rule forgives this (gate): every change specs/gate.yaml declares
        moves a value, none removes one, so a value that vanished under a rule
        is a failure the rule was not written for.
        """
        return self.current is None or self.current == "-" or self.current.startswith("- ")

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
        return f"{self.where()}: {self.published} -> {now}"


# --------------------------------------------------------------------------- #
# Comparing two printed values
# --------------------------------------------------------------------------- #

_NUM = re.compile(r"[+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?")


def _places(token: str) -> int:
    return len(token.split(".")[1]) if "." in token else 0


def _sig(token: str) -> int:
    return len(token.lower().split("e")[0].lstrip("+-").replace(".", "").lstrip("0")) or 1


def _sig3(token: str) -> float:
    return float(f"{float(token):.3g}")


def _num_same(a: str, b: str) -> bool:
    """Two printed numbers agree to the coarser of their printed precisions,
    and to three significant digits.

    The second condition is for a value printed with ``:.3g`` (a Wilcoxon p),
    which drops trailing zeros: "1" is 1.00 and "0.5" is 0.500, not a number
    known only to the unit or the tenth. At the coarser precision alone, "1"
    would match anything in [0.5, 1.5) and "0.5" anything in [0.45, 0.55).
    Where both sides print the same number of places, the first condition
    already implies the second.
    """
    if _sig3(a) != _sig3(b):             # -0.0 == 0.0: a signed zero is no change
        return False
    if "e" in a.lower() or "e" in b.lower():
        sig = min(_sig(a), _sig(b))
        return f"{float(a):.{sig - 1}e}" == f"{float(b):.{sig - 1}e}"
    q = Decimal(1).scaleb(-min(_places(a), _places(b)))
    return (Decimal(a).quantize(q, rounding=ROUND_HALF_UP)
            == Decimal(b).quantize(q, rounding=ROUND_HALF_UP))


def same(published: str, current: str) -> bool:
    """Whether a value, as printed today, is the value the snapshot printed.

    The text around the numbers has to match exactly -- a tie mark, a sign
    of incompleteness, an sd that appeared or vanished is a change -- and the
    numbers have to agree at the precision the snapshot (or today, if it
    prints fewer digits) printed them with, and to three significant digits
    (_num_same).
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


@dataclass
class Held:
    """What one kind of check found: published values, changes, new values."""

    published: int = 0
    changes: list = field(default_factory=list)
    new: list = field(default_factory=list)
    notes: list = field(default_factory=list)
    by_section: dict = field(default_factory=dict)   # section -> [published, changed]

    def tally(self, section: str, changed: bool) -> None:
        self.published += 1
        n = self.by_section.setdefault(section, [0, 0])
        n[0] += 1
        n[1] += changed


def hold_cells(tables: Path, cells_csv: Path, archived_rules: list[dict]) -> Held:
    """Every published MSE/MAE cell of the per-setup sections, from cells.csv."""
    results = str(Path(__file__).resolve().parent.parent)
    if results not in sys.path:
        sys.path.insert(0, results)
    import plot_metrics as pm

    cells = load_cells(cells_csv)
    held = Held()
    absent = archived = 0

    def lookup(setup, model_disp, dataset, lookback, horizon, k):
        ds = pm.short_dataset(dataset)
        slug = DATASET_SLUG.get(dataset, DATASET_SLUG.get(ds, ds.lower()))
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
        if math.isnan(r.value):
            # A '-' in the table: that grid cell was never run. A frames row
            # for it now is a value the snapshot never had -- new, not a change.
            absent += 1
            got = row and (row["mse"] if r.metric == "MSE" else row["mae"])
            if got:
                held.new.append(Change("cells", key, "-", f"{float(got):.4f}"))
            continue
        if row is None and is_archived(archived_rules, setup, ds, r.lookback, r.horizon):
            # Published once, deliberately withdrawn since. Not held: there is
            # nothing left to hold it against.
            archived += 1
            continue
        got = "" if row is None else (row["mse"] if r.metric == "MSE" else row["mae"])
        published = f"{r.value:.4f}"
        changed = got == "" or round(float(got), 4) != round(r.value, 4)
        held.tally(r.section, changed)
        if changed:
            held.changes.append(Change("cells", key, published,
                                       None if got == "" else f"{float(got):.4f}"))
    if absent:
        held.notes.append(f"{absent} cells are '-' in the snapshot (never run); "
                          f"{len(held.new)} of them have a frames row now")
    if archived or archived_rules:
        held.notes.append(f"{archived} cells withdrawn by specs/runs.yaml "
                          f"'archived' rules -- not held")
    return held


def hold_rendered(kind: str, published_text: str, current_text: str, parse,
                  section_of) -> Held:
    """A kind whose values are read from the snapshot and today's render alike."""
    names = KINDS[kind]
    before, after = parse(published_text), parse(current_text)
    held = Held()
    for key, value in before.items():
        now = after.get(key)
        changed = now is None or not same(value, now)
        held.tally(section_of(key), changed)
        if changed:
            held.changes.append(Change(kind, dict(zip(names, key[1:])), value, now))
    for key in sorted(after.keys() - before.keys()):
        held.new.append(Change(kind, dict(zip(names, key[1:])), "", after[key]))
    return held


# --------------------------------------------------------------------------- #
# Expected changes, from specs/gate.yaml
# --------------------------------------------------------------------------- #

def load_expected(spec: Path | None, snapshot: str) -> tuple[list[dict], list[str]]:
    """The rules of specs/gate.yaml that apply to ``snapshot``, and notes.

    A rule written for another snapshot, a rule with no reason, and a rule
    naming a field its kind does not have are all left out, with a note:
    each can only make the gate stricter by being dropped, never looser.
    """
    if spec is None or not spec.exists():
        return [], []
    try:
        import yaml
        doc = yaml.safe_load(spec.read_text()) or {}
    except Exception as exc:
        return [], [f"WARNING: could not read {spec}: {exc}; no change is expected"]
    written_for = str(doc.get("snapshot") or "")
    rules = list(doc.get("expected") or [])
    if rules and written_for != snapshot:
        return [], [f"NOTE: {spec.name}'s {len(rules)} expected changes were written "
                    f"against {written_for or 'no snapshot'}, not {snapshot}; "
                    f"none applies"]
    kept, notes = [], []
    for i, rule in enumerate(rules, 1):
        if not isinstance(rule, dict) or rule.get("kind") not in KINDS:
            notes.append(f"WARNING: {spec.name} rule {i}: kind must be one of "
                         f"{', '.join(KINDS)}; ignoring it")
            continue
        if not str(rule.get("reason") or "").strip():
            notes.append(f"WARNING: {spec.name} rule {i}: no reason; ignoring it")
            continue
        unknown = set(rule) - set(KINDS[rule["kind"]]) - {"kind", "reason", "k_min"}
        if unknown:
            notes.append(f"WARNING: {spec.name} rule {i}: {rule['kind']} values have no "
                         f"field {', '.join(sorted(unknown))}; ignoring it")
            continue
        kept.append(dict(rule, _n=i))
    return kept, notes


def matches(rule: dict, change: Change) -> bool:
    if rule["kind"] != change.kind:
        return False
    for name in KINDS[change.kind]:
        if name not in rule:
            continue
        allowed = rule[name] if isinstance(rule[name], list) else [rule[name]]
        if str(change.key[name]).casefold() not in {str(a).casefold() for a in allowed}:
            return False
    if "k_min" in rule:
        k = str(change.key.get("k", ""))
        if not k.isdigit() or int(k) < int(rule["k_min"]):
            return False
    return True


def _headline(reason: str) -> str:
    text = " ".join(str(reason).split())
    first = re.split(r"(?<=\.)\s", text, maxsplit=1)[0]
    return first if len(first) <= 110 else first[:107] + "..."


# --------------------------------------------------------------------------- #
# The gate
# --------------------------------------------------------------------------- #

@dataclass
class Gate:
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

    def coverage(self, title: str) -> str:
        """What the gate held of one rendered section, for the text-drift note."""
        for kind, held in self.held.items():
            if title not in held.by_section:
                continue
            n, changed = held.by_section[title]
            where = f"({KIND_WHERE[kind]})"
            if not changed:
                return f"its {n} gated values {where} all reproduce"
            bad = sum(1 for c in self.unexpected if _section(c) == title)
            return (f"{changed} of its {n} gated values {where} changed: "
                    f"{changed - bad} expected, {bad} unexpected")
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
         runs_spec: Path | None = None, gate_spec: Path | None = None,
         verbose: bool = False, out=sys.stdout) -> Gate:
    """Hold every value ``published`` prints against today's frames and render.

    ``sections`` is today's report, as make.py tables renders it:
    (slug, title, markdown). Prints the account and returns it; the gate
    passes when no change is unexpected.
    """
    from .tables import SWEEP_TITLE

    specs = Path(__file__).resolve().parent.parent / "specs"
    runs_spec = specs / "runs.yaml" if runs_spec is None else runs_spec
    gate_spec = specs / "gate.yaml" if gate_spec is None else gate_spec

    text = published.read_text()
    current = "\n\n".join(md.strip() for _, _, md in sections)
    held = {
        "cells": hold_cells(published, cells_csv, load_archived(runs_spec)),
        "scoreboard": hold_rendered("scoreboard", text, current, scoreboard_values,
                                    lambda key: key[1]),
        "sweep": hold_rendered("sweep", text, current, sweep_values,
                               lambda key: SWEEP_TITLE),
    }
    rules, notes = load_expected(gate_spec, published.name)
    expected = [(rule, []) for rule in rules]
    unexpected = []
    for h in held.values():
        for change in h.changes:
            hit = None if change.vanished else next(
                (cs for rule, cs in expected if matches(rule, change)), None)
            (unexpected if hit is None else hit).append(change)
    result = Gate(published.name, held, expected, unexpected, notes)
    report(result, verbose=verbose, out=out)
    return result


def report(g: Gate, *, verbose: bool = False, out=sys.stdout) -> None:
    print(f"value gate: {g.snapshot} against today's frames", file=out)
    for kind, h in g.held.items():
        changed = len(h.changes)
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
        print(f"expected changes (specs/gate.yaml, snapshot {g.snapshot}): "
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
        print(f"FAIL: {len(g.unexpected)} published values changed and no rule in "
              f"specs/gate.yaml expects it", file=out)
    out.flush()           # before make.py's own messages on stderr
