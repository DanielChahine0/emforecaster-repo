"""The only place an artifact path is built.

Every figure used to be addressed by a hand-typed ``--outdir``, which is how
159 of the 230 files in figures/ came to exist with no command that
reproduces them, and how a bare ``multivariate_mse.png`` ended up meaning a
different (L, O) slice in three different directories.

The grammar, in one line:

    <family>/[<group>/]<subject?>_<axes...>_<metric>[_<view>].<ext>

with two separators doing two jobs -- ``_`` between axis tokens, ``-`` inside
one token -- and every numeric token self-labelling (``l336``, ``o96``,
``k5``), so a name parses without positional guessing.

Rule N6 is the one that matters most: an inapplicable axis is omitted, but a
*pooled* one is named. ``l48-480x48_o96`` says "averaged over the ladder";
never signal pooling by leaving the token out, or today's file quietly means
something else the day another window is added.
"""

from __future__ import annotations

import re
from pathlib import Path

# Families that fan out per setup, and get a directory level for it.
SETUP_FAMILIES = {"by-dataset", "by-model", "tradeoff", "trustworthiness"}

# Families whose natural second level is the dataset.
DATASET_FAMILIES = {"per-band", "emf-levels", "heatmaps", "loss-curves", "forecasts"}

# The section names the plotters use -> the setup slug in a path.
SETUP_SLUGS = {
    "Multivariate": "multivariate",
    "Univariate Series": "series",
    "Univariate (Per Channel)": "perchannel",
    "Cluster (k-Grouped Channels)": "cluster",
}

# Legacy stem -> setup slug, for the --section aliases plot_metrics still uses.
STEM_SLUGS = {"multivariate": "multivariate", "univariate": "series",
              "perchannel": "perchannel", "cluster": "cluster"}

_SLUG_RE = re.compile(r"[^a-z0-9]+")


def slug(text: str) -> str:
    """Lowercase, hyphen-joined, no spaces or parentheses. Rule N1."""
    return _SLUG_RE.sub("-", text.strip().lower()).strip("-")


def window_token(configs) -> str:
    """The (L, O) part of a name, naming any pooling rather than hiding it.

    One config          -> ``l336_o96``
    An even ladder      -> ``l48-480x48_o96``   (48 to 480, every 48)
    Anything else       -> ``lall_o96`` / ``lall_oall``
    """
    if not configs:
        return "lall_oall"
    lookbacks = sorted({int(l) for l, _ in configs})
    horizons = sorted({int(o) for _, o in configs})

    def part(prefix: str, values: list[int]) -> str:
        if len(values) == 1:
            return f"{prefix}{values[0]}"
        if len(values) == 2:
            # Two values are their own range; a synthetic step would read as a
            # ladder ("l192-336x144") when it is just the pair {192, 336}.
            return f"{prefix}{values[0]}-{values[1]}"
        steps = {b - a for a, b in zip(values, values[1:])}
        if len(steps) == 1:
            return f"{prefix}{values[0]}-{values[-1]}x{steps.pop()}"
        return f"{prefix}all"

    return f"{part('l', lookbacks)}_{part('o', horizons)}"


def artifact_path(
    root: Path,
    family: str,
    *,
    group: str | None = None,
    subject: str | None = None,
    configs=None,
    k=None,
    dataset: str | None = None,
    metric: str | None = None,
    view: str | None = None,
) -> tuple[Path, str]:
    """(directory, stem) for one artifact. The only path constructor.

    ``family`` is always the directory and never a filename token, which is
    what makes ``trustworthiness/multivariate_trustworthiness.png`` -- a name
    that says its family twice -- unrepresentable.
    """
    directory = root / family
    if group:
        directory = directory / slug(group)

    tokens: list[str] = []
    if subject:
        tokens.append(slug(subject))
    if dataset:
        tokens.append(slug(dataset))
    if configs is not None:
        tokens.append(window_token(configs))
    if k is not None:
        tokens.append(f"k{k}")
    if metric:
        tokens.append(slug(metric))
    if view:
        tokens.append(slug(view))
    return directory, "_".join(t for t in tokens if t)
