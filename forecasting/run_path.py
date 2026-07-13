"""Single source of truth for the on-disk logs/ layout.

A run is stored at::

    logs/<setup>/<model_id>/<dataset>/L<seq>_O<pred>_<protocol>[/ablation_<k>][/<tag>]/<seed>/

- <setup> is "univariate" or "multivariate" (fixed per forecasting subpackage).
- <protocol> is "cp" when conformal prediction is on, else "pf" (point forecast).
- The ablation_<k> level appears only for tuning/ablation trials (ablation_id != 1).
- The <tag> level appears only when exp.id is set to a non-empty free-form string.

build_run_dir() builds that path from config values. iter_run_dirs() walks an
existing tree and yields each run dir (the parent of the <seed>/ leaves that
hold log.json), regardless of how many grouping levels sit above it.
"""

import glob
import os

# Canonical short folder names. Anything not listed falls back to a lowercased
# copy of the raw dataset name, so a new dataset still gets a sane folder without
# a code change; add an entry here only to shorten it.
DATASET_SLUGS = {
    "ED2022": "ed2022",
    "UH2022": "uh2022",
    "EMF_2023_university_5_with_PTV": "emf2023",
    "rf_emf_det": "det",
    "rf_emf_ptv": "ptv",
    "rf_emf_ptv2": "ptv2",
}


def dataset_slug(dataset):
    """Short folder name for a dataset, e.g. 'ED2022' -> 'ed2022'."""
    return DATASET_SLUGS.get(dataset, str(dataset).strip().lower())


def build_run_dir(setup, model_id, dataset, seq_len, pred_len, conf_on,
                  ablation_id=1, tag="", logs_dir="logs"):
    """Return the run directory (the parent of the <seed>/ leaf) for one run."""
    protocol = "cp" if conf_on else "pf"
    parts = [logs_dir, setup, str(model_id), dataset_slug(dataset),
             f"L{seq_len}_O{pred_len}_{protocol}"]
    if int(ablation_id) != 1:
        parts.append(f"ablation_{int(ablation_id)}")
    if tag:
        parts.append(str(tag))
    return os.path.join(*parts)


def iter_run_dirs(logs_dir="logs"):
    """Every run dir under logs_dir, deduped and sorted oldest-to-newest by mtime."""
    run_dirs = {}
    for log_path in glob.glob(os.path.join(logs_dir, "**", "log.json"), recursive=True):
        run = os.path.dirname(os.path.dirname(log_path))
        run_dirs[run] = os.path.getmtime(log_path)
    return sorted(run_dirs, key=run_dirs.get)


def relative_run_path(path, logs_dir="logs"):
    """Path of `path` relative to logs_dir, for mirroring into old_logs/."""
    return os.path.relpath(path, logs_dir)
