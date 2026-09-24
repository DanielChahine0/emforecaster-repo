"""Provenance stamped into every run's log.json under parameters/exp/*.

Why this exists: on 2026-09-19 the three paradigm trees under logs/ turned out
to have been produced by three different commits (multivariate and per-channel
on 2026-08-01/03, cluster on 2026-08-25) and no run recorded which, so "same
network, different paradigm" could not be asserted from disk. A run now says
which code, which machine, how many trainings shared the GPU while it ran, and
whether they shared it through CUDA MPS. See .personal/PLAN.md, contract C1
(commit) and C7 (concurrency).

Everything here is best-effort: a missing git binary, a tarball checkout or a
CPU-only box must never abort a training run, so failures degrade to "unknown".
"""

import os
import socket
import subprocess

import torch

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Paths whose uncommitted edits change what a training run computes. logs/,
# results/ and data/ are deliberately absent: they change all the time and say
# nothing about the code that produced a run.
CODE_PATHS = ("forecasting", "main.py", "pyproject.toml", "TimeSeriesCCM-main")

# Where a CUDA client looks for the MPS control daemon when
# CUDA_MPS_PIPE_DIRECTORY is unset (nvidia-cuda-mps-control(1)).
MPS_DEFAULT_PIPE_DIRECTORY = "/tmp/nvidia-mps"


def _git(*argv):
    # GIT_OPTIONAL_LOCKS=0 keeps `git status` read-only. Without it status
    # opportunistically rewrites the index under .git/index.lock, and when the
    # timeout below kills it mid-rewrite -- which happened on 2026-09-21, with a
    # dozen trainings starting at once under heavy disk load -- the lock is left
    # behind and every later commit in the repo fails until it is removed by hand.
    env = {**os.environ, "GIT_OPTIONAL_LOCKS": "0"}
    try:
        out = subprocess.run(["git", "-C", _REPO_ROOT, *argv], capture_output=True,
                             text=True, check=True, timeout=15, env=env)
        return out.stdout.strip()
    except Exception:
        return None


def git_commit():
    """Full SHA of the checked-out HEAD, or "unknown" outside a git checkout."""
    return _git("rev-parse", "HEAD") or "unknown"


def git_dirty():
    """True if any CODE_PATHS file has uncommitted changes.

    False when git is unavailable: git_commit() already reads "unknown" in that
    case, which is the stronger signal, and a bool keeps the logged field one
    type across runs.
    """
    out = _git("status", "--porcelain", "--", *CODE_PATHS)
    return bool(out.strip()) if out is not None else False


def gpu_name():
    try:
        if torch.cuda.is_available():
            return torch.cuda.get_device_name(0)
        if torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"


def mps_pipe_directory():
    """The directory a CUDA client in this process looks for MPS in.

    Stamped beside cuda_mps, whether or not CUDA is usable, so a run can be
    matched to the daemon its launcher pointed it at: a private directory per
    sweep, or the machine-wide default.
    """
    return os.environ.get("CUDA_MPS_PIPE_DIRECTORY") or MPS_DEFAULT_PIPE_DIRECTORY


def cuda_mps():
    """True if this process's CUDA work runs through an MPS server.

    Why this exists: the per-channel sweep of 2026-09-21 ran part of its bands
    under CUDA MPS, and only .personal/PLAN.md recorded which (finding E10).
    MPS changes how the processes' kernels share the GPU, not what they
    compute, but it changes what a contended timing means.

    A CUDA client attaches to MPS when a control daemon listens in its pipe
    directory -- CUDA_MPS_PIPE_DIRECTORY, else /tmp/nvidia-mps -- and runs on
    its own context otherwise; the daemon's socket there is `control`. The
    grid drivers hand their environment to every run they start, so a run
    sees the pipe directory the launcher set. A daemon killed rather than told
    to quit can leave the socket behind, and a run then reads True without
    having attached. It describes the process, not the model: an ARIMA run,
    which fits on the CPU, reads True under a daemon too.
    """
    try:
        if not torch.cuda.is_available():
            return False
        return os.path.exists(os.path.join(mps_pipe_directory(), "control"))
    except Exception:
        return False


def run_provenance(args=None):
    """The fields Experiment.init_logger writes under parameters/exp/.

    `concurrency` is read from args.exp.concurrency, which the grid drivers set
    to their --jobs value; a single-job run leaves it at the config default 1.

    `cuda_mps` and `cuda_mps_pipe_dir` are not the config's exp.mps, which
    selects Apple silicon's Metal backend and is logged as parameters/exp/mps
    with the rest of the config; these fields are written after it, so a
    shared name would overwrite it.
    """
    exp = getattr(args, "exp", None)
    concurrency = getattr(exp, "concurrency", 1) if exp is not None else 1
    return {
        "git_commit": git_commit(),
        "git_dirty": git_dirty(),
        "hostname": socket.gethostname(),
        "gpu_name": gpu_name(),
        "torch_version": torch.__version__,
        "concurrency": int(concurrency),
        "cuda_mps": cuda_mps(),
        "cuda_mps_pipe_dir": mps_pipe_directory(),
    }
