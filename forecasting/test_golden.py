"""Golden hashes: the existing models' numerics, pinned bit for bit.

The tripwire for changes that must not move an existing model. Extending CCM to
more models touches get_model, the model constructors and the cluster wrapper,
and every run already under logs/ has to stay reproducible by the code that
replaces them: a byte of drift in an init or a forward and the old and new runs
are no longer the same network.

Each case builds one network through the grid's own path (the rf_emf_mv/* job
config, the grid's window and overrides, get_model) from a fixed seed and
records three sha256 digests:

  state    the state_dict, names and bytes: the init.
  rng      the CPU RNG state right after the build: what the constructor drew,
           which shifts every later draw (dropout masks, the CCM assigner).
  forward  an eval-mode forward on a fixed input; for CCM through
           CCMForecaster, so the assigner's membership is in it.

The cases are the six gradient-trained multivariate models and the CCM builds
of DLinear/PatchTST/TSMixer at k in {1, 3, 9}, each at the EMF benchmark shape
(27 bands, L336/O96) and the VoIP one (6 channels, L48/O12, VOIP_OVERRIDES).

A failure means a change altered an existing model's numerics: fix the change,
not the fixture. The fixture is only regenerated for a deliberate numerics
change or a new torch, from a clean tree:

    venv/bin/python -m forecasting.test_golden --regenerate

The digests are only bit-exact for the torch build and CPU kernel set that
recorded them (AVX2 and AVX512 kernels round differently), so the whole module
skips when either differs from the fixture's.
"""

import argparse
import hashlib
import json
import os
import random
import subprocess
import sys

import numpy as np
import pytest
import torch

from forecasting import grid, grids, provenance
from forecasting.multivariate.utils.models import get_model

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIXTURE = os.path.join(_REPO_ROOT, "forecasting", "test_golden.json")

SEED = grids.SEED
INPUT_SEED = 0
# More than one window, so the CCM assigner's batch-mean membership is pinned
# as the pipeline computes it.
BATCH = 4
# CPU kernels split reductions by thread count (12 of the 30 forwards change
# at 24 threads), so one thread keeps the bytes independent of the machine.
NUM_THREADS = 1

# (name, num_channels, seq_len, pred_len, dataset, grid overrides)
SHAPES = [
    ("emf", 27, 336, 96, "ED2022", None),
    ("voip", 6, 48, 12, "VOIP_MOBILE", grids.VOIP_OVERRIDES),
]
MV_IDS = ["DLinear", "PatchTST", "TSMixer", "RF_EMF_Transformer", "EMForecaster",
          "LSTM"]
CCM_IDS = ["DLinear", "PatchTST", "TSMixer"]
K_VALUES = [1, 3, 9]

_JOBS = {model_id: job for _, model_id, job in grids.MV_MODELS}


def _cases():
    cases = []
    for shape in SHAPES:
        cases += [(shape, model_id, None) for model_id in MV_IDS]
        cases += [(shape, model_id, k) for model_id in CCM_IDS for k in K_VALUES]
    return cases


def case_id(shape, model_id, k):
    return f"{shape[0]}-{model_id}" + (f"-k{k}" if k is not None else "")


CASES = {case_id(*case): case for case in _cases()}


def build_args(shape, model_id, k):
    """The config run_grid would hand Experiment for this cell.

    hardware.normalize_config is left out: it only sets batch, worker and
    compile fields, none of which reach a constructor, and applying it would
    tie the fixture to the host. num_channels is set by hand because
    get_loaders infers it from the data, which the build does not load.
    """
    _, num_channels, seq_len, pred_len, dataset, overrides = shape
    load_config, _ = grid.setup_modules("multivariate")
    args = load_config(grid.args_path("multivariate", _JOBS[model_id]))
    args.data.dataset = dataset
    args.data.seq_len = seq_len
    args.data.pred_len = pred_len
    args.data.num_channels = num_channels
    args.exp.seed_list = [SEED]
    args.exp.seed = SEED
    if k is not None:
        args.cluster.enabled = True
        args.cluster.k_clusters = k
    grid.apply_overrides(args, overrides)
    return args


def _update(h, name, tensor):
    t = tensor.detach().cpu().contiguous()
    h.update(f"{name}|{t.dtype}|{tuple(t.shape)}\n".encode())
    h.update(t.reshape(-1).view(torch.uint8).numpy().tobytes())


def state_hash(model):
    h = hashlib.sha256()
    for name, tensor in model.state_dict().items():
        _update(h, name, tensor)
    return h.hexdigest()


def tensor_hash(tensor):
    h = hashlib.sha256()
    _update(h, "", tensor)
    return h.hexdigest()


def compute(case):
    """The recorded digests for one case.

    Global RNG state and the thread count are restored afterwards, so the
    tests around this one see the process as they left it.
    """
    shape, model_id, k = case
    _, num_channels, seq_len, _, _, _ = shape
    args = build_args(*case)

    saved = torch.get_rng_state(), np.random.get_state(), random.getstate()
    threads = torch.get_num_threads()
    torch.set_num_threads(NUM_THREADS)
    try:
        # Seeded the way Experiment.run seeds before it builds the model.
        torch.manual_seed(SEED)
        np.random.seed(SEED)
        random.seed(SEED)
        model = get_model(args, torch.Generator().manual_seed(SEED))
        rng = hashlib.sha256(torch.get_rng_state().numpy().tobytes()).hexdigest()
        state = state_hash(model)

        x = torch.randn(BATCH, num_channels, seq_len,
                        generator=torch.Generator().manual_seed(INPUT_SEED))
        model.eval()
        with torch.no_grad():
            out = model(x)
    finally:
        torch.set_num_threads(threads)
        torch.set_rng_state(saved[0])
        np.random.set_state(saved[1])
        random.setstate(saved[2])

    return {
        "state": state,
        "rng": rng,
        "forward": tensor_hash(out),
        "num_params": sum(p.numel() for p in model.parameters()),
        "out_shape": list(out.shape),
    }


def _load_fixture():
    with open(FIXTURE) as f:
        return json.load(f)


GOLDEN = _load_fixture() if os.path.exists(FIXTURE) else None


def _skip_reason():
    if GOLDEN is None:
        return None
    if torch.__version__ != GOLDEN["torch"]:
        return (f"golden hashes were recorded under torch {GOLDEN['torch']}; this is "
                f"torch {torch.__version__}, whose kernels need not be bit-identical")
    capability = torch.backends.cpu.get_cpu_capability()
    if capability != GOLDEN["cpu_capability"]:
        return (f"golden hashes were recorded with {GOLDEN['cpu_capability']} CPU "
                f"kernels; this machine runs {capability}, which rounds differently")
    return None


_SKIP = _skip_reason()
pytestmark = pytest.mark.skipif(_SKIP is not None, reason=_SKIP or "")


# ---------------------------------------------------------------------------
# The check
# ---------------------------------------------------------------------------


def test_fixture_exists():
    assert GOLDEN is not None, f"{FIXTURE} is missing; see the module docstring"


def test_fixture_covers_exactly_the_cases():
    # A case added here without a regeneration would otherwise never be checked,
    # and one dropped would leave a stale entry that reads as coverage.
    assert GOLDEN is not None
    assert list(GOLDEN["cases"]) == list(CASES)


def test_fixture_records_how_it_was_computed():
    # Changing any of these changes every digest; they are recorded so a
    # mismatch can be told apart from a settings change.
    assert GOLDEN is not None
    assert (GOLDEN["seed"], GOLDEN["input_seed"], GOLDEN["batch"],
            GOLDEN["num_threads"]) == (SEED, INPUT_SEED, BATCH, NUM_THREADS)


@pytest.mark.parametrize("name", list(CASES))
def test_golden_hashes(name):
    assert GOLDEN is not None and name in GOLDEN["cases"]
    want = GOLDEN["cases"][name]
    got = compute(CASES[name])
    # Structure first: a changed parameter count names the problem outright.
    assert got["num_params"] == want["num_params"], f"{name}: parameter count changed"
    assert got["out_shape"] == want["out_shape"], f"{name}: output shape changed"
    assert got["state"] == want["state"], f"{name}: initial weights changed"
    assert got["rng"] == want["rng"], f"{name}: the build draws from the RNG differently"
    assert got["forward"] == want["forward"], f"{name}: eval forward changed"


# ---------------------------------------------------------------------------
# Regeneration
# ---------------------------------------------------------------------------


def _dirty_code_paths():
    """Changed or untracked files that a training run could import.

    Tests, their fixtures and markdown are left out, the same idea as the code
    fingerprint in results/emfres/frames.py: no run reads them. It also lets
    this module and its fixture be written before their first commit.
    """
    out = subprocess.run(
        ["git", "-C", _REPO_ROOT, "status", "--porcelain",
         "--untracked-files=all", "--", *provenance.CODE_PATHS],
        capture_output=True, text=True, check=True).stdout
    dirty = []
    for line in out.splitlines():
        path = line[3:].split(" -> ")[-1]
        name = path.rsplit("/", 1)[-1]
        if not (name.endswith(".md") or name.startswith("test_")):
            dirty.append(path)
    return dirty


def regenerate():
    commit = provenance.git_commit()
    if commit == "unknown":
        sys.exit("not a git checkout; the fixture has to name the commit it pins")
    dirty = _dirty_code_paths()
    if dirty:
        sys.exit("refusing to regenerate on a dirty tree; the fixture must pin "
                 "committed code:\n  " + "\n  ".join(dirty))
    golden = {
        "torch": torch.__version__,
        "cpu_capability": torch.backends.cpu.get_cpu_capability(),
        "commit": commit,
        "seed": SEED,
        "input_seed": INPUT_SEED,
        "batch": BATCH,
        "num_threads": NUM_THREADS,
        "cases": {name: compute(case) for name, case in CASES.items()},
    }
    changed = [name for name, digests in golden["cases"].items()
               if GOLDEN is None or GOLDEN["cases"].get(name) != digests]
    with open(FIXTURE, "w") as f:
        json.dump(golden, f, indent=2)
        f.write("\n")
    print(f"wrote {len(golden['cases'])} cases to {FIXTURE} "
          f"({len(changed)} new or changed) at {commit[:12]}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--regenerate", action="store_true",
                        help="rewrite the fixture from the current (clean) tree")
    if not parser.parse_args().regenerate:
        parser.error("nothing to do: pytest runs the check; pass --regenerate "
                     "to rewrite the fixture")
    regenerate()
