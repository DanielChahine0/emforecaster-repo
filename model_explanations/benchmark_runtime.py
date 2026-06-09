"""
benchmark_runtime.py
--------------------
Measures, for every model actually used in the rf_emf experiments, the real
runtime on THIS machine: parameter count, forward latency, forward+backward
latency, and a per-component breakdown (via leaf-module timing hooks). It also
reads the exact train/val/test batch counts from the real data pipeline so the
full-run estimates are grounded, not guessed.

Run from the repo root:   python model_explanations/benchmark_runtime.py
Writes:                    model_explanations/_runtime_results.json
"""

import os
import json
import time
import warnings
from collections import defaultdict

warnings.filterwarnings("ignore")

import torch
import torch.nn as nn

from emforecaster.config.config import load_config
from emforecaster.utils.models import get_model
from emforecaster.utils.dataloading import get_loaders

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CFG = os.path.join(REPO, "emforecaster", "jobs", "exp", "rf_emf")

# (label, config-dir) for the 8 models that have experiment configs
MODELS = [
    ("RF_EMF_MLP",          "baselines/mlp"),
    ("RF_EMF_CNN",          "baselines/cnn"),
    ("RF_EMF_LSTM",         "baselines/lstm"),
    ("RF_EMF_Transformer",  "baselines/transformer"),
    ("Linear (DLinear)",    "dlinear"),
    ("TSMixer",             "tsmixer"),
    ("PatchTST",            "patchtst"),
    ("EMForecaster",        "emforecaster"),
]

TIME_BUDGET_S = 3.0   # aim ~3s of timed iters per measurement
MIN_ITERS, MAX_ITERS = 5, 60
WARMUP = 5


def pick_iters(one_iter_s):
    if one_iter_s <= 0:
        return MAX_ITERS
    return max(MIN_ITERS, min(MAX_ITERS, int(TIME_BUDGET_S / one_iter_s)))


def time_loop(fn, iters):
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    return (time.perf_counter() - t0) / iters


def leaf_modules(model):
    return [
        (name, m)
        for name, m in model.named_modules()
        if name != "" and len(list(m.children())) == 0
    ]


def component_breakdown(model, x, iters):
    """Time each leaf module via hooks, then group by top-level child name."""
    leaf_t = defaultdict(float)
    starts = {}
    handles = []

    def make_pre(name):
        def pre(mod, inp):
            starts[name] = time.perf_counter()
        return pre

    def make_post(name):
        def post(mod, inp, out):
            leaf_t[name] += time.perf_counter() - starts.get(name, time.perf_counter())
        return post

    for name, m in leaf_modules(model):
        handles.append(m.register_forward_pre_hook(make_pre(name)))
        handles.append(m.register_forward_hook(make_post(name)))

    model.eval()
    with torch.no_grad():
        for _ in range(iters):
            model(x)
    for h in handles:
        h.remove()

    comp = defaultdict(float)
    for name, t in leaf_t.items():
        comp[name.split(".")[0]] += t
    return dict(comp)


def measure_model(label, cfg_dir):
    args = load_config(os.path.join(CFG, cfg_dir, "args.yaml"))
    # Make data loading cheap + synchronous for a single batch grab.
    args.data.num_workers = 0
    args.data.pin_memory = False

    # Exact batch counts from the real pipeline.
    train_loader, val_loader, test_loader = get_loaders(
        args, flag="sl", loader_type="all"
    )
    n_train, n_val, n_test = len(train_loader), len(val_loader), len(test_loader)
    batch = next(iter(train_loader))
    x = batch[0].float()
    in_shape = tuple(x.shape)

    model = get_model(args)
    n_params = sum(p.numel() for p in model.parameters())

    # ---- forward (eval, no grad) ----
    model.eval()
    with torch.no_grad():
        for _ in range(WARMUP):
            model(x)
        one = time_loop(lambda: model(x), 1)
    k_fwd = pick_iters(one)
    with torch.no_grad():
        fwd = time_loop(lambda: model(x), k_fwd)

    # ---- forward + backward (train) ----
    model.train()

    def step():
        out = model(x)
        loss = out.float().pow(2).mean()
        loss.backward()
        model.zero_grad(set_to_none=True)

    for _ in range(WARMUP):
        step()
    one_b = time_loop(step, 1)
    k_step = pick_iters(one_b)
    step_t = time_loop(step, k_step)

    # ---- component breakdown (forward) ----
    comp = component_breakdown(model, x, max(5, k_fwd))
    comp_sum = sum(comp.values()) or 1e-9
    # attribute the clean forward total across components by measured share
    comp_ms = {k: 1000 * fwd * (v / comp_sum) for k, v in comp.items()}
    other_frac = max(0.0, 1.0 - comp_sum / max(comp_sum, comp_sum))  # 0; glue shown via residual below

    epochs = args.sl.epochs
    seeds = len(args.exp.seed_list)

    # per-epoch: train = (fwd+bwd) * train_batches ; val = fwd * val_batches
    t_epoch = step_t * n_train + fwd * n_val
    t_seed_100 = t_epoch * epochs + fwd * n_test
    t_total_100 = t_seed_100 * seeds

    return {
        "label": label,
        "params": n_params,
        "in_shape": in_shape,
        "batch_size": args.sl.batch_size,
        "epochs": epochs,
        "seeds": seeds,
        "batches": {"train": n_train, "val": n_val, "test": n_test},
        "fwd_ms": 1000 * fwd,
        "step_ms": 1000 * step_t,
        "components_ms": comp_ms,
        "t_epoch_s": t_epoch,
        "t_seed_100ep_s": t_seed_100,
        "t_total_5seed_100ep_s": t_total_100,
    }


def main():
    torch.manual_seed(0)
    print(f"PyTorch {torch.__version__} | device=CPU | threads={torch.get_num_threads()}")
    results = []
    for label, cfg_dir in MODELS:
        try:
            r = measure_model(label, cfg_dir)
            results.append(r)
            comp = ", ".join(
                f"{k}:{v:.1f}ms" for k, v in sorted(
                    r["components_ms"].items(), key=lambda kv: -kv[1]
                )
            )
            print(
                f"\n{label:20s} params={r['params']:>9,} in={r['in_shape']}\n"
                f"  fwd={r['fwd_ms']:.2f}ms  fwd+bwd={r['step_ms']:.2f}ms  "
                f"batches(tr/va/te)={r['batches']['train']}/{r['batches']['val']}/{r['batches']['test']}\n"
                f"  components: {comp}\n"
                f"  ~per-epoch={r['t_epoch_s']:.2f}s  ~per-seed(100ep)={r['t_seed_100ep_s']/60:.1f}min  "
                f"~full(5 seeds)={r['t_total_5seed_100ep_s']/60:.1f}min"
            )
        except Exception as e:
            print(f"\n{label:20s} FAILED: {type(e).__name__}: {e}")
            results.append({"label": label, "error": f"{type(e).__name__}: {e}"})

    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_runtime_results.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
