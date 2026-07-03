# forecasting

Deep-learning time-series forecasting package used to reproduce the EMForecaster paper.
It ships two coexisting setups under one package: a **univariate** setup and a **multivariate** extension.
See the repo-root [README](../README.md) for project-level context, and the per-setup guides linked below for the full detail.

## The two setups

- [`univariate/`](./univariate/README.md) - single-channel EMF forecasting (the paper's main setup).
- [`multivariate/`](./multivariate/README.md) - the multivariate extension over the same subtree.

The multivariate setup is a near-identical twin of the univariate one.
Its config differs by exactly one added field, `exp.eval_test_per_epoch` (per-epoch test-loss logging; model selection still uses the validation loss only).
The genuine multivariate logic lives in five files: `utils/dataloading.py`, `exp/exp.py`, `utils/models.py`, `models/emforecaster.py`, `config/config.py`.
Everything else is byte-identical apart from the package import prefix.

## Setup selection (root `main.py`)

The root `main.py` auto-selects the setup by job-name prefix:

```python
package = "forecasting.multivariate" if job_name.startswith("rf_emf_mv") else "forecasting.univariate"
```

It then dynamically imports `{package}.config.config.load_config` and `{package}.exp.exp.Experiment`, and reads `forecasting/<setup>/jobs/exp/<job_name>/args.yaml`.
Job names starting with `rf_emf_mv` dispatch to the multivariate setup; every other job name dispatches to univariate.
Flags: `--quick` (single seed, `seed_list[:1]`), `--ablation '<json>'` (dotted `group.param` / `global.group.param` overrides), `--ablation_id`.

## Shared subtree layout

Each setup mirrors the same subtree:

| Dir | Purpose |
| --- | --- |
| `config/` | Pydantic config (`config.py`, `load_config` -> `Global`). |
| `exp/` | `Experiment` orchestration (seed, device, train, test, logging). |
| `models/` | Model definitions. |
| `layers/` | Reusable model building blocks. |
| `conformal/` | Conformal prediction: `coverage.py` (calibrate/infer) + `tradeoff.py` (Trade-off Score analysis). |
| `tuning/` | Grid-search entrypoint `tune.py`. |
| `jobs/` | Experiment configs (`jobs/exp/`) and HPC submission (`jobs/hpc/submit.py`). |
| `utils/` | Data loading, model factory, datasets, helpers. |
| `analysis/` | Standalone plotting/stats and Neptune result aggregation. |

## Shared components

- **Config**: pydantic sub-models loaded from each job's `args.yaml`; job overrides realize the actual behavior, not the class defaults.
- **Conformal prediction**: gated by `conf.conf`; `coverage.py` and `tradeoff.py` are byte-identical between the two setups.
- **Tuning**: per-setup `tuning/tune.py <job_name> [--cc] [--acc <account>]` reads a job's `ablation.yaml` and builds an `itertools.product` grid.
- **HPC**: `--cc` submits SLURM / Compute Canada jobs via `jobs/hpc/submit.py` (submitit).
- **Neptune** (optional): enabled by `exp.neptune` + `NEPTUNE_API_TOKEN`; otherwise runs log offline to `log.json`.

## Run examples (from the repository root)

```bash
# Univariate smoke test (EMForecaster, 5 epochs)
python main.py test --quick

# Univariate EMForecaster on the PTV 2023 dataset
python main.py rf_emf/emforecaster

# Multivariate EMForecaster (dispatches to forecasting.multivariate)
python main.py rf_emf_mv/emforecaster

# Multivariate smoke test (EMForecaster, 5 epochs, ED2022, 27 channels)
python main.py rf_emf_mv/test --quick
```

## Outputs

Offline runs land in `logs/<ablation_id>_<model_id>_<exp.id>/<seed>/`, containing `args.yaml` (resolved config), `log.json` (offline metric log), `supervised.pth` (best-by-validation weights), sampled test plots `plot_<idx>.png`, and a conformal `scores` file when calibration scores are persisted.

## More detail

- Univariate models, datasets, jobs, and run flow: [`univariate/README.md`](./univariate/README.md).
- Multivariate extensions (channel independence, Italy datasets, per-channel MSE): [`multivariate/README.md`](./multivariate/README.md).
