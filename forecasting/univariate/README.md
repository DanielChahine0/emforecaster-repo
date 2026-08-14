# forecasting/univariate

Detailed reference for the univariate forecasting setup.
For the concise hub that ties the two setups together, see [../README.md](../README.md).
For the multivariate twin, see [../multivariate/README.md](../multivariate/README.md).

## What "univariate" means here

A univariate run forecasts a single EMF exposure series: one channel in, one channel out.
The setup is realized entirely by the job `args.yaml` overrides, not by the config defaults (whose defaults are generic/multivariate).
Every univariate job sets:

- `data.univariate: True`
- `data.num_channels: 1`
- `sl.dataset_class: univariate_forecasting`
- `exp.task: forecasting` (the four `baselines/*` jobs omit `exp.task`, which is harmless since the `RF_EMF_*` models ignore it)

Channels are flattened into the window dimension inside `load_forecasting`, so each training example is a `(seq_len,)` window mapped to a `(pred_len,)` target.

## Dispatch (from the repo root)

The root [`main.py`](../../main.py) selects the setup by job name:
`package = "forecasting.multivariate" if job_name.startswith("rf_emf_mv") else "forecasting.univariate"`.
Every job whose name does not start with `rf_emf_mv` dispatches here.
It dynamically imports `forecasting.univariate.config.config.load_config` and `forecasting.univariate.exp.exp.Experiment`, reads `forecasting/univariate/jobs/exp/<job_name>/args.yaml`, then loops over `exp.seed_list` (setting `exp.seed` and `exp.time` per run) and calls `exp.run()`.

### Running

```
python main.py <job_name>                                          # all seeds in exp.seed_list
python main.py <job_name> --quick                                  # single seed (seed_list[:1])
python main.py <job_name> --ablation '<json>'                      # dotted overrides, e.g. {"sl.lr": 0.001}
python main.py <job_name> --ablation '<json>' --ablation_id <int>  # same, tagging the log dir with ablation_id
```

`--ablation` keys are applied via `update_global_config` and accept 2-part `group.param` or 3-part `global.group.param` forms.
`--ablation_id <int>` only takes effect together with `--ablation` (the tuner always passes both).
On its own it is ignored, so `exp.ablation_id` keeps its default of `1` and the log-dir prefix is unchanged.

## Models (`utils/models.py`, `get_model`)

Exactly 13 `model_id` branches; any other id raises `ValueError`:
`PatchTST`, `RecurrentModel`, `Linear`, `DLinear`, `EMForecaster`, `ModernTCN`, `TimesNet`, `KNN`, `TSMixer`, `RF_EMF_MLP`, `RF_EMF_CNN`, `RF_EMF_LSTM`, `RF_EMF_Transformer`.

- `RecurrentModel` picks LSTM/RNN/GRU/Mamba via `exp.backbone_id` (Mamba is not its own `model_id`).
- `KNN` is a tslearn classifier (optionally wrapped in `GridSearchCV`), not part of the deep forecasting `forward_pass`/`compute_loss` path.

Model files under `models/`: `dlinear.py`, `emforecaster.py`, `linear.py`, `mamba.py`, `modern_tcn.py`, `patchtst.py`, `recurrent.py`, `rf_emf_baselines.py` (defines `MLP, CNN, LSTM, Transformer`, imported as the `RF_EMF_*` ids), `timesnet.py`, `tsmixer.py`.

## Runnable jobs (`jobs/exp/`)

A job is runnable when its folder contains `args.yaml`.
There are 9:

| job_name | model | dataset | notes |
| --- | --- | --- | --- |
| `rf_emf/emforecaster` | EMForecaster (patch_model_id TSMixer, backbone Linear) | `rf_emf_ptv2` | `conf.conf: True` |
| `rf_emf/dlinear` | DLinear | `rf_emf_ptv2` | |
| `rf_emf/patchtst` | PatchTST | `rf_emf_ptv2` | |
| `rf_emf/tsmixer` | TSMixer | `rf_emf_ptv2` | |
| `rf_emf/baselines/cnn` | RF_EMF_CNN | `rf_emf_ptv2` | `conf.conf: True`, `neptune: True` |
| `rf_emf/baselines/lstm` | RF_EMF_LSTM | `rf_emf_ptv2` | |
| `rf_emf/baselines/mlp` | RF_EMF_MLP | `rf_emf_ptv2` | `conf.conf: True`, `neptune: True` |
| `rf_emf/baselines/transformer` | RF_EMF_Transformer | `rf_emf_ptv2` | |
| `test` | EMForecaster smoke test (5 epochs, seq_len 20 / pred_len 30) | `rf_emf_det` | |

All 8 `rf_emf/*` jobs also carry an `ablation.yaml` (so they can be tuned); `test` does not.
The `rf_emf/timesnet` and `rf_emf/compare` folders exist but have no `args.yaml`, so they are not runnable.

## Datasets

On disk in `data/univariate/`:

| id | CSV | series |
| --- | --- | --- |
| `rf_emf_det` | `DET_Monitoring.csv` | DET 2022 |
| `rf_emf_ptv` | `PTV_Monitoring.csv` | PTV 2022 |
| `rf_emf_ptv2` | `PTV_Monitoring2.csv` | PTV 2023 |

The loader reads column index 1 (the EMF value) and reshapes to `(N, 1)`.

Honest note: the `get_loaders` whitelist in `utils/dataloading.py` also names `ETTh1/ETTh2/ETTm1/ETTm2`, `electricity`, `traffic`, `weather`, `illness`, `rf_emf` (`LongTerm17.mat`), and `rf_emf_tur` (`Tur_Monitoring.csv`), but those files are not present, so only `rf_emf_det/ptv/ptv2` are reachable.
`load_forecasting` handles a few more ids (`rf_emf_det2`, `rf_emf_tur2`, `rf_emf_nov`), but they are absent from the whitelist and therefore unreachable through the normal pipeline.

### Dataset classes (`utils/datasets.py`)

Four are defined; two are wired into `get_loader` via `sl.dataset_class`:

- `univariate_forecasting` -> `UnivariateForecastingDataset(x, y)` over pre-windowed `(num_windows, seq_len)` / `(num_windows, pred_len)` tensors.
  This is what every univariate job uses.
- `forecasting` -> `ForecastingDataset(data, seq_len, pred_len)` slicing windows on the fly, `__getitem__` returning `(C, seq_len)`, `(C, pred_len)`.

## Config (`config/config.py`)

`load_config(path)` reads YAML into a pydantic `Global` with 15 sub-models:
`exp`, `data`, `sl`, `ddp`, `early_stopping`, `ema`, `scheduler`, `open_neuro`, `mlp_mixer`, `patchtst`, `dlinear`, `tsmixer`, `timesnet`, `moderntcn`, `conf`.
Defaults are generic (`data.univariate=False`, `data.num_channels=321`, `sl.dataset_class="forecasting"`, `exp.task="classification"`, `sl.criterion="MSE"`, `sl.scheduler=None`); the univariate behavior comes from the per-job overrides described above.
Unlike multivariate, this config has no `exp.eval_test_per_epoch` field (that field is the sole multivariate config addition).
Per-run configs live at `jobs/exp/<job_name>/args.yaml`, and the fully resolved config is also written to each run's log dir.

## Experiment.run() flow (`exp/exp.py`)

1. Seed everything (torch/cuda/numpy/random from `exp.seed`, build `self.generator`).
2. `init_logger()` - create `logs/<ablation_id>_<model_id>_<id>/<seed>/`, write `args.yaml`, set up Neptune or an in-memory dict plus `log.json`; if `conf.conf`, set `self.score_path`.
3. `init_device()` - `exp.mps` takes precedence (MPS), else CUDA `cuda:{gpu_id}` if available, else CPU.
4. `supervised_train()` - build train+val loaders (`data.seq_load=True` is required), `get_model`, `get_optim` (adam/adamw, weight decay excluded from bias/norm/RevIN), `get_criterion` (default MSE), `get_scheduler` if `sl.scheduler != "None"` (jobs use `onecycle`), then `train()` and `test()`.
5. Flush the logger (`logger.stop()` for Neptune, else `json.dump` to `log.json`).

`train()` validates each epoch, saving the best model by `exp.best_model_metric` (default `loss`) to `supervised.pth`, with optional `EarlyStopping`.
`test()` rebuilds the `test` (or `all` if `exp.calibrate`) loader, loads `supervised.pth`, and calls `evaluate(coverage=conf.conf, plot_results=True, num_plots=10)`.

## Conformal prediction (`conformal/`)

Toggled by `conf.conf`; key params are `conf.alpha` (default 0.1), `conf.corrected` (Bonferroni), `conf.intervals`, `conf.validation_eval`.
Both `coverage.py` and `tradeoff.py` are byte-identical to the multivariate copies.

- `coverage.py`: `get_all_critical_scores` (calibrate on the val loader) and `get_coverage` (inference producing independent coverage `ic`, joint coverage `jc`, and, when `conf.intervals`, `interval_width_mean` / `interval_width_std`).
- Calibration runs every epoch when `conf.validation_eval`, else only on the last epoch or at early stop, filling `self.calibration_scores = {"uncorrected", "corrected"}`.
- `tradeoff.py`: `compute_and_save_tos` computes an offline efficiency/coverage Trade-off Score; it is not called from the training loop.
  Run it separately as an analysis step on completed runs.

Note: the on-disk `scores` file is written only when `conf.conf` AND `conf.validation_eval` are both True.
With `validation_eval=False` (the shipped `emforecaster`/`cnn`/`mlp` jobs), calibration scores stay in memory and no `scores` file appears.

## Tuning (`tuning/tune.py`)

```
python -m forecasting.univariate.tuning.tune <job_name> [--cc] [--acc <account>]
```

Reads `<job>/ablation.yaml` (an `ablations:` grid plus optional `cc:` base) and expands the full `itertools.product` grid.
Local mode calls `main(job_name, ablation, ablation_id=i)` per combination.
`--cc` submits SLURM / Compute Canada jobs through `jobs/hpc/submit.py` (submitit); `--acc` overrides `slurm_account`.
HPC yaml templates live under `jobs/hpc/`.
Only the 8 `rf_emf/*` jobs have an `ablation.yaml`, so `test` cannot be grid-searched.

## Output layout

Each run writes to `logs/<exp.ablation_id>_<exp.model_id>_<exp.id>/<seed>/`:

- `args.yaml` - full resolved config (always).
- `log.json` - offline metric log (only when `exp.neptune=False`).
- `supervised.pth` - best-by-validation `state_dict`.
- `scores` - conformal calibration scores (only when `conf.conf` and `conf.validation_eval`).
- `plot_<idx>.png` - prediction-vs-target plots for `num_plots=10` randomly sampled test indices, y-axis "Normalized EMF Exposure (V/m)".

`exp.id` defaults to a random 10-char string but jobs may pin it (e.g. `emforecaster` uses `rf_emf_ptv2_L336_O336_cp`).

## Analysis (`analysis/`)

`analysis/neptune/` (`get_results.py`, `compile_results.py`, `post_hoc.py`, `stats.py`) aggregates Neptune runs.
`analysis/general/` holds standalone plotting and stats scripts (`adf.py`, `arima.py`, `amplitudes.py`, `bar_graph.py`, and others). Loss curves moved to `results/emfres/fig_curves.py`, which had been duplicated byte-for-byte here and under `forecasting/multivariate/`.
Neptune is optional and gated on `exp.neptune` plus `NEPTUNE_API_TOKEN`.

## Directory map

```
forecasting/univariate/
├── config/        config.py (pydantic Global, load_config)
├── conformal/     coverage.py, tradeoff.py
├── exp/           exp.py (Experiment.run)
├── models/        dlinear, emforecaster, linear, mamba, modern_tcn,
│                  patchtst, recurrent, rf_emf_baselines, timesnet, tsmixer
├── layers/        shared model building blocks
├── tuning/        tune.py, scripts/
├── jobs/
│   ├── exp/       rf_emf/{emforecaster,dlinear,patchtst,tsmixer},
│   │              rf_emf/baselines/{cnn,lstm,mlp,transformer}, test
│   └── hpc/       submit.py + SLURM templates
├── utils/         dataloading.py, datasets.py, models.py, train.py,
│                  logger.py, schedulers.py, utils.py, ...
└── analysis/      neptune/, general/
```
