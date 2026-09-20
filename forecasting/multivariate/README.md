# forecasting/multivariate

Multivariate EMF forecasting: jointly forecasting many frequency-band channels at once.

This setup is a near-identical twin of [`../univariate`](../univariate/README.md).
It reuses the same config system, experiment loop, models, conformal stack, tuner, and job layout.
Read the univariate README first for the shared mechanics (dispatch flags, `Experiment.run` flow, conformal, output layout, tuning).
See also the setup hub at [`../README.md`](../README.md) and the repo root [`../../README.md`](../../README.md).

This document only covers what is genuinely different here.

## What multivariate means here

Each dataset is a table of many frequency-band channels sampled over time.
The task is to forecast all channels jointly rather than one EMF value.
Channels are handled by channel independence: every channel is normalized and forecast on its own, sharing one set of model weights, then the per-channel forecasts are stacked back into a `(B, C, pred_len)` tensor.

## The five files that actually differ

Every other `*.py` under `forecasting/multivariate/` is byte-identical to its univariate twin once the `forecasting.univariate` import prefix is renamed to `forecasting.multivariate`.
`conformal/coverage.py` and `conformal/tradeoff.py` are byte-identical between the two setups.
The genuine multivariate logic lives in exactly five files:

1. `config/config.py`
2. `utils/dataloading.py`
3. `models/emforecaster.py`
4. `utils/models.py`
5. `exp/exp.py`

(`tuning/tune.py` and `jobs/hpc/submit.py` also differ, but only in hardcoded `.../multivariate/...` path strings, not behavior.)

### 1. config/config.py: one added field

The only config difference is a single field on the `exp` group:

```python
eval_test_per_epoch: bool = Field(default=False, ...)
```

When true, the test loader is evaluated every epoch and its loss is logged, producing a per-epoch test-loss curve.
Model selection still uses the validation loss only, so there is no leakage; the final reported test metric is the best (validation-selected) model.
Nothing else in the config differs from univariate.

### 2. utils/dataloading.py: Italy datasets, drop/interpolate, auto num_channels

A new `load_forecasting` branch handles the five multivariate datasets, reading from `data/multivariate/processed/<name>.csv`.
Before use it drops the aggregate and leakage columns: any column named `Timestamp`, any column starting with `EMF_`, and the `PTV` column.
It coerces the remaining bands to numeric and repairs gaps with linear interpolation (`interpolate(method="linear", limit_direction="both").ffill().bfill()`).
The `get_loaders` whitelist adds the same five dataset ids.

Crucially, `num_channels` is auto-set from the data: after loading, `if not args.data.univariate: args.data.num_channels = int(train_data.shape[0])`.
The band count in the loaded CSV wins (27 or 29 across the Italy files), so the `num_channels` value in `args.yaml` is advisory only.

### 3. models/emforecaster.py: channel independence

`forward` takes `x` of shape `(B, C, L)`:

- per-channel RevIN normalize over `(B, C, L)` (RevIN is built with `num_channels`),
- `x = x.reshape(B*C, 1, L)` folds the channels into the batch dimension,
- forecast jointly through the shared patch backbone in one pass,
- `x = x.reshape(B, C, pred_len)` unfolds before per-channel RevIN denormalize,
- return `(B, C, pred_len)`.

For `C == 1` this reduces exactly to the univariate path.

### 4. utils/models.py: TimesNet head + forecasting squeeze skip

TimesNet is built with `c_out=args.data.num_channels`, so its head emits one output per channel (univariate used `args.timesnet.c_out`).
In the `prepare_output_and_target` post-processing helper, forecasting returns early without the classification squeeze: `if args.exp.task == "forecasting": return output.to(device), target.to(device).float()`.
This preserves the `(batch, num_channels, pred_len)` layout and avoids dropping the batch dim on a size-1 final batch.

### 5. exp/exp.py: per-epoch test eval, per-channel MSE, 3-D conformal

- Per-epoch test loss gated on `eval_test_per_epoch`: when set, `supervised_train` builds the test loader up front (`loader_type="all"`), passes it into `train`, and logs a per-epoch test loss. Model selection still uses validation only.
- Per-channel MSE at evaluate time for 3-D forecasting preds: `per_channel_mse = ((preds-targets)**2).mean(dim=(0,2))`, with `stats["mse"]` the channel mean and `stats["mse_per_channel"]` the per-band vector.
- Timing logs: per-epoch `epoch_time` and `total_training_time` / `total_training_seconds`.
- Conformal reshaping: 3-D `(n_samples, num_channels, pred_len)` preds/targets are transposed to `(n_samples, pred_len, num_channels)` so the critical-scores / coverage / Bonferroni stack is per-horizon-step and per-channel.
- Conformal recalibration on the best model: when `conf.conf and not conf.validation_eval`, `test()` rebuilds the loaders and re-runs calibration on the validation-selected checkpoint so calibration and test residuals come from the same predictor.

## Datasets

On disk in `data/multivariate/processed/`: `ED2022.csv`, `ED2024.csv`, `UH2022.csv`, `UH2024.csv`, `EMF_2023_university_5_with_PTV.csv`.
Unprocessed twins live in `data/multivariate/unprocessed/`.
After dropping `Timestamp` / `EMF_*` / `PTV` the loaded band count is 27 or 29 depending on the file.

The same directory also holds `VOIP_MOBILE.csv` and `VOIP_FIXED.csv`, built from `data/ML_DATASETS/` by `build_voip.py`. They take a different path through `load_forecasting`: `pooled_trace_splits` splits each of their eight codec traces on its own and concatenates the parts, flagging one row per join so the existing synthetic-window exclusion drops any window that would span two calls. `Trace` is dropped alongside `Timestamp`, leaving six channels. See the repo README's *VoIP call-quality data*.

## args.yaml switches vs univariate

The multivariate jobs flip three switches relative to the univariate jobs:

- `data.univariate: False`
- `sl.dataset_class: "forecasting"` (windows sliced on the fly, `__getitem__` returns `(C, seq_len), (C, pred_len)`)
- `exp.eval_test_per_epoch: True`

They also set `exp.task: "forecasting"`, `data.dataset: "ED2022"`, and `data.num_channels: 27` (advisory, auto-overwritten at load time).

## Jobs and how to run them

`jobs/exp/rf_emf_mv/` holds eight model jobs plus a smoke test, one model each:

| job_name | model_id | notes |
| --- | --- | --- |
| `rf_emf_mv/arima` | ARIMA | classical, `exp.no_train: True` (no epoch loop, no checkpoint); one statsmodels ARIMA per band |
| `rf_emf_mv/dlinear` | DLinear | |
| `rf_emf_mv/emforecaster` | EMForecaster | `patch_model_id: TSMixer`, `conf.conf: True` |
| `rf_emf_mv/lstm` | LSTM | `models/recurrent.py::RecurrentModel`, `backbone_id: LSTM`; requires `data.patching: True`, `last_state: False`, `avg_state: True`, `revout: True` |
| `rf_emf_mv/patchtst` | PatchTST | in the grid until 2026-07-24; job still runnable |
| `rf_emf_mv/persistence` | Persistence | last-value baseline, `exp.no_train: True` (no epoch loop, no checkpoint); VoIP grids only |
| `rf_emf_mv/timesnet` | TimesNet | multivariate-only; no univariate TimesNet job exists |
| `rf_emf_mv/transformer` | RF_EMF_Transformer | channel-independent fold; `d_model` must be 2, `batch_size` reduced to 128 |
| `rf_emf_mv/tsmixer` | TSMixer | |
| `rf_emf_mv/test` | EMForecaster | multivariate smoke test (5 epochs, `seq_len` 20 / `pred_len` 30, ED2022, `num_workers: 0`); the multivariate counterpart of the univariate `test` job |

Run one from the repository root:

```bash
python main.py rf_emf_mv/emforecaster

# Fast end-to-end pipeline check (single seed):
python main.py rf_emf_mv/test --quick
```

`main.py` dispatches to `forecasting.multivariate` because the job_name starts with `rf_emf_mv`; every other prefix goes to `forecasting.univariate`.
The same root flags apply: `--quick` (first seed only), `--ablation '<json>'`, `--ablation_id`.
Note that the effective batch through the patch backbone is `batch_size * num_channels`, so lower `sl.batch_size` if memory is tight.

## Tuning caveat

`python -m forecasting.multivariate.tuning.tune <job_name> [--cc] [--acc <account>]` reads `<job>/ablation.yaml`.
The `rf_emf_mv/*` folders contain only `args.yaml` and no `ablation.yaml`, so the tuner cannot grid-search the multivariate-specific jobs out of the box.

## Mirrored univariate jobs are dead here

`jobs/exp/` also mirrors the univariate `rf_emf/*` (plus `baselines/{cnn,lstm,mlp,transformer}`) and `test/` jobs.
Those job_names do not start with `rf_emf_mv`, so `main.py` dispatches them to `forecasting.univariate`.
The copies under `forecasting/multivariate/jobs/exp/` are effectively unused; run those experiments through the univariate setup instead.

## Directory map

```
forecasting/multivariate/
  config/       config.py (adds exp.eval_test_per_epoch), load_config -> Global
  exp/          exp.py (per-epoch test eval, per-channel MSE, 3-D conformal)
  models/       emforecaster.py (channel independence) + dlinear, linear, mamba,
                modern_tcn, patchtst, recurrent, rf_emf_baselines, timesnet, tsmixer
  layers/       shared building blocks
  utils/        dataloading.py (Italy datasets, auto num_channels),
                models.py (TimesNet c_out=num_channels, forecasting squeeze skip),
                datasets.py, train.py, logger.py, schedulers.py, ...
  conformal/    coverage.py, tradeoff.py (byte-identical to univariate)
  tuning/       tune.py (needs <job>/ablation.yaml), scripts/
  jobs/         exp/rf_emf_mv/{dlinear,emforecaster,patchtst,timesnet,tsmixer},
                mirrored exp/rf_emf/* + test (dispatch to univariate), hpc/
  analysis/     general/ plotting + neptune/ aggregation
```
