> **Superseded.** Frozen predecessor of the current tables, kept as the
> provenance record. Live numbers: `results/tables/` (regenerate with
> `python results/make.py extract && python results/make.py tables`).

# Point-Forecasting Performance

Test-set metrics, averaged over seeds.
Datasets: **DET** = `rf_emf_det`, **PTV** = `rf_emf_ptv`, **PTV2** = `rf_emf_ptv2`.
$L$ = input/lookback length (`seq_len`), $O$ = forecast horizon (`pred_len`).
Values rounded to 4 decimals; **bold** = best (lowest) value in the row, among populated cells.

Results are split by whether the run used **conformal prediction** (`conf.conf` in each run's `args.yaml`, mirrored as `parameters/conf/conf` in `log.json`). Conformal runs hold out a calibration split and additionally log coverage and interval-width metrics (`sl_test/ic`, `sl_test/jc`, `sl_test/interval_width_mean`); non-conformal runs log only `sl_test/loss` and `sl_test/mae`.

> **Which models were run in each setting.** All six deep models (EMForecaster, DLinear, TSMixer, PatchTST, Transformer, LSTM) now have a full 12-config grid in **both** settings, so both sections below are populated for every deep model. ARIMA is a classical baseline with no conformal variant, so it appears only in the without-CP tables. Empty cells mean **no run of that type exists**, not a missing value (the only such cells here are ARIMA's entire Section 2).

---

# 1 · Point-Forecasting Performance (without conformal prediction)

Runs with `conf.conf = false`. All six deep models plus the classical **ARIMA** baseline are evaluated here.

## Table A | MSE

**Bold** = best (lowest) value in the row. ARIMA does not use a learned lookback, so its $L$192 and $L$336 rows are identical for a given (dataset, $O$).

| Dataset | $L$ | $O$ | EMForecaster (2025) | DLinear (2023) | TSMixer (2023) | PatchTST (2023) | Transformer (2017) | LSTM (1997) | ARIMA (1970) |
|---|---|---|---|---|---|---|---|---|---|
| DET  | 192 | 96 | **0.3111** | 0.4524 | 0.3678 | 0.4153 | 1.0383 | 0.3404 | 1.2371 |
| DET  | 192 | 336 | **0.4229** | 0.5306 | 0.4583 | 0.5373 | 1.1477 | 0.4948 | 1.4194 |
| DET  | 336 | 96 | **0.3009** | 0.3995 | 0.3560 | 0.3963 | 1.0601 | 0.3500 | 1.2371 |
| DET  | 336 | 336 | **0.3940** | 0.5046 | 0.4619 | 0.5127 | 1.4810 | 0.4834 | 1.4194 |
| PTV  | 192 | 96 | **0.2614** | 0.4710 | 0.3212 | 0.3772 | 1.0271 | 0.3501 | 1.1840 |
| PTV  | 192 | 336 | **0.3198** | 0.4747 | 0.3657 | 0.4538 | 1.3877 | 0.5308 | 1.3061 |
| PTV  | 336 | 96 | **0.2435** | 0.3545 | 0.3074 | 0.3263 | 1.0398 | 0.3514 | 1.1840 |
| PTV  | 336 | 336 | **0.3137** | 0.4256 | 0.3846 | 0.3876 | 1.1085 | 0.5473 | 1.3061 |
| PTV2 | 192 | 96 | **0.3220** | 0.4003 | 0.3348 | 0.3994 | 0.3746 | 0.3293 | 1.0128 |
| PTV2 | 192 | 336 | **0.3586** | 0.4130 | 0.3594 | 0.4486 | 0.8356 | 0.3627 | 1.1346 |
| PTV2 | 336 | 96 | **0.2999** | 0.3312 | 0.3038 | 0.3428 | 0.7625 | 0.3170 | 1.0128 |
| PTV2 | 336 | 336 | 0.3387 | 0.3745 | **0.3354** | 0.4141 | 0.8474 | 0.3542 | 1.1346 |

## Table B | Training Time

Actual training time logged per run (`parameters/running_time` in each `log.json`), averaged over seeds. Times are full training runs with early stopping (patience 20, max 100 epochs). **Bold** = fastest in the row. ARIMA is not gradient-trained and logs no `running_time` (`X`).

| Dataset | $L$ | $O$ | EMForecaster (2025) | DLinear (2023) | TSMixer (2023) | PatchTST (2023) | Transformer (2017) | LSTM (1997) | ARIMA (1970) |
|---|---|---|---|---|---|---|---|---|---|
| DET  | 192 | 96 | 18m 34s | **36s** | 1m 11s | 1m 42s | 5m 16s | 2m 3s | X |
| DET  | 192 | 336 | 11m 13s | **1m 1s** | 1m 10s | 1m 34s | 6m 40s | 2m 22s | X |
| DET  | 336 | 96 | 26m 21s | **39s** | 1m 26s | 2m 40s | 9m 40s | 2m 4s | X |
| DET  | 336 | 336 | 19m 55s | **1m 8s** | 1m 29s | 3m 0s | 8m 12s | 2m 17s | X |
| PTV  | 192 | 96 | 7m 23s | **22s** | 42s | 54s | 4m 4s | 1m 16s | X |
| PTV  | 192 | 336 | 7m 18s | **38s** | 50s | 39s | 1m 20s | 1m 26s | X |
| PTV  | 336 | 96 | 8m 28s | **28s** | 54s | 2m 3s | 6m 42s | 1m 18s | X |
| PTV  | 336 | 336 | 8m 32s | **54s** | 1m 8s | 1m 40s | 5m 28s | 1m 24s | X |
| PTV2 | 192 | 96 | 17m 24s | **54s** | 1m 56s | 2m 14s | 9m 32s | 4m 27s | X |
| PTV2 | 192 | 336 | 14m 27s | 1m 58s | 2m 37s | **1m 12s** | 13m 58s | 5m 21s | X |
| PTV2 | 336 | 96 | 21m 36s | **1m 12s** | 2m 38s | 3m 56s | 17m 22s | 4m 41s | X |
| PTV2 | 336 | 336 | 31m 0s | 2m 53s | **2m 32s** | 2m 40s | 15m 12s | 5m 22s | X |

## Table C | Per-Epoch Training Time

Table B uses early stopping, so different runs stop at different epochs and a slower-per-step model can post a shorter total just because it stopped sooner. This table divides each run's logged `running_time` by the number of epochs it actually ran (`len(sl_train/loss)`) **per seed**, then averages over seeds, a better estimate of relative cost than Table B. **Bold** = fastest in the row.

| Dataset | $L$ | $O$ | EMForecaster (2025) | DLinear (2023) | TSMixer (2023) | PatchTST (2023) | Transformer (2017) | LSTM (1997) | ARIMA (1970) |
|---|---|---|---|---|---|---|---|---|---|
| DET  | 192 | 96 | 12.4s | **0.4s** | 0.7s | 1.0s | 7.0s | 1.2s | X |
| DET  | 192 | 336 | 10.3s | **0.6s** | 0.7s | 1.3s | 10.0s | 1.4s | X |
| DET  | 336 | 96 | 21.4s | **0.4s** | 0.9s | 1.6s | 17.2s | 1.2s | X |
| DET  | 336 | 336 | 17.3s | **0.7s** | 0.9s | 1.8s | 19.6s | 1.4s | X |
| PTV  | 192 | 96 | 5.5s | **0.2s** | 0.4s | 0.5s | 3.8s | 0.8s | X |
| PTV  | 192 | 336 | 7.0s | **0.4s** | 0.5s | 0.5s | 5.5s | 0.9s | X |
| PTV  | 336 | 96 | 8.8s | **0.3s** | 0.5s | 1.2s | 10.3s | 0.8s | X |
| PTV  | 336 | 336 | 10.2s | **0.5s** | 0.7s | 1.0s | 12.0s | 0.8s | X |
| PTV2 | 192 | 96 | 19.2s | **0.5s** | 1.2s | 1.7s | 12.3s | 2.7s | X |
| PTV2 | 192 | 336 | 18.8s | **1.2s** | 1.6s | 1.6s | 19.6s | 3.2s | X |
| PTV2 | 336 | 96 | 27.9s | **0.7s** | 1.6s | 3.2s | 36.7s | 2.8s | X |
| PTV2 | 336 | 336 | 38.9s | 1.7s | **1.5s** | 3.7s | 39.6s | 3.2s | X |

### Per-model summary

**Mean / epoch** divides each run's logged `parameters/running_time` by the epochs it actually ran (`len(sl_train/loss)`), then averages over every config and seed for the model. **Estimation for 100 ep** scales that mean to the 100-epoch early-stopping cap (`mean × 100`). Weights ranges and exact per-config parameter counts are CP-independent and listed once in [Model sizes](#model-sizes-shared-across-both-settings).

| Model | Weights | Mean / epoch | Estimation for 100 ep |
|---|---:|---:|---:|
| DLinear (2023)     | 37k–226k | **0.64s** | **~1m 4s** |
| TSMixer (2023)     | 31k–135k | 0.93s | ~1m 33s |
| PatchTST (2023)    | 44k–234k | 1.61s | ~2m 41s |
| LSTM (1997)        | 32k–58k | 1.70s | ~2m 50s |
| EMForecaster (2025) | 1.5M–7.6M | 16.10s | ~26m 50s |
| Transformer (2017) | 131k–456k | 16.14s | ~26m 54s |

## Notes

- **MSE:** EMForecaster's without-CP point forecasts post the lowest MSE in 11 of 12 settings; TSMixer edges it out only on PTV2 / $L$336 / $O$336 (0.3354 vs 0.3387). Every deep model beats the classical ARIMA baseline by a wide margin (deep-model MSE roughly 0.25–0.55 against ARIMA's 1.01–1.42). The Transformer is the weakest deep model, exceeding 1.0 MSE on most DET and PTV settings.
- **Training time:** DLinear is the fastest model to train in every row; EMForecaster is the slowest by a large margin (it is also by far the largest model, 1.5M–7.6M weights). Some EMForecaster rows in Table B look out of order (for example its `O=336` faster than its `O=96`); this is an early-stopping artifact that the per-epoch normalization in Table C removes.
- **Imputed training times (EMForecaster only):** Four EMForecaster seeds logged anomalously long wall-clock times for an otherwise normal run (correct epoch count, valid loss), caused by the machine being throttled or idle-suspended during an overnight run rather than by any extra computation. Three of them fall in this without-CP section: PTV2 / $L$192 / $O$336 seed 69421 (1hr 17min logged), DET / $L$192 / $O$96 seed 69421 (1hr 11min), and DET / $L$336 / $O$96 seed 69423 (1hr 0min). Each was replaced with the mean per-epoch time of the clean seeds of the **same (model, dataset, $L$, $O$) configuration**, rescaled by that seed's actual epoch count, so the substitute is an average of directly comparable runs rather than an invented number. Only `parameters/running_time` was changed; MSE, MAE, coverage, epoch counts, and every other metric are the originals.
- **Seed aggregation:** EMForecaster DET rows average **5 seeds** (69421–69425); EMForecaster PTV and PTV2 rows average **2 seeds** (69421, 69422). The other five deep models (DLinear, TSMixer, PatchTST, Transformer, LSTM) average **2 seeds** (69421, 69422) in every row. So each DET row mixes a 5-seed EMForecaster cell with 2-seed cells in the other columns.
- A 3-epoch smoke-test run (`1_EMForecaster_cptest_off`, PTV2 / $L$336 / $O$336, single seed 69421) is excluded from all aggregates.

---

# 2 · Point-Forecasting Performance (with conformal prediction)

Runs with `conf.conf = true` at the significance level **$\alpha$ = 0.1 (90% target coverage)**. All six deep models have a full grid here. ARIMA has no conformal variant, so its column is blank.

## Table A | MSE

**Bold** = best (lowest) value in the row.

| Dataset | $L$ | $O$ | EMForecaster (2025) | DLinear (2023) | TSMixer (2023) | PatchTST (2023) | Transformer (2017) | LSTM (1997) | ARIMA (1970) |
|---|---|---|---|---|---|---|---|---|---|
| DET  | 192 | 96 | **0.3124** | 0.4517 | 0.3648 | 0.4400 | 1.0383 | 0.3404 |  |
| DET  | 192 | 336 | **0.4239** | 0.5303 | 0.4639 | 0.5440 | 1.1477 | 0.4949 |  |
| DET  | 336 | 96 | **0.3009** | 0.3993 | 0.3551 | 0.3957 | 1.0601 | 0.3500 |  |
| DET  | 336 | 336 | **0.3966** | 0.5048 | 0.4602 | 0.5111 | 1.4810 | 0.4834 |  |
| PTV  | 192 | 96 | **0.2614** | 0.4710 | 0.3212 | 0.3772 | 1.0271 | 0.3501 |  |
| PTV  | 192 | 336 | **0.3198** | 0.4747 | 0.3657 | 0.4538 | 1.3877 | 0.5307 |  |
| PTV  | 336 | 96 | **0.2435** | 0.3545 | 0.3074 | 0.3263 | 1.0398 | 0.3514 |  |
| PTV  | 336 | 336 | **0.3137** | 0.4256 | 0.3846 | 0.3876 | 1.1085 | 0.5473 |  |
| PTV2 | 192 | 96 | **0.3220** | 0.4003 | 0.3348 | 0.3994 | 0.3746 | 0.3293 |  |
| PTV2 | 192 | 336 | **0.3586** | 0.4130 | 0.3594 | 0.4486 | 0.8356 | 0.3626 |  |
| PTV2 | 336 | 96 | **0.2999** | 0.3312 | 0.3038 | 0.3428 | 0.7625 | 0.3169 |  |
| PTV2 | 336 | 336 | 0.3387 | 0.3745 | **0.3354** | 0.4141 | 0.8474 | 0.3541 |  |

## Table B | Training Time

Actual logged `parameters/running_time`, averaged over seeds (early stopping, patience 20, max 100 epochs). **Bold** = fastest in the row.

| Dataset | $L$ | $O$ | EMForecaster (2025) | DLinear (2023) | TSMixer (2023) | PatchTST (2023) | Transformer (2017) | LSTM (1997) | ARIMA (1970) |
|---|---|---|---|---|---|---|---|---|---|
| DET  | 192 | 96 | 11m 25s | **43s** | 1m 23s | 2m 36s | 4m 22s | 2m 26s | X |
| DET  | 192 | 336 | 10m 14s | **1m 58s** | 2m 20s | 2m 28s | 5m 22s | 2m 30s | X |
| DET  | 336 | 96 | 19m 56s | **49s** | 1m 13s | 2m 26s | 8m 28s | 2m 23s | X |
| DET  | 336 | 336 | 14m 41s | **2m 8s** | 2m 15s | 4m 28s | 8m 21s | 2m 30s | X |
| PTV  | 192 | 96 | 6m 27s | **36s** | 44s | 1m 7s | 4m 17s | 1m 24s | X |
| PTV  | 192 | 336 | 5m 18s | **1m 9s** | 1m 23s | 1m 24s | 1m 31s | 1m 32s | X |
| PTV  | 336 | 96 | 7m 6s | **37s** | 53s | 2m 12s | 7m 16s | 1m 24s | X |
| PTV  | 336 | 336 | 7m 58s | **1m 22s** | 1m 36s | 2m 53s | 5m 37s | 1m 33s | X |
| PTV2 | 192 | 96 | 17m 39s | **1m 22s** | 2m 5s | 2m 58s | 9m 9s | 4m 40s | X |
| PTV2 | 192 | 336 | 15m 40s | 3m 34s | 4m 8s | **1m 44s** | 12m 32s | 5m 32s | X |
| PTV2 | 336 | 96 | 30m 13s | **1m 36s** | 3m 4s | 5m 44s | 15m 45s | 4m 49s | X |
| PTV2 | 336 | 336 | 30m 12s | 3m 55s | 4m 43s | **2m 50s** | 15m 50s | 5m 36s | X |

## Table C | Per-Epoch Training Time

Logged `running_time` divided by epochs actually run (`len(sl_train/loss)`) per seed, then averaged over seeds. **Bold** = fastest in the row.

| Dataset | $L$ | $O$ | EMForecaster (2025) | DLinear (2023) | TSMixer (2023) | PatchTST (2023) | Transformer (2017) | LSTM (1997) | ARIMA (1970) |
|---|---|---|---|---|---|---|---|---|---|
| DET  | 192 | 96 | 7.7s | **0.4s** | 0.8s | 1.6s | 5.9s | 1.5s | X |
| DET  | 192 | 336 | 9.2s | **1.2s** | 1.4s | 1.7s | 8.6s | 1.5s | X |
| DET  | 336 | 96 | 16.1s | **0.5s** | 0.7s | 1.5s | 15.2s | 1.4s | X |
| DET  | 336 | 336 | 12.7s | **1.3s** | 1.3s | 2.7s | 19.6s | 1.5s | X |
| PTV  | 192 | 96 | 4.8s | **0.4s** | 0.4s | 0.7s | 4.0s | 0.8s | X |
| PTV  | 192 | 336 | 5.1s | **0.7s** | 0.8s | 1.0s | 6.3s | 0.9s | X |
| PTV  | 336 | 96 | 7.4s | **0.4s** | 0.5s | 1.3s | 11.0s | 0.8s | X |
| PTV  | 336 | 336 | 9.5s | **0.8s** | 1.0s | 1.7s | 12.6s | 0.9s | X |
| PTV2 | 192 | 96 | 19.5s | **0.8s** | 1.2s | 2.2s | 11.8s | 2.8s | X |
| PTV2 | 192 | 336 | 20.4s | **2.1s** | 2.5s | 2.3s | 17.4s | 3.3s | X |
| PTV2 | 336 | 96 | 39.0s | **1.0s** | 1.8s | 4.6s | 33.2s | 2.9s | X |
| PTV2 | 336 | 336 | 38.1s | **2.3s** | 2.8s | 4.0s | 40.9s | 3.4s | X |

### Per-model summary

| Model | Weights | Mean / epoch | Estimation for 100 ep |
|---|---:|---:|---:|
| DLinear (2023)     | 37k–226k | **0.94s** | **~1m 34s** |
| TSMixer (2023)     | 31k–135k | 1.22s | ~2m 2s |
| LSTM (1997)        | 32k–58k | 1.82s | ~3m 2s |
| PatchTST (2023)    | 44k–234k | 2.02s | ~3m 22s |
| Transformer (2017) | 131k–456k | 15.54s | ~25m 54s |
| EMForecaster (2025) | 1.5M–7.6M | 15.80s | ~26m 20s |

(EMForecaster's CP grid had one contaminated wall-clock seed on PTV2 / $L$192 / $O$336 that logged 8hr 24min for 46 epochs; it has been imputed from the clean seeds of that same configuration, the same way as the without-CP runs. See the imputation note in Section 1.)

## Table D | Marginal Coverage (%)

Per-timestep empirical coverage of the conformal intervals (`sl_test/ic`). **Target = 90%** ($\alpha$=0.1); values above are conservative (wider than needed), values below indicate under-coverage. Not bolded, since "best" means closest to the 90% target rather than highest.

| Dataset | $L$ | $O$ | EMForecaster (2025) | DLinear (2023) | TSMixer (2023) | PatchTST (2023) | Transformer (2017) | LSTM (1997) | ARIMA (1970) |
|---|---|---|---|---|---|---|---|---|---|
| DET  | 192 | 96 | 90.65 | 91.71 | 91.33 | 91.60 | 92.38 | 88.79 |  |
| DET  | 192 | 336 | 91.20 | 91.71 | 91.70 | 92.01 | 90.20 | 89.21 |  |
| DET  | 336 | 96 | 90.86 | 91.63 | 91.21 | 91.20 | 89.86 | 87.51 |  |
| DET  | 336 | 336 | 91.31 | 91.62 | 91.68 | 91.64 | 92.24 | 88.81 |  |
| PTV  | 192 | 96 | 88.45 | 86.39 | 88.64 | 87.56 | 80.73 | 83.30 |  |
| PTV  | 192 | 336 | 87.21 | 84.63 | 86.39 | 86.02 | 81.64 | 81.15 |  |
| PTV  | 336 | 96 | 88.91 | 86.84 | 90.12 | 88.67 | 79.98 | 82.84 |  |
| PTV  | 336 | 336 | 87.80 | 84.89 | 87.30 | 86.14 | 82.97 | 80.74 |  |
| PTV2 | 192 | 96 | 87.27 | 86.00 | 85.92 | 86.56 | 90.45 | 89.85 |  |
| PTV2 | 192 | 336 | 85.81 | 84.78 | 84.12 | 88.75 | 90.88 | 89.67 |  |
| PTV2 | 336 | 96 | 87.18 | 86.19 | 85.96 | 87.15 | 88.04 | 90.12 |  |
| PTV2 | 336 | 336 | 86.10 | 84.63 | 83.97 | 89.23 | 90.67 | 89.71 |  |

## Table E | Joint Coverage (%)

Fraction of test windows where the interval simultaneously covers **all** forecast horizons (`sl_test/jc`). This is much stricter than marginal coverage and, without Bonferroni correction (`conf.corrected = false`), falls far below the nominal level, especially as $O$ grows.

| Dataset | $L$ | $O$ | EMForecaster (2025) | DLinear (2023) | TSMixer (2023) | PatchTST (2023) | Transformer (2017) | LSTM (1997) | ARIMA (1970) |
|---|---|---|---|---|---|---|---|---|---|
| DET  | 192 | 96 | 40.89 | 49.31 | 47.32 | 50.78 | 50.41 | 19.11 |  |
| DET  | 192 | 336 | 20.59 | 27.00 | 25.40 | 28.15 | 18.47 | 2.06 |  |
| DET  | 336 | 96 | 39.63 | 50.41 | 45.42 | 47.38 | 43.42 | 13.69 |  |
| DET  | 336 | 336 | 20.39 | 27.75 | 24.83 | 26.06 | 22.80 | 1.17 |  |
| PTV  | 192 | 96 | 23.78 | 27.17 | 31.94 | 29.31 | 28.43 | 7.38 |  |
| PTV  | 192 | 336 | 4.66 | 5.67 | 5.16 | 6.39 | 2.79 | 0.26 |  |
| PTV  | 336 | 96 | 24.57 | 23.18 | 38.09 | 24.44 | 29.18 | 7.48 |  |
| PTV  | 336 | 336 | 5.84 | 6.42 | 8.13 | 2.51 | 3.67 | 0.33 |  |
| PTV2 | 192 | 96 | 20.14 | 17.31 | 18.23 | 19.69 | 7.14 | 4.27 |  |
| PTV2 | 192 | 336 | 1.45 | 0.84 | 1.10 | 4.74 | 7.97 | 0.04 |  |
| PTV2 | 336 | 96 | 18.93 | 17.95 | 17.55 | 20.13 | 10.34 | 3.12 |  |
| PTV2 | 336 | 336 | 1.80 | 0.76 | 1.02 | 7.19 | 6.98 | 0.01 |  |

## Table F | Mean Interval Width

Average width of the conformal prediction intervals (`sl_test/interval_width_mean`), in the scaled target units. Narrower is better **only when coverage is met**, a narrow interval that under-covers (cross-reference Table D) is not an improvement.

| Dataset | $L$ | $O$ | EMForecaster (2025) | DLinear (2023) | TSMixer (2023) | PatchTST (2023) | Transformer (2017) | LSTM (1997) | ARIMA (1970) |
|---|---|---|---|---|---|---|---|---|---|
| DET  | 192 | 96 | 3.389 | 3.578 | 3.525 | 3.583 | 3.375 | 1.712 |  |
| DET  | 192 | 336 | 4.531 | 4.603 | 4.609 | 4.803 | 3.457 | 2.170 |  |
| DET  | 336 | 96 | 3.430 | 3.605 | 3.508 | 3.493 | 3.252 | 1.665 |  |
| DET  | 336 | 336 | 4.602 | 4.638 | 4.608 | 4.599 | 4.172 | 2.122 |  |
| PTV  | 192 | 96 | 2.973 | 2.977 | 3.144 | 3.099 | 2.575 | 1.573 |  |
| PTV  | 192 | 336 | 3.814 | 3.677 | 3.900 | 4.104 | 3.078 | 1.868 |  |
| PTV  | 336 | 96 | 2.968 | 2.937 | 3.240 | 3.024 | 2.525 | 1.548 |  |
| PTV  | 336 | 336 | 3.831 | 3.767 | 4.005 | 3.920 | 2.827 | 1.884 |  |
| PTV2 | 192 | 96 | 3.185 | 3.020 | 3.056 | 3.072 | 1.986 | 1.827 |  |
| PTV2 | 192 | 336 | 3.934 | 3.764 | 3.829 | 4.214 | 3.157 | 1.913 |  |
| PTV2 | 336 | 96 | 3.145 | 3.038 | 3.055 | 3.169 | 2.594 | 1.790 |  |
| PTV2 | 336 | 336 | 3.949 | 3.789 | 3.831 | 4.435 | 3.156 | 1.878 |  |

## Notes

- **MSE (point forecasts):** With conformal prediction, EMForecaster posts the lowest MSE in 11 of 12 settings; TSMixer wins the remaining one (PTV2 / $L$336 / $O$336, 0.3354 vs 0.3387). Among the smaller models TSMixer is generally the strongest, with LSTM close behind on DET and PTV2. The Transformer is the weakest by a wide margin on DET and PTV (≈1.03–1.48 MSE) and trails on PTV2 except at $L$192 / $O$96. EMForecaster's CP point forecasts are essentially identical to its without-CP forecasts (its conformal calibration is applied on top of the trained model without changing the point predictions), so its Section 1 and Section 2 MSE columns nearly coincide; the other models retrain after the calibration split and shift slightly between sections.
- **Coverage.** At $\alpha$=0.1 (90% target), marginal coverage lands near the nominal level on DET (≈87.5–92.4%, mostly slightly conservative) but under-covers on PTV and PTV2 for several models (Transformer and LSTM dip to ≈80% on PTV). EMForecaster's marginal coverage is close to target on DET (≈90.6–91.3%) and somewhat low on PTV/PTV2 (≈85.8–88.9%).
- **Joint vs marginal.** Joint coverage (Table E) is far below marginal coverage and collapses as the horizon $O$ grows (often near 0% at $O$=336). This is the expected consequence of running per-step conformal calibration without the Bonferroni correction (`conf.corrected = false`); covering all horizons simultaneously requires the corrected critical scores.
- **Interval width tracks model scale and dataset.** LSTM produces the narrowest intervals everywhere, but also among the lowest coverage, so its width advantage partly reflects under-coverage rather than sharper uncertainty.
- **Seed aggregation:** DLinear, TSMixer, and PatchTST average **5 seeds** on DET and **2 seeds** on PTV/PTV2. EMForecaster, LSTM, and Transformer average **2 seeds** (69421, 69422) in every row. So each DET row mixes 5-seed and 2-seed cells across columns.

---

# Model sizes (shared across both settings)

Parameter counts depend only on $L$ and $O$ (and the model), not on the dataset or on whether conformal prediction is enabled, so they are reported once here. Weights come from `parameters/sl/num_params` in each log.

| Model | L192 / O96 | L192 / O336 | L336 / O96 | L336 / O336 |
|---|---:|---:|---:|---:|
| DLinear (2023)     | 37,056    | 129,696   | 64,704    | 226,464   |
| TSMixer (2023)     | 31,139    | 77,459    | 54,323    | 135,203   |
| LSTM (1997)        | 32,096    | 40,016    | 50,528    | 58,448    |
| PatchTST (2023)    | 44,194    | 136,594   | 72,130    | 233,650   |
| Transformer (2017) | 130,528   | 179,248   | 372,736   | 456,016   |
| EMForecaster (2025)| 1,495,235 | 4,444,595 | 2,416,907 | 7,578,107 |
