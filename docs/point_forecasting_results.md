# Point-Forecasting Performance

Test-set metrics, averaged over seeds.
Datasets: **DET** = `rf_emf_det`, **PTV** = `rf_emf_ptv`, **PTV2** = `rf_emf_ptv2`.
$L$ = input/lookback length (`seq_len`), $O$ = forecast horizon (`pred_len`).
Values rounded to 4 decimals; **bold** = best (lowest) value in the row, among populated cells.

Results are split by whether the run used **conformal prediction** (`conf.conf` in each run's `args.yaml`, mirrored as `parameters/conf/conf` in `log.json`). Conformal runs hold out a calibration split and additionally log coverage and interval-width metrics (`sl_test/ic`, `sl_test/jc`, `sl_test/interval_width_mean`); non-conformal runs log only `sl_test/loss` and `sl_test/mae`.

> **Which models were run in each setting.** Every model *except* EMForecaster was run **only** with conformal prediction enabled, so the without-CP tables below are blank for them. EMForecaster is the opposite: it has a full grid of **without-CP** runs but no with-CP runs. ARIMA is a classical baseline with no conformal variant, so it appears only in the without-CP tables. Empty cells mean **no run of that type exists**, not a missing value.

---

# 1 · Point-Forecasting Performance (without conformal prediction)

Runs with `conf.conf = false`. Only **EMForecaster** (deep model) and **ARIMA** (classical baseline) were evaluated without conformal prediction; all other model columns are intentionally blank.

## Table A | MSE

**Bold** = best (lowest) value in the row. ARIMA does not use a learned lookback, so its $L$192 and $L$336 rows are identical for a given (dataset, $O$).

| Dataset | $L$ | $O$ | EMForecaster (2025) | DLinear (2023) | TSMixer (2023) | PatchTST (2023) | Transformer (2017) | LSTM (1997) | ARIMA (1970) |
|---|---|---|---|---|---|---|---|---|---|
| DET  | 192 | 96  | **0.3111** |  |  |  |  |  | 1.2371 |
| DET  | 192 | 336 | **0.4229** |  |  |  |  |  | 1.4194 |
| DET  | 336 | 96  | **0.3009** |  |  |  |  |  | 1.2371 |
| DET  | 336 | 336 | **0.3940** |  |  |  |  |  | 1.4194 |
| PTV  | 192 | 96  | **0.2614** |  |  |  |  |  | 1.1840 |
| PTV  | 192 | 336 | **0.3198** |  |  |  |  |  | 1.3061 |
| PTV  | 336 | 96  | **0.2435** |  |  |  |  |  | 1.1840 |
| PTV  | 336 | 336 | **0.3137** |  |  |  |  |  | 1.3061 |
| PTV2 | 192 | 96  | **0.3220** |  |  |  |  |  | 1.0128 |
| PTV2 | 192 | 336 | **0.3586** |  |  |  |  |  | 1.1346 |
| PTV2 | 336 | 96  | **0.2999** |  |  |  |  |  | 1.0128 |
| PTV2 | 336 | 336 | **0.3387** |  |  |  |  |  | 1.1346 |

## Table B | Training Time

Actual training time logged per run (`parameters/running_time` in each `log.json`), averaged over seeds. Times are full training runs with early stopping (patience 20, max 100 epochs). ARIMA is not gradient-trained and logs no `running_time` (`X`).

| Dataset | $L$ | $O$ | EMForecaster (2025) | DLinear (2023) | TSMixer (2023) | PatchTST (2023) | Transformer (2017) | LSTM (1997) | ARIMA (1970) |
|---|---|---|---|---|---|---|---|---|---|
| DET  | 192 | 96  | 28m 45s |  |  |  |  |  | X |
| DET  | 192 | 336 | 11m 13s |  |  |  |  |  | X |
| DET  | 336 | 96  | 32m 56s |  |  |  |  |  | X |
| DET  | 336 | 336 | 19m 55s |  |  |  |  |  | X |
| PTV  | 192 | 96  | 7m 23s  |  |  |  |  |  | X |
| PTV  | 192 | 336 | 7m 18s  |  |  |  |  |  | X |
| PTV  | 336 | 96  | 8m 28s  |  |  |  |  |  | X |
| PTV  | 336 | 336 | 8m 32s  |  |  |  |  |  | X |
| PTV2 | 192 | 96  | 17m 24s |  |  |  |  |  | X |
| PTV2 | 192 | 336 | 45m 46s |  |  |  |  |  | X |
| PTV2 | 336 | 96  | 21m 36s |  |  |  |  |  | X |
| PTV2 | 336 | 336 | 31m 0s  |  |  |  |  |  | X |

## Table C | Per-Epoch Training Time

Table B uses early stopping, so different runs stop at different epochs and a slower-per-step model can post a shorter total just because it stopped sooner. This table divides each run's logged `running_time` by the number of epochs it actually ran (`len(sl_train/loss)`) **per seed**, then averages over seeds, a better estimate of relative cost than Table B.

| Dataset | $L$ | $O$ | EMForecaster (2025) | DLinear (2023) | TSMixer (2023) | PatchTST (2023) | Transformer (2017) | LSTM (1997) | ARIMA (1970) |
|---|---|---|---|---|---|---|---|---|---|
| DET  | 192 | 96  | 18.5s |  |  |  |  |  | X |
| DET  | 192 | 336 | 10.3s |  |  |  |  |  | X |
| DET  | 336 | 96  | 26.4s |  |  |  |  |  | X |
| DET  | 336 | 336 | 17.3s |  |  |  |  |  | X |
| PTV  | 192 | 96  | 5.5s  |  |  |  |  |  | X |
| PTV  | 192 | 336 | 7.0s  |  |  |  |  |  | X |
| PTV  | 336 | 96  | 8.8s  |  |  |  |  |  | X |
| PTV  | 336 | 336 | 10.2s |  |  |  |  |  | X |
| PTV2 | 192 | 96  | 19.2s |  |  |  |  |  | X |
| PTV2 | 192 | 336 | 59.7s |  |  |  |  |  | X |
| PTV2 | 336 | 96  | 27.9s |  |  |  |  |  | X |
| PTV2 | 336 | 336 | 38.9s |  |  |  |  |  | X |

### Per-model summary

**Mean / epoch** divides each run's logged `parameters/running_time` by the epochs it actually ran (`len(sl_train/loss)`), then averages over every config and seed for the model. **Estimation for 100 ep** scales that mean to the 100-epoch early-stopping cap (`mean × 100`). Weights ranges and exact per-config parameter counts are CP-independent and listed once in [Model sizes](#model-sizes-shared-across-both-settings).

| Model | Weights | Mean / epoch | Estimation for 100 ep |
|---|---:|---:|---:|
| EMForecaster (2025) | 1.5M–7.6M | 19.92s | ~33m 12s |

## Notes

- **MSE:** EMForecaster's without-CP point forecasts beat the classical ARIMA baseline by a wide margin in every setting (roughly 3–4× lower MSE). Direct comparison against the other deep models is not possible here because they were only run with conformal prediction (see Section 2).
- **Training time:** Only EMForecaster has logged without-CP training runs; ARIMA is not gradient-trained and has no `running_time`. Some rows in Table B look out of order (e.g. EMForecaster's `O=336` faster than its `O=96`); this is an early-stopping artifact that the per-epoch normalization in Table C removes.
- **Seed aggregation:** EMForecaster DET rows average 5 seeds; PTV and PTV2 rows average 2 seeds (`69421`, `69422`).
- A 3-epoch smoke-test run (`1_EMForecaster_cptest_off`, PTV2 / $L$336 / $O$336) is excluded from all aggregates.

---

# 2 · Point-Forecasting Performance (with conformal prediction)

Runs with `conf.conf = true` at the significance level **$\alpha$ = 0.1 (90% target coverage)**. DLinear, TSMixer, PatchTST, Transformer, and LSTM have a full grid here. EMForecaster has no conformal run, and ARIMA has no conformal variant; both columns are blank.

## Table A | MSE

**Bold** = best (lowest) value in the row.

| Dataset | $L$ | $O$ | EMForecaster (2025) | DLinear (2023) | TSMixer (2023) | PatchTST (2023) | Transformer (2017) | LSTM (1997) | ARIMA (1970) |
|---|---|---|---|---|---|---|---|---|---|
| DET  | 192 | 96 |  | 0.4517 | 0.3648 | 0.4400 | 1.0383 | **0.3404** |  |
| DET  | 192 | 336 |  | 0.5303 | **0.4639** | 0.5440 | 1.1477 | 0.4949 |  |
| DET  | 336 | 96 |  | 0.3993 | 0.3551 | 0.3957 | 1.0601 | **0.3500** |  |
| DET  | 336 | 336 |  | 0.5048 | **0.4602** | 0.5111 | 1.4810 | 0.4834 |  |
| PTV  | 192 | 96 |  | 0.4710 | **0.3212** | 0.3772 | 1.0271 | 0.3501 |  |
| PTV  | 192 | 336 |  | 0.4747 | **0.3657** | 0.4538 | 1.3877 | 0.5307 |  |
| PTV  | 336 | 96 |  | 0.3545 | **0.3074** | 0.3263 | 1.0398 | 0.3514 |  |
| PTV  | 336 | 336 |  | 0.4256 | **0.3846** | 0.3876 | 1.1085 | 0.5473 |  |
| PTV2 | 192 | 96 |  | 0.4003 | 0.3348 | 0.3994 | 0.3746 | **0.3293** |  |
| PTV2 | 192 | 336 |  | 0.4130 | **0.3594** | 0.4486 | 0.8356 | 0.3626 |  |
| PTV2 | 336 | 96 |  | 0.3312 | **0.3038** | 0.3428 | 0.7625 | 0.3169 |  |
| PTV2 | 336 | 336 |  | 0.3745 | **0.3354** | 0.4141 | 0.8474 | 0.3541 |  |

## Table B | Training Time

Actual logged `parameters/running_time`, averaged over seeds (early stopping, patience 20, max 100 epochs). **Bold** = fastest in the row.

| Dataset | $L$ | $O$ | EMForecaster (2025) | DLinear (2023) | TSMixer (2023) | PatchTST (2023) | Transformer (2017) | LSTM (1997) | ARIMA (1970) |
|---|---|---|---|---|---|---|---|---|---|
| DET  | 192 | 96 |  | **43s** | 1m 23s | 2m 36s | 4m 22s | 2m 26s | X |
| DET  | 192 | 336 |  | **1m 58s** | 2m 20s | 2m 28s | 5m 22s | 2m 30s | X |
| DET  | 336 | 96 |  | **49s** | 1m 13s | 2m 26s | 8m 28s | 2m 23s | X |
| DET  | 336 | 336 |  | **2m 8s** | 2m 15s | 4m 28s | 8m 21s | 2m 30s | X |
| PTV  | 192 | 96 |  | **36s** | 44s | 1m 7s | 4m 17s | 1m 24s | X |
| PTV  | 192 | 336 |  | **1m 9s** | 1m 23s | 1m 24s | 1m 31s | 1m 32s | X |
| PTV  | 336 | 96 |  | **37s** | 53s | 2m 12s | 7m 16s | 1m 24s | X |
| PTV  | 336 | 336 |  | **1m 22s** | 1m 36s | 2m 53s | 5m 37s | 1m 33s | X |
| PTV2 | 192 | 96 |  | **1m 22s** | 2m 5s | 2m 58s | 9m 9s | 4m 40s | X |
| PTV2 | 192 | 336 |  | 3m 34s | 4m 8s | **1m 44s** | 12m 32s | 5m 32s | X |
| PTV2 | 336 | 96 |  | **1m 36s** | 3m 4s | 5m 44s | 15m 45s | 4m 49s | X |
| PTV2 | 336 | 336 |  | 3m 55s | 4m 43s | **2m 50s** | 15m 50s | 5m 36s | X |

## Table C | Per-Epoch Training Time

Logged `running_time` divided by epochs actually run (`len(sl_train/loss)`) per seed, then averaged over seeds. **Bold** = fastest in the row.

| Dataset | $L$ | $O$ | EMForecaster (2025) | DLinear (2023) | TSMixer (2023) | PatchTST (2023) | Transformer (2017) | LSTM (1997) | ARIMA (1970) |
|---|---|---|---|---|---|---|---|---|---|
| DET  | 192 | 96 |  | **0.4s** | 0.8s | 1.6s | 5.9s | 1.5s | X |
| DET  | 192 | 336 |  | **1.2s** | 1.4s | 1.7s | 8.6s | 1.5s | X |
| DET  | 336 | 96 |  | **0.5s** | 0.7s | 1.5s | 15.2s | 1.4s | X |
| DET  | 336 | 336 |  | **1.3s** | 1.3s | 2.7s | 19.6s | 1.5s | X |
| PTV  | 192 | 96 |  | **0.4s** | 0.4s | 0.7s | 4.0s | 0.8s | X |
| PTV  | 192 | 336 |  | **0.7s** | 0.8s | 1.0s | 6.3s | 0.9s | X |
| PTV  | 336 | 96 |  | **0.4s** | 0.5s | 1.3s | 11.0s | 0.8s | X |
| PTV  | 336 | 336 |  | **0.8s** | 1.0s | 1.7s | 12.6s | 0.9s | X |
| PTV2 | 192 | 96 |  | **0.8s** | 1.2s | 2.2s | 11.8s | 2.8s | X |
| PTV2 | 192 | 336 |  | **2.1s** | 2.5s | 2.3s | 17.4s | 3.3s | X |
| PTV2 | 336 | 96 |  | **1.0s** | 1.8s | 4.6s | 33.2s | 2.9s | X |
| PTV2 | 336 | 336 |  | **2.3s** | 2.8s | 4.0s | 40.9s | 3.4s | X |

### Per-model summary

| Model | Weights | Mean / epoch | Estimation for 100 ep |
|---|---:|---:|---:|
| DLinear (2023)     | 37k–226k  | **0.94s** | **~1m 34s** |
| TSMixer (2023)     | 31k–135k  | 1.22s     | ~2m 2s |
| LSTM (1997)        | 32k–58k   | 1.82s     | ~3m 2s |
| PatchTST (2023)    | 44k–234k  | 2.02s     | ~3m 22s |
| Transformer (2017) | 131k–456k | 15.54s    | ~25m 54s |

(EMForecaster has no conformal run; its without-CP grid averaged 19.92s/epoch, in Section 1.)

## Table D | Marginal Coverage (%)

Per-timestep empirical coverage of the conformal intervals (`sl_test/ic`). **Target = 90%** ($\alpha$=0.1); values above are conservative (wider than needed), values below indicate under-coverage. Not bolded, since "best" means closest to the 90% target rather than highest.

| Dataset | $L$ | $O$ | EMForecaster (2025) | DLinear (2023) | TSMixer (2023) | PatchTST (2023) | Transformer (2017) | LSTM (1997) | ARIMA (1970) |
|---|---|---|---|---|---|---|---|---|---|
| DET  | 192 | 96 |  | 91.71 | 91.33 | 91.60 | 92.38 | 88.79 |  |
| DET  | 192 | 336 |  | 91.71 | 91.70 | 92.01 | 90.20 | 89.21 |  |
| DET  | 336 | 96 |  | 91.63 | 91.21 | 91.20 | 89.86 | 87.51 |  |
| DET  | 336 | 336 |  | 91.62 | 91.68 | 91.64 | 92.24 | 88.81 |  |
| PTV  | 192 | 96 |  | 86.39 | 88.64 | 87.56 | 80.73 | 83.30 |  |
| PTV  | 192 | 336 |  | 84.63 | 86.39 | 86.02 | 81.64 | 81.15 |  |
| PTV  | 336 | 96 |  | 86.84 | 90.12 | 88.67 | 79.98 | 82.84 |  |
| PTV  | 336 | 336 |  | 84.89 | 87.30 | 86.14 | 82.97 | 80.74 |  |
| PTV2 | 192 | 96 |  | 86.00 | 85.92 | 86.56 | 90.45 | 89.85 |  |
| PTV2 | 192 | 336 |  | 84.78 | 84.12 | 88.75 | 90.88 | 89.67 |  |
| PTV2 | 336 | 96 |  | 86.19 | 85.96 | 87.15 | 88.04 | 90.12 |  |
| PTV2 | 336 | 336 |  | 84.63 | 83.97 | 89.23 | 90.67 | 89.71 |  |

## Table E | Joint Coverage (%)

Fraction of test windows where the interval simultaneously covers **all** forecast horizons (`sl_test/jc`). This is much stricter than marginal coverage and, without Bonferroni correction (`conf.corrected = false`), falls far below the nominal level, especially as $O$ grows.

| Dataset | $L$ | $O$ | EMForecaster (2025) | DLinear (2023) | TSMixer (2023) | PatchTST (2023) | Transformer (2017) | LSTM (1997) | ARIMA (1970) |
|---|---|---|---|---|---|---|---|---|---|
| DET  | 192 | 96 |  | 49.31 | 47.32 | 50.78 | 50.41 | 19.11 |  |
| DET  | 192 | 336 |  | 27.00 | 25.40 | 28.15 | 18.47 | 2.06 |  |
| DET  | 336 | 96 |  | 50.41 | 45.42 | 47.38 | 43.42 | 13.69 |  |
| DET  | 336 | 336 |  | 27.75 | 24.83 | 26.06 | 22.80 | 1.17 |  |
| PTV  | 192 | 96 |  | 27.17 | 31.94 | 29.31 | 28.43 | 7.38 |  |
| PTV  | 192 | 336 |  | 5.67 | 5.16 | 6.39 | 2.79 | 0.26 |  |
| PTV  | 336 | 96 |  | 23.18 | 38.09 | 24.44 | 29.18 | 7.48 |  |
| PTV  | 336 | 336 |  | 6.42 | 8.13 | 2.51 | 3.67 | 0.33 |  |
| PTV2 | 192 | 96 |  | 17.31 | 18.23 | 19.69 | 7.14 | 4.27 |  |
| PTV2 | 192 | 336 |  | 0.84 | 1.10 | 4.74 | 7.97 | 0.04 |  |
| PTV2 | 336 | 96 |  | 17.95 | 17.55 | 20.13 | 10.34 | 3.12 |  |
| PTV2 | 336 | 336 |  | 0.76 | 1.02 | 7.19 | 6.98 | 0.01 |  |

## Table F | Mean Interval Width

Average width of the conformal prediction intervals (`sl_test/interval_width_mean`), in the scaled target units. Narrower is better **only when coverage is met**, a narrow interval that under-covers (cross-reference Table D) is not an improvement.

| Dataset | $L$ | $O$ | EMForecaster (2025) | DLinear (2023) | TSMixer (2023) | PatchTST (2023) | Transformer (2017) | LSTM (1997) | ARIMA (1970) |
|---|---|---|---|---|---|---|---|---|---|
| DET  | 192 | 96 |  | 3.578 | 3.525 | 3.583 | 3.375 | 1.712 |  |
| DET  | 192 | 336 |  | 4.603 | 4.609 | 4.803 | 3.457 | 2.170 |  |
| DET  | 336 | 96 |  | 3.605 | 3.508 | 3.493 | 3.252 | 1.665 |  |
| DET  | 336 | 336 |  | 4.638 | 4.608 | 4.599 | 4.172 | 2.122 |  |
| PTV  | 192 | 96 |  | 2.977 | 3.144 | 3.099 | 2.575 | 1.573 |  |
| PTV  | 192 | 336 |  | 3.677 | 3.900 | 4.104 | 3.078 | 1.868 |  |
| PTV  | 336 | 96 |  | 2.937 | 3.240 | 3.024 | 2.525 | 1.548 |  |
| PTV  | 336 | 336 |  | 3.767 | 4.005 | 3.920 | 2.827 | 1.884 |  |
| PTV2 | 192 | 96 |  | 3.020 | 3.056 | 3.072 | 1.986 | 1.827 |  |
| PTV2 | 192 | 336 |  | 3.764 | 3.829 | 4.214 | 3.157 | 1.913 |  |
| PTV2 | 336 | 96 |  | 3.038 | 3.055 | 3.169 | 2.594 | 1.790 |  |
| PTV2 | 336 | 336 |  | 3.789 | 3.831 | 4.435 | 3.156 | 1.878 |  |

## Notes

- **MSE (point forecasts):** With conformal prediction, TSMixer is the strongest point forecaster, posting the lowest MSE in 9 of 12 settings; LSTM wins the remaining 3 (DET/$L$192/$O$96, DET/$L$336/$O$96, PTV2/$L$192/$O$96). The Transformer is the weakest by a wide margin on DET and PTV (≈1.0–1.5 MSE) and remains well behind on PTV2 except at $L$192/$O$96. Note these are *with-CP* numbers (the model trains on a reduced set after the calibration split) and are not directly comparable to EMForecaster's *without-CP* numbers in Section 1.
- **Coverage.** At $\alpha$=0.1 (90% target), marginal coverage lands near the nominal level on DET (≈91–92%, slightly conservative) but under-covers on PTV/PTV2 for several models (LSTM and Transformer dip to ≈80–83%).
- **Joint vs marginal.** Joint coverage (Table E) is far below marginal coverage and collapses as the horizon $O$ grows (often near 0% at $O$=336). This is the expected consequence of running per-step conformal calibration without the Bonferroni correction (`conf.corrected = false`); covering all horizons simultaneously requires the corrected critical scores.
- **Interval width tracks model scale and dataset.** LSTM produces the narrowest intervals everywhere, but also the lowest coverage, so its width advantage partly reflects under-coverage rather than sharper uncertainty.
- **Seed aggregation:** DLinear, TSMixer, and PatchTST average 5 seeds on DET and 2 seeds on PTV/PTV2. LSTM and Transformer average 2 seeds (`69421`, `69422`) in every row.

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
