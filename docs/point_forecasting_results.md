# Point-Forecasting Performance (without conformal prediction)

Test-set metrics, averaged over seeds.
Datasets: **DET** = `rf_emf_det`, **PTV** = `rf_emf_ptv`, **PTV2** = `rf_emf_ptv2`.
$L$ = input/lookback length (`seq_len`), $O$ = forecast horizon (`pred_len`).

**Bold** = best (lowest) value in the row. 
Values rounded to 4 decimals.

## Table A | MSE

| Dataset | $L$ | $O$ | EMForecaster | DLinear | PatchTST | TSMixer | LSTM | Transformer | ARIMA |
|---|---|---|---|---|---|---|---|---|---|
| DET  | 192 | 96  | **0.3111** | 0.4517 | 0.4400 | 0.3648 | 0.3404 | 1.0383 | 1.2371 |
| DET  | 192 | 336 | **0.4229** | 0.5303 | 0.5440 | 0.4639 | 0.4949 | 1.1477 | 1.4194 |
| DET  | 336 | 96  | **0.3009** | 0.3993 | 0.3957 | 0.3551 | 0.3500 | 1.0601 | 1.2371 |
| DET  | 336 | 336 | **0.3940** | 0.5048 | 0.5111 | 0.4602 | 0.4834 | 1.4810 | 1.4194 |
| PTV  | 192 | 96  | **0.2614** | 0.4710 | 0.3772 | 0.3212 | 0.3501 | 1.0271 | 1.1840 |
| PTV  | 192 | 336 | **0.3198** | 0.4747 | 0.4538 | 0.3657 | 0.5307 | 1.3877 | 1.3061 |
| PTV  | 336 | 96  | **0.2435** | 0.3545 | 0.3263 | 0.3074 | 0.3514 | 1.0398 | 1.1840 |
| PTV  | 336 | 336 | **0.3137** | 0.4256 | 0.3876 | 0.3846 | 0.5473 | 1.1085 | 1.3061 |
| PTV2 | 192 | 96  | **0.3220** | 0.4003 | 0.3994 | 0.3348 | 0.3293 | 0.3746 | 1.0128 |
| PTV2 | 192 | 336 | **0.3586** | 0.4130 | 0.4486 | 0.3594 | 0.3626 | 0.8356 | 1.1346 |
| PTV2 | 336 | 96  | **0.2999** | 0.3312 | 0.3428 | 0.3038 | 0.3169 | 0.7625 | 1.0128 |
| PTV2 | 336 | 336 | 0.3387 | 0.3745 | 0.4141 | **0.3354** | 0.3541 | 0.8474 | 1.1346 |

## Table B | Training Time

Actual training time logged per run (`parameters/running_time` in each `log.json`), averaged over seeds. Times are full training runs with early stopping (patience 20, max 100 epochs)

**Bold** = fastest (lowest) value in the row. ARIMA is not gradient-trained and logs no `running_time`, so it has no entry.

| Dataset | $L$ | $O$ | EMForecaster | DLinear | PatchTST | TSMixer | LSTM | Transformer | ARIMA |
|---|---|---|---|---|---|---|---|---|---|
| DET  | 192 | 96  | 28m 45s | **43s** | 2m 36s | 1m 23s | 2m 26s | 4m 22s | — |
| DET  | 192 | 336 | 11m 13s | **1m 58s** | 2m 28s | 2m 20s | 2m 30s | 5m 22s | — |
| DET  | 336 | 96  | 32m 56s | **49s** | 2m 26s | 1m 13s | 2m 23s | 8m 28s | — |
| DET  | 336 | 336 | 19m 55s | **2m 8s** | 4m 28s | 2m 15s | 2m 30s | 8m 21s | — |
| PTV  | 192 | 96  | 7m 23s | **36s** | 1m 7s | 44s | 1m 24s | 4m 17s | — |
| PTV  | 192 | 336 | 7m 18s | **1m 9s** | 1m 24s | 1m 23s | 1m 32s | 1m 31s | — |
| PTV  | 336 | 96  | 8m 28s | **37s** | 2m 12s | 53s | 1m 24s | 7m 16s | — |
| PTV  | 336 | 336 | 8m 32s | **1m 22s** | 2m 53s | 1m 36s | 1m 33s | 5m 37s | — |
| PTV2 | 192 | 96  | 17m 24s | **1m 22s** | 2m 58s | 2m 5s | 4m 40s | 9m 9s | — |
| PTV2 | 192 | 336 | 45m 46s | 3m 34s | **1m 44s** | 4m 8s | 5m 32s | 12m 32s | — |
| PTV2 | 336 | 96  | 21m 36s | **1m 36s** | 5m 44s | 3m 4s | 4m 49s | 15m 45s | — |
| PTV2 | 336 | 336 | 21m 4s | 3m 55s | **2m 50s** | 4m 43s | 5m 36s | 15m 50s | — |

## Table C | Per-Epoch Training Time (early-stopping–normalized)

Table B is has early stopping: different runs stop at different epochs, so a slower-per-step model can post a shorter training time just because it stopped sooner. For a better estimate, this table divides each run's logged `running_time` by the **number of epochs it actually ran**, then averages over seeds. This is the training cost per epoch which is a better estimate of relative cost than Table B.

**Bold** = fastest (lowest) value in the row.

| Dataset | $L$ | $O$ | EMForecaster | DLinear | PatchTST | TSMixer | LSTM | Transformer | ARIMA |
|---|---|---|---|---|---|---|---|---|---|
| DET  | 192 | 96  | 18.5s | **0.4s** | 1.6s | 0.8s | 1.5s | 5.9s | — |
| DET  | 192 | 336 | 10.3s | **1.2s** | 1.7s | 1.4s | 1.5s | 8.6s | — |
| DET  | 336 | 96  | 26.4s | **0.5s** | 1.5s | 0.7s | 1.4s | 15.2s | — |
| DET  | 336 | 336 | 17.3s | **1.3s** | 2.7s | 1.3s | 1.5s | 19.6s | — |
| PTV  | 192 | 96  | 5.5s  | **0.4s** | 0.7s | 0.4s | 0.8s | 4.0s | — |
| PTV  | 192 | 336 | 7.0s  | **0.7s** | 1.0s | 0.8s | 0.9s | 6.3s | — |
| PTV  | 336 | 96  | 8.8s  | **0.4s** | 1.3s | 0.5s | 0.8s | 11.0s | — |
| PTV  | 336 | 336 | 10.2s | **0.8s** | 1.7s | 1.0s | 0.9s | 12.6s | — |
| PTV2 | 192 | 96  | 19.2s | **0.8s** | 2.2s | 1.2s | 2.8s | 11.8s | — |
| PTV2 | 192 | 336 | 59.7s | **2.1s** | 2.3s | 2.5s | 3.3s | 17.4s | — |
| PTV2 | 336 | 96  | 27.9s | **1.0s** | 4.6s | 1.8s | 2.9s | 33.2s | — |
| PTV2 | 336 | 336 | 34.0s | **2.3s** | 4.0s | 2.8s | 3.4s | 40.9s | — |

### Per-model summary
Weights come from `parameters/sl/num_params` in each log. Median per-epoch time across all configs/seeds (median is robust to the early-epoch outliers), with a full-seed estimate projected to the 100-epoch cap. Weights are reported at $L=336$, $O=96$ for every model (parameter counts grow with $L$ and $O$); the Transformer ranges 130k–456k across configs.

| Model | Weights | Median /epoch | Est. per seed (100 ep) |
|---|---:|---:|---:|
| DLinear      | 64.7k  | **0.8s** | **~1m 20s** |
| TSMixer      | 54.3k  | 1.1s     | ~1m 50s |
| LSTM         | 50.5k  | 1.5s     | ~2m 30s |
| PatchTST     | 72.1k  | 1.6s     | ~2m 40s |
| Transformer  | 372.7k | 11.8s    | ~19m 40s |
| EMForecaster | 2.42M  | 16.1s    | ~26m 50s |

## Notes

- **MSE:** EMForecaster is best in 11 of 12 settings; TSMixer edges it out on PTV2 / L=336 / O=336 (0.3354 vs 0.3387). The Transformer backbone is the weakest deep model by a wide margin — roughly 2–4× EMForecaster's MSE — and even trails the classical ARIMA baseline on DET / L=336 / O=336 (1.4810 vs 1.4194) and PTV / L=192 / O=336 (1.3877 vs 1.3061).
- **Reading the tables:** Table B = *actual* logged wall-clock per experiment (early-stopped). Table C = per-epoch cost normalized for early stopping — the better apples-to-apples estimate of training cost. The laptop micro-benchmark in [`model_explanations/41-runtime-comparison.md`](model_explanations/41-runtime-comparison.md) gives a controlled per-component breakdown.
- **Training time:** Normalized per epoch (Table C), DLinear is fastest in every one of the 12 settings. The two heavyweights — EMForecaster (2.42M weights) and the Transformer (130k–456k, growing with $L$) — are the slowest: EMForecaster is slowest in 7 of 12 settings and the Transformer in the other 5, all at $L=336$, where the Transformer's parameter count climbs to 373k–456k and overtakes EMForecaster per-epoch. The non-monotonic ranks in Table B (e.g. EMForecaster's `O=336` appearing faster than `O=96`) are an early-stopping artifact that this normalization removes.
- ARIMA is not gradient-trained, so it logs no per-epoch `running_time` and has no entry. It also does not use a learned lookback, so its L=192 and L=336 rows are identical for a given (dataset, O) in Table A.
- Seed aggregation: for EMForecaster, DLinear, PatchTST, and TSMixer, DET rows average 5 seeds and PTV/PTV2 rows average 2 seeds; LSTM and Transformer average 2 seeds (`69421`, `69422`) in every row.
