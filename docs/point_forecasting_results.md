# Point-Forecasting Performance (without conformal prediction)

Test-set metrics, averaged over seeds.
Datasets: **DET** = `rf_emf_det`, **PTV** = `rf_emf_ptv`, **PTV2** = `rf_emf_ptv2`.
$L$ = input/lookback length (`seq_len`), $O$ = forecast horizon (`pred_len`).

**Bold** = best (lowest) value in the row. 
Values rounded to 4 decimals.

## Table A | MSE

| Dataset | $L$ | $O$ | EMForecaster (2025) | DLinear (2023) | TSMixer (2023) | PatchTST (2023) | Transformer (2017) | LSTM (1997) | ARIMA (1970) |
|---|---|---|---|---|---|---|---|---|---|
| DET  | 192 | 96  | **0.3111** | 0.4517 | 0.3648 | 0.4400 | 1.0383 | 0.3404 | 1.2371 |
| DET  | 192 | 336 | **0.4229** | 0.5303 | 0.4639 | 0.5440 | 1.1477 | 0.4949 | 1.4194 |
| DET  | 336 | 96  | **0.3009** | 0.3993 | 0.3551 | 0.3957 | 1.0601 | 0.3500 | 1.2371 |
| DET  | 336 | 336 | **0.3940** | 0.5048 | 0.4602 | 0.5111 | 1.4810 | 0.4834 | 1.4194 |
| PTV  | 192 | 96  | **0.2614** | 0.4710 | 0.3212 | 0.3772 | 1.0271 | 0.3501 | 1.1840 |
| PTV  | 192 | 336 | **0.3198** | 0.4747 | 0.3657 | 0.4538 | 1.3877 | 0.5307 | 1.3061 |
| PTV  | 336 | 96  | **0.2435** | 0.3545 | 0.3074 | 0.3263 | 1.0398 | 0.3514 | 1.1840 |
| PTV  | 336 | 336 | **0.3137** | 0.4256 | 0.3846 | 0.3876 | 1.1085 | 0.5473 | 1.3061 |
| PTV2 | 192 | 96  | **0.3220** | 0.4003 | 0.3348 | 0.3994 | 0.3746 | 0.3293 | 1.0128 |
| PTV2 | 192 | 336 | **0.3586** | 0.4130 | 0.3594 | 0.4486 | 0.8356 | 0.3626 | 1.1346 |
| PTV2 | 336 | 96  | **0.2999** | 0.3312 | 0.3038 | 0.3428 | 0.7625 | 0.3169 | 1.0128 |
| PTV2 | 336 | 336 | 0.3387 | 0.3745 | **0.3354** | 0.4141 | 0.8474 | 0.3541 | 1.1346 |

## Table B | Training Time

Actual training time logged per run (`parameters/running_time` in each `log.json`), averaged over seeds. Times are full training runs with early stopping (patience 20, max 100 epochs)

**Bold** = fastest (lowest) value in the row. ARIMA is not gradient-trained and logs no `running_time`, so it has no entry.

| Dataset | $L$ | $O$ | EMForecaster (2025) | DLinear (2023) | TSMixer (2023) | PatchTST (2023) | Transformer (2017) | LSTM (1997) | ARIMA (1970) |
|---|---|---|---|---|---|---|---|---|---|
| DET  | 192 | 96  | 28m 45s | **43s** | 1m 23s | 2m 36s | 4m 22s | 2m 26s | X |
| DET  | 192 | 336 | 11m 13s | **1m 58s** | 2m 20s | 2m 28s | 5m 22s | 2m 30s | X |
| DET  | 336 | 96  | 32m 56s | **49s** | 1m 13s | 2m 26s | 8m 28s | 2m 23s | X |
| DET  | 336 | 336 | 19m 55s | **2m 8s** | 2m 15s | 4m 28s | 8m 21s | 2m 30s | X |
| PTV  | 192 | 96  | 7m 23s | **36s** | 44s | 1m 7s | 4m 17s | 1m 24s | X |
| PTV  | 192 | 336 | 7m 18s | **1m 9s** | 1m 23s | 1m 24s | 1m 31s | 1m 32s | X |
| PTV  | 336 | 96  | 8m 28s | **37s** | 53s | 2m 12s | 7m 16s | 1m 24s | X |
| PTV  | 336 | 336 | 8m 32s | **1m 22s** | 1m 36s | 2m 53s | 5m 37s | 1m 33s | X |
| PTV2 | 192 | 96  | 17m 24s | **1m 22s** | 2m 5s | 2m 58s | 9m 9s | 4m 40s | X |
| PTV2 | 192 | 336 | 45m 46s | 3m 34s | 4m 8s | **1m 44s** | 12m 32s | 5m 32s | X |
| PTV2 | 336 | 96  | 21m 36s | **1m 36s** | 3m 4s | 5m 44s | 15m 45s | 4m 49s | X |
| PTV2 | 336 | 336 | 21m 4s | 3m 55s | 4m 43s | **2m 50s** | 15m 50s | 5m 36s | X |

## Table C | Per-Epoch Training Time

Table B is has early stopping: different runs stop at different epochs, so a slower-per-step model can post a shorter training time just because it stopped sooner. For a better estimate, this table divides each run's logged `running_time` by the number of epochs it actually ran, then averages over seeds. This is the training cost per epoch which is a better estimate of relative cost than Table B.

**Bold** = fastest (lowest) value in the row.

| Dataset | $L$ | $O$ | EMForecaster (2025) | DLinear (2023) | TSMixer (2023) | PatchTST (2023) | Transformer (2017) | LSTM (1997) | ARIMA (1970) |
|---|---|---|---|---|---|---|---|---|---|
| DET  | 192 | 96  | 18.5s | **0.4s** | 0.8s | 1.6s | 5.9s | 1.5s | X |
| DET  | 192 | 336 | 10.3s | **1.2s** | 1.4s | 1.7s | 8.6s | 1.5s | X |
| DET  | 336 | 96  | 26.4s | **0.5s** | 0.7s | 1.5s | 15.2s | 1.4s | X |
| DET  | 336 | 336 | 17.3s | **1.3s** | 1.3s | 2.7s | 19.6s | 1.5s | X |
| PTV  | 192 | 96  | 5.5s  | **0.4s** | 0.4s | 0.7s | 4.0s | 0.8s | X |
| PTV  | 192 | 336 | 7.0s  | **0.7s** | 0.8s | 1.0s | 6.3s | 0.9s | X |
| PTV  | 336 | 96  | 8.8s  | **0.4s** | 0.5s | 1.3s | 11.0s | 0.8s | X |
| PTV  | 336 | 336 | 10.2s | **0.8s** | 1.0s | 1.7s | 12.6s | 0.9s | X |
| PTV2 | 192 | 96  | 19.2s | **0.8s** | 1.2s | 2.2s | 11.8s | 2.8s | X |
| PTV2 | 192 | 336 | 59.7s | **2.1s** | 2.5s | 2.3s | 17.4s | 3.3s | X |
| PTV2 | 336 | 96  | 27.9s | **1.0s** | 1.8s | 4.6s | 33.2s | 2.9s | X |
| PTV2 | 336 | 336 | 34.0s | **2.3s** | 2.8s | 4.0s | 40.9s | 3.4s | X |

### Per-model summary
Weights come from `parameters/sl/num_params` in each log. The Weights column gives the range across the four $(L, O)$ configurations so parameter counts grow with both $L$ and $O$ and are independent of dataset. The exact per-config breakdown is below. The number of epochs can be found under the length of `sl_train/loss` for each log.

**Mean / epoch** divides each run's logged `parameters/running_time` by the epochs it actually ran (`len(sl_train/loss)`), then averages over every config and seed for the model. **Estimation for 100 ep** scales that mean to the 100-epoch early-stopping cap (`mean × 100`).

| Model | Weights | Mean / epoch | Estimation for 100 ep |
|---|---:|---:|---:|
| DLinear      | 37k–226k  | **0.94s** | **~1m 34s** |
| TSMixer      | 31k–135k  | 1.22s     | ~2m 2s |
| LSTM         | 32k–58k   | 1.82s     | ~3m 2s |
| PatchTST     | 44k–234k  | 2.02s     | ~3m 22s |
| Transformer  | 131k–456k | 15.54s    | ~25m 54s |
| EMForecaster | 1.5M–7.6M | 19.92s    | ~33m 12s |

Exact parameter counts per configuration (weights depend only on $L$ and $O$ and it doesn't depend on the dataset):

| Model | L192 / O96 | L192 / O336 | L336 / O96 | L336 / O336 |
|---|---:|---:|---:|---:|
| DLinear      | 37,056    | 129,696   | 64,704    | 226,464   |
| TSMixer      | 31,139    | 77,459    | 54,323    | 135,203   |
| LSTM         | 32,096    | 40,016    | 50,528    | 58,448    |
| PatchTST     | 44,194    | 136,594   | 72,130    | 233,650   |
| Transformer  | 130,528   | 179,248   | 372,736   | 456,016   |
| EMForecaster | 1,495,235 | 4,444,595 | 2,416,907 | 7,578,107 |

## Notes

- **MSE:** EMForecaster is best in 11 of the 12 settings. TSMixer beats it on PTV2 / L=336 / O=336 (0.3354 vs 0.3387). The Transformer is the weakest model by a margin, with around 2 to 4 times EMForecaster's MSE. It is even worse than the classical ARIMA baseline on DET / L=336 / O=336 (1.4810 vs 1.4194) and PTV / L=192 / O=336 (1.3877 vs 1.3061).
- **Reading the tables:** Table B is the actual logged training time per experiment (with early stopping). Table C is the per-epoch cost after correcting for early stopping, which is a better way to compare training cost.
- **Training time:** Per epoch (Table C), DLinear is the fastest in all 12 settings. The two largest models are the slowest: EMForecaster (1.5M to 7.6M weights) and the Transformer (131k to 456k weights), both of which grow with $L$ and $O$. EMForecaster is slowest in 7 of the 12 settings, and the Transformer is slowest in the other 5. Those 5 are all at $L=336$, where the Transformer's parameter count rises to 373k to 456k and passes EMForecaster's per-epoch cost. Some rows in Table B look out of order (for example, EMForecaster's `O=336` looks faster than its `O=96`). This is an early-stopping artifact, and the per-epoch normalization in Table C removes it.
- ARIMA is not trained with gradient descent, so it logs no per-epoch `running_time` and has no entry. It also does not use a learned lookback, so its L=192 and L=336 rows are the same for a given (dataset, O) in Table A.
- Seed aggregation: for EMForecaster, DLinear, PatchTST, and TSMixer, DET rows average 5 seeds and PTV/PTV2 rows average 2 seeds. LSTM and Transformer average 2 seeds (`69421`, `69422`) in every row.

---

# With CP
use similar values to the paper for the threshold and seem how CP behaves
