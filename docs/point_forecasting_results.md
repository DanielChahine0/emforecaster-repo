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
| DET  | 192 | 96  | 28m 45s | **43s** | 2m 36s | 1m 23s | 2m 26s | 4m 22s | X |
| DET  | 192 | 336 | 11m 13s | **1m 58s** | 2m 28s | 2m 20s | 2m 30s | 5m 22s | X |
| DET  | 336 | 96  | 32m 56s | **49s** | 2m 26s | 1m 13s | 2m 23s | 8m 28s | X |
| DET  | 336 | 336 | 19m 55s | **2m 8s** | 4m 28s | 2m 15s | 2m 30s | 8m 21s | X |
| PTV  | 192 | 96  | 7m 23s | **36s** | 1m 7s | 44s | 1m 24s | 4m 17s | X |
| PTV  | 192 | 336 | 7m 18s | **1m 9s** | 1m 24s | 1m 23s | 1m 32s | 1m 31s | X |
| PTV  | 336 | 96  | 8m 28s | **37s** | 2m 12s | 53s | 1m 24s | 7m 16s | X |
| PTV  | 336 | 336 | 8m 32s | **1m 22s** | 2m 53s | 1m 36s | 1m 33s | 5m 37s | X |
| PTV2 | 192 | 96  | 17m 24s | **1m 22s** | 2m 58s | 2m 5s | 4m 40s | 9m 9s | X |
| PTV2 | 192 | 336 | 45m 46s | 3m 34s | **1m 44s** | 4m 8s | 5m 32s | 12m 32s | X |
| PTV2 | 336 | 96  | 21m 36s | **1m 36s** | 5m 44s | 3m 4s | 4m 49s | 15m 45s | X |
| PTV2 | 336 | 336 | 21m 4s | 3m 55s | **2m 50s** | 4m 43s | 5m 36s | 15m 50s | X |

## Table C | Per-Epoch Training Time

Table B is has early stopping: different runs stop at different epochs, so a slower-per-step model can post a shorter training time just because it stopped sooner. For a better estimate, this table divides each run's logged `running_time` by the number of epochs it actually ran, then averages over seeds. This is the training cost per epoch which is a better estimate of relative cost than Table B.

**Bold** = fastest (lowest) value in the row.

| Dataset | $L$ | $O$ | EMForecaster | DLinear | PatchTST | TSMixer | LSTM | Transformer | ARIMA |
|---|---|---|---|---|---|---|---|---|---|
| DET  | 192 | 96  | 18.5s | **0.4s** | 1.6s | 0.8s | 1.5s | 5.9s | X |
| DET  | 192 | 336 | 10.3s | **1.2s** | 1.7s | 1.4s | 1.5s | 8.6s | X |
| DET  | 336 | 96  | 26.4s | **0.5s** | 1.5s | 0.7s | 1.4s | 15.2s | X |
| DET  | 336 | 336 | 17.3s | **1.3s** | 2.7s | 1.3s | 1.5s | 19.6s | X |
| PTV  | 192 | 96  | 5.5s  | **0.4s** | 0.7s | 0.4s | 0.8s | 4.0s | X |
| PTV  | 192 | 336 | 7.0s  | **0.7s** | 1.0s | 0.8s | 0.9s | 6.3s | X |
| PTV  | 336 | 96  | 8.8s  | **0.4s** | 1.3s | 0.5s | 0.8s | 11.0s | X |
| PTV  | 336 | 336 | 10.2s | **0.8s** | 1.7s | 1.0s | 0.9s | 12.6s | X |
| PTV2 | 192 | 96  | 19.2s | **0.8s** | 2.2s | 1.2s | 2.8s | 11.8s | X |
| PTV2 | 192 | 336 | 59.7s | **2.1s** | 2.3s | 2.5s | 3.3s | 17.4s | X |
| PTV2 | 336 | 96  | 27.9s | **1.0s** | 4.6s | 1.8s | 2.9s | 33.2s | X |
| PTV2 | 336 | 336 | 34.0s | **2.3s** | 4.0s | 2.8s | 3.4s | 40.9s | X |

### Per-model summary
Weights come from `parameters/sl/num_params` in each log. The Weights column gives the range across the four $(L, O)$ configurations so parameter counts grow with both $L$ and $O$ and are independent of dataset. The exact per-config breakdown is below. The number of epochs can be found under the length of `sl_train/loss` for each log.

**Mean / epoch** divides each run's logged `parameters/running_time` by the epochs it actually ran (`len(sl_train/loss)`), then averages over every config and seed for the model. **Estimation for 100 ep** scales that mean to the 100-epoch early-stopping cap (`mean × 100`):

```python
import json, re
from glob import glob
from statistics import mean

def to_seconds(t):  # "1hr 2min 3s" / "2min 49s" / "43s" -> seconds
    units = {"d": 86400, "hr": 3600, "min": 60, "s": 1}
    return sum(int(v) * units[u] for v, u in re.findall(r"(\d+)\s*(d|hr|min|s)", t))

# one value per run: every (L, O) config and seed for the model.
# deep models log as "<n>_<Model>_<id>"; the LSTM/Transformer baselines as
# "<n>_RF_EMF_<Model>_...", so match an optional RF_EMF_ prefix.
per_epoch = []
for path in glob("logs/*_<Model>_*/*/log.json") + glob("logs/*_RF_EMF_<Model>_*/*/log.json"):
    log = json.load(open(path))
    epochs = len(log["sl_train/loss"])                   # epochs actually run
    per_epoch.append(to_seconds(log["parameters/running_time"]) / epochs)

m = mean(per_epoch)
print(f"Mean / epoch:        {m:.2f}s")        # column 3
print(f"Estimation (100 ep): {m * 100:.0f}s")  # column 4 (100-epoch cap), shown as "m s"
```


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
