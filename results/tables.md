# Univariate Series

- Models: EMForecaster (2025), DLinear (2023), TSMixer (2023), PatchTST (2023), Transformer (2017)
- Datasets: DET, PTV, PTV2
- Grid: L192/O96, L192/O336, L336/O96, L336/O336
- Runs: 80 logs; each cell averages **1-3** replicate run(s)
- Conformal prediction: on for every run

## 1 · Forecast accuracy

**Bold** = best (lowest) in the row, <u>underline</u> = runner-up. `-` = not logged.

### 1.1 · MSE (`sl_test/loss`)

| Dataset | $L$ | $O$ | EMForecaster (2025) | DLinear (2023) | TSMixer (2023) | PatchTST (2023) | Transformer (2017) |
|:---|---:|---:|---:|---:|---:|---:|---:|
| DET | 192 | 96 | **0.3135** | 0.4520 | <u>0.3625</u> | 0.3988 | 0.5986 |
| DET | 192 | 336 | **0.4261** | 0.5301 | <u>0.4602</u> | 0.5246 | 0.9783 |
| DET | 336 | 96 | **0.3024** | 0.3993 | <u>0.3561</u> | 0.3951 | 1.0484 |
| DET | 336 | 336 | **0.3950** | 0.5044 | <u>0.4585</u> | 0.5129 | 1.4857 |
| PTV | 192 | 96 | **0.2610** | 0.4703 | <u>0.3135</u> | 0.3564 | 0.6426 |
| PTV | 192 | 336 | **0.3224** | 0.4739 | <u>0.3672</u> | 0.4157 | 1.3925 |
| PTV | 336 | 96 | **0.2404** | 0.3539 | <u>0.3075</u> | 0.3283 | 1.0292 |
| PTV | 336 | 336 | **0.3181** | 0.4252 | <u>0.3838</u> | 0.3909 | 0.8120 |
| PTV2 | 192 | 96 | **0.3199** | 0.4003 | <u>0.3339</u> | 0.3658 | 0.3819 |
| PTV2 | 192 | 336 | **0.3574** | 0.4130 | <u>0.3601</u> | 0.4146 | 1.1660 |
| PTV2 | 336 | 96 | **0.3002** | 0.3312 | <u>0.3038</u> | 0.3367 | 0.3542 |
| PTV2 | 336 | 336 | <u>0.3375</u> | 0.3745 | **0.3348** | 0.4380 | 0.4592 |

### 1.2 · MAE (`sl_test/mae`)

| Dataset | $L$ | $O$ | EMForecaster (2025) | DLinear (2023) | TSMixer (2023) | PatchTST (2023) | Transformer (2017) |
|:---|---:|---:|---:|---:|---:|---:|---:|
| DET | 192 | 96 | **0.3915** | 0.4961 | <u>0.4253</u> | 0.4527 | 0.5937 |
| DET | 192 | 336 | **0.4654** | 0.5318 | <u>0.4851</u> | 0.5232 | 0.7618 |
| DET | 336 | 96 | **0.3791** | 0.4449 | <u>0.4230</u> | 0.4537 | 0.7931 |
| DET | 336 | 336 | **0.4436** | 0.5106 | <u>0.4849</u> | 0.5151 | 1.0062 |
| PTV | 192 | 96 | **0.3740** | 0.5178 | <u>0.4036</u> | 0.4441 | 0.5832 |
| PTV | 192 | 336 | **0.4201** | 0.5069 | <u>0.4312</u> | 0.4723 | 0.9897 |
| PTV | 336 | 96 | **0.3569** | 0.4274 | <u>0.3947</u> | 0.4263 | 0.7943 |
| PTV | 336 | 336 | **0.4151** | 0.4760 | <u>0.4387</u> | 0.4542 | 0.7102 |
| PTV2 | 192 | 96 | **0.4372** | 0.4945 | <u>0.4449</u> | 0.4672 | 0.4871 |
| PTV2 | 192 | 336 | <u>0.4678</u> | 0.5030 | **0.4651** | 0.4952 | 0.8950 |
| PTV2 | 336 | 96 | <u>0.4243</u> | 0.4434 | **0.4225** | 0.4467 | 0.4687 |
| PTV2 | 336 | 336 | <u>0.4542</u> | 0.4755 | **0.4483** | 0.5092 | 0.5352 |

## 2 · Training cost

**Bold** = fastest in the row, <u>underline</u> = runner-up.

### 2.1 · Total training time (`parameters/running_time`)

| Dataset | $L$ | $O$ | EMForecaster (2025) | DLinear (2023) | TSMixer (2023) | PatchTST (2023) | Transformer (2017) |
|:---|---:|---:|---:|---:|---:|---:|---:|
| DET | 192 | 96 | 1m 19s | **28s** | 38s | 41s | <u>35s</u> |
| DET | 192 | 336 | 54s | **30s** | 40s | 42s | <u>35s</u> |
| DET | 336 | 96 | 1m 14s | **33s** | <u>36s</u> | 47s | 1m 6s |
| DET | 336 | 336 | 1m 12s | **31s** | <u>40s</u> | 48s | 1m 11s |
| PTV | 192 | 96 | 46s | **27s** | <u>35s</u> | 38s | 52s |
| PTV | 192 | 336 | 36s | **28s** | <u>35s</u> | 38s | 37s |
| PTV | 336 | 96 | 40s | **28s** | <u>32s</u> | 40s | 46s |
| PTV | 336 | 336 | 41s | **28s** | <u>36s</u> | 41s | 1m 0s |
| PTV2 | 192 | 96 | 1m 3s | **41s** | <u>54s</u> | 1m 10s | 1m 9s |
| PTV2 | 192 | 336 | 1m 9s | <u>42s</u> | 55s | **38s** | 1m 46s |
| PTV2 | 336 | 96 | 1m 35s | **43s** | 53s | <u>52s</u> | 1m 36s |
| PTV2 | 336 | 336 | 3m 37s | <u>44s</u> | 57s | **31s** | 2m 2s |

### 2.2 · Per-epoch training time (`running_time / len(sl_train/loss)`)

| Dataset | $L$ | $O$ | EMForecaster (2025) | DLinear (2023) | TSMixer (2023) | PatchTST (2023) | Transformer (2017) |
|:---|---:|---:|---:|---:|---:|---:|---:|
| DET | 192 | 96 | 0.8s | **0.3s** | <u>0.4s</u> | 0.4s | 0.7s |
| DET | 192 | 336 | 0.8s | **0.3s** | <u>0.4s</u> | 0.4s | 0.8s |
| DET | 336 | 96 | 0.9s | **0.3s** | <u>0.4s</u> | 0.5s | 0.9s |
| DET | 336 | 336 | 1.2s | **0.3s** | <u>0.4s</u> | 0.5s | 1.4s |
| PTV | 192 | 96 | 0.6s | **0.3s** | <u>0.3s</u> | 0.4s | 0.5s |
| PTV | 192 | 336 | 0.6s | **0.3s** | <u>0.3s</u> | 0.4s | 0.6s |
| PTV | 336 | 96 | 0.7s | **0.3s** | <u>0.3s</u> | 0.4s | 0.7s |
| PTV | 336 | 336 | 0.9s | **0.3s** | <u>0.4s</u> | 0.4s | 1.0s |
| PTV2 | 192 | 96 | 1.2s | **0.4s** | <u>0.5s</u> | 0.7s | 1.1s |
| PTV2 | 192 | 336 | 1.3s | **0.4s** | <u>0.6s</u> | 0.7s | 1.4s |
| PTV2 | 336 | 96 | 2.2s | **0.4s** | <u>0.5s</u> | 0.7s | 1.7s |
| PTV2 | 336 | 336 | 4.8s | **0.4s** | <u>0.6s</u> | 0.8s | 2.6s |

### 2.3 · Mean epoch cost by model

Pooled over every run of that model in this section.

| Model | Mean / epoch | Estimation for 100 ep | Runs |
|:---|---:|---:|---:|
| DLinear (2023) | 0.3s | ~34s | 18 |
| TSMixer (2023) | 0.4s | ~42s | 15 |
| PatchTST (2023) | 0.5s | ~51s | 17 |
| Transformer (2017) | 1.1s | ~1m 52s | 15 |
| EMForecaster (2025) | 1.3s | ~2m 12s | 15 |

## 3 · Conformal prediction

Not marked -- "best" coverage means closest to the target, not highest, and a narrower interval only counts if that target is met.

### 3.1 · Marginal coverage % (`sl_test/ic`)

| Dataset | $L$ | $O$ | EMForecaster (2025) | DLinear (2023) | TSMixer (2023) | PatchTST (2023) | Transformer (2017) |
|:---|---:|---:|---:|---:|---:|---:|---:|
| DET | 192 | 96 | 88.93 | 91.12 | 90.18 | 90.35 | 90.38 |
| DET | 192 | 336 | 88.38 | 90.57 | 89.47 | 89.57 | 88.46 |
| DET | 336 | 96 | 87.70 | 90.67 | 89.40 | 89.76 | 89.95 |
| DET | 336 | 336 | 87.93 | 90.54 | 89.17 | 90.17 | 91.92 |
| PTV | 192 | 96 | 87.06 | 87.19 | 90.71 | 89.43 | 78.75 |
| PTV | 192 | 336 | 85.14 | 86.38 | 88.21 | 87.13 | 81.54 |
| PTV | 336 | 96 | 88.23 | 88.31 | 92.95 | 89.78 | 80.55 |
| PTV | 336 | 336 | 86.06 | 86.05 | 89.45 | 86.90 | 83.68 |
| PTV2 | 192 | 96 | 90.43 | 89.48 | 90.76 | 89.76 | 89.98 |
| PTV2 | 192 | 336 | 90.06 | 89.97 | 90.45 | 90.28 | 84.27 |
| PTV2 | 336 | 96 | 90.89 | 91.06 | 91.38 | 91.36 | 90.36 |
| PTV2 | 336 | 336 | 90.73 | 90.57 | 90.99 | 90.36 | 88.04 |

### 3.2 · Joint coverage % (`sl_test/jc`)

| Dataset | $L$ | $O$ | EMForecaster (2025) | DLinear (2023) | TSMixer (2023) | PatchTST (2023) | Transformer (2017) |
|:---|---:|---:|---:|---:|---:|---:|---:|
| DET | 192 | 96 | 19.04 | 39.10 | 22.51 | 29.88 | 31.91 |
| DET | 192 | 336 | 3.16 | 14.84 | 7.06 | 8.22 | 12.18 |
| DET | 336 | 96 | 10.46 | 43.04 | 17.30 | 34.07 | 43.80 |
| DET | 336 | 336 | 1.83 | 12.63 | 5.26 | 15.19 | 22.31 |
| PTV | 192 | 96 | 5.15 | 14.33 | 9.74 | 13.55 | 22.78 |
| PTV | 192 | 336 | 0.02 | 0.59 | 0.86 | 0.39 | 3.54 |
| PTV | 336 | 96 | 4.74 | 9.68 | 26.47 | 15.57 | 28.79 |
| PTV | 336 | 336 | 0.00 | 0.00 | 3.13 | 1.29 | 2.60 |
| PTV2 | 192 | 96 | 4.77 | 5.99 | 5.73 | 5.30 | 7.38 |
| PTV2 | 192 | 336 | 0.00 | 0.00 | 0.17 | 0.36 | 0.00 |
| PTV2 | 336 | 96 | 3.78 | 5.82 | 4.83 | 6.18 | 5.72 |
| PTV2 | 336 | 336 | 0.00 | 0.00 | 0.01 | 0.25 | 0.05 |

### 3.3 · Mean interval width (`sl_test/interval_width_mean`)

| Dataset | $L$ | $O$ | EMForecaster (2025) | DLinear (2023) | TSMixer (2023) | PatchTST (2023) | Transformer (2017) |
|:---|---:|---:|---:|---:|---:|---:|---:|
| DET | 192 | 96 | 1.570 | 2.190 | 1.888 | 2.023 | 2.407 |
| DET | 192 | 336 | 1.930 | 2.429 | 2.190 | 2.376 | 3.027 |
| DET | 336 | 96 | 1.452 | 2.097 | 1.790 | 1.969 | 3.223 |
| DET | 336 | 336 | 1.817 | 2.403 | 2.154 | 2.425 | 4.162 |
| PTV | 192 | 96 | 1.408 | 1.949 | 1.759 | 1.817 | 1.945 |
| PTV | 192 | 336 | 1.524 | 1.895 | 1.760 | 1.836 | 3.077 |
| PTV | 336 | 96 | 1.374 | 1.718 | 1.922 | 1.725 | 2.572 |
| PTV | 336 | 336 | 1.534 | 1.778 | 1.909 | 1.782 | 2.499 |
| PTV2 | 192 | 96 | 1.840 | 2.021 | 1.915 | 1.941 | 1.977 |
| PTV2 | 192 | 336 | 1.930 | 2.082 | 1.974 | 2.125 | 3.158 |
| PTV2 | 336 | 96 | 1.791 | 1.908 | 1.848 | 1.945 | 1.909 |
| PTV2 | 336 | 336 | 1.893 | 2.010 | 1.930 | 2.179 | 2.066 |

## 4 · Model size

### 4.1 · Trainable weights (`parameters/sl/num_params`)

For multivariate logs this varies by dataset (different channel counts), not just by (model, L, O); shown here is the first value found per (model, L, O).

| Model | L192/O96 | L192/O336 | L336/O96 | L336/O336 |
|:---|---:|---:|---:|---:|
| EMForecaster (2025) | 1,495,235 | 4,444,595 | 2,416,907 | 7,578,107 |
| DLinear (2023) | 37,056 | 129,696 | 64,704 | 226,464 |
| TSMixer (2023) | 31,139 | 77,459 | 54,323 | 135,203 |
| PatchTST (2023) | 44,194 | 136,594 | 72,130 | 233,650 |
| Transformer (2017) | 130,528 | 179,248 | 372,736 | 456,016 |

# Univariate (Per Channel)

- Models: EMForecaster (2025), DLinear (2023), TSMixer (2023), PatchTST (2023), Transformer (2017), LSTM (1997), ARIMA (1970)
- Datasets: ED2022, UH2022, EMF2023 (University-5 w/ PTV)
- Grid: L336/O96
- Runs: 581 logs; each cell is a mean over **27-29** single-channel runs (1 seed each)
- Conformal prediction: on for every run

## 1 · Forecast accuracy

**Bold** = best (lowest) in the row, <u>underline</u> = runner-up. `-` = not logged.

### 1.1 · MSE (`sl_test/loss`)

| Dataset | $L$ | $O$ | EMForecaster (2025) | DLinear (2023) | TSMixer (2023) | PatchTST (2023) | Transformer (2017) | LSTM (1997) | ARIMA (1970) |
|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ED2022 | 336 | 96 | **0.4848** | <u>0.5321</u> | 0.5484 | 0.5419 | 0.6904 | 0.5785 | 1.0194 |
| UH2022 | 336 | 96 | **1.1704** | <u>1.1900</u> | 1.2932 | 1.2623 | 1.2270 | 1.2180 | 1.6017 |
| EMF2023 (University-5 w/ PTV) | 336 | 96 | **1.5054** | <u>1.5163</u> | 1.5262 | 1.5266 | 1.5808 | 1.5435 | 1.8089 |

### 1.2 · MAE (`sl_test/mae`)

| Dataset | $L$ | $O$ | EMForecaster (2025) | DLinear (2023) | TSMixer (2023) | PatchTST (2023) | Transformer (2017) | LSTM (1997) | ARIMA (1970) |
|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ED2022 | 336 | 96 | **0.3428** | <u>0.3781</u> | 0.3875 | 0.3817 | 0.4429 | 0.4160 | 0.6076 |
| UH2022 | 336 | 96 | **0.4585** | <u>0.4851</u> | 0.4961 | 0.4988 | 0.5075 | 0.5089 | 0.6807 |
| EMF2023 (University-5 w/ PTV) | 336 | 96 | **0.4500** | 0.4677 | <u>0.4591</u> | 0.4701 | 0.5006 | 0.4795 | 0.6212 |

## 2 · Training cost

**Bold** = fastest in the row, <u>underline</u> = runner-up. `-` = not logged (ARIMA is not gradient-trained).

### 2.1 · Total training time (`parameters/running_time`)

| Dataset | $L$ | $O$ | EMForecaster (2025) | DLinear (2023) | TSMixer (2023) | PatchTST (2023) | Transformer (2017) | LSTM (1997) | ARIMA (1970) |
|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ED2022 | 336 | 96 | 1m 46s | 56s | 1m 11s | **48s** | 3m 58s | <u>50s</u> | - |
| UH2022 | 336 | 96 | 1m 53s | 1m 11s | 1m 29s | **55s** | 5m 1s | <u>56s</u> | - |
| EMF2023 (University-5 w/ PTV) | 336 | 96 | 3m 38s | 1m 40s | 2m 39s | <u>1m 35s</u> | 8m 55s | **1m 30s** | - |

### 2.2 · Per-epoch training time (`running_time / len(sl_train/loss)`)

| Dataset | $L$ | $O$ | EMForecaster (2025) | DLinear (2023) | TSMixer (2023) | PatchTST (2023) | Transformer (2017) | LSTM (1997) | ARIMA (1970) |
|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ED2022 | 336 | 96 | 2.1s | 0.6s | 0.7s | <u>0.5s</u> | 5.0s | **0.5s** | - |
| UH2022 | 336 | 96 | 2.6s | 0.7s | 0.9s | <u>0.6s</u> | 6.7s | **0.6s** | - |
| EMF2023 (University-5 w/ PTV) | 336 | 96 | 5.1s | 1.2s | 1.6s | <u>1.1s</u> | 12.1s | **0.9s** | - |

### 2.3 · Mean epoch cost by model

Pooled over every run of that model in this section.

| Model | Mean / epoch | Estimation for 100 ep | Runs |
|:---|---:|---:|---:|
| LSTM (1997) | 0.7s | ~1m 7s | 83 |
| PatchTST (2023) | 0.7s | ~1m 14s | 83 |
| DLinear (2023) | 0.9s | ~1m 28s | 83 |
| TSMixer (2023) | 1.1s | ~1m 48s | 83 |
| EMForecaster (2025) | 3.3s | ~5m 28s | 83 |
| Transformer (2017) | 8.0s | ~13m 24s | 83 |

## 3 · Conformal prediction

Not marked -- "best" coverage means closest to the target, not highest, and a narrower interval only counts if that target is met.

### 3.1 · Marginal coverage % (`sl_test/ic`)

| Dataset | $L$ | $O$ | EMForecaster (2025) | DLinear (2023) | TSMixer (2023) | PatchTST (2023) | Transformer (2017) | LSTM (1997) | ARIMA (1970) |
|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ED2022 | 336 | 96 | 84.94 | 84.64 | 85.59 | 85.05 | 84.46 | 85.11 | 86.43 |
| UH2022 | 336 | 96 | 86.71 | 86.24 | 86.70 | 85.97 | 85.78 | 85.82 | 84.28 |
| EMF2023 (University-5 w/ PTV) | 336 | 96 | 90.46 | 90.76 | 91.33 | 90.66 | 90.57 | 90.05 | 88.65 |

### 3.2 · Joint coverage % (`sl_test/jc`)

| Dataset | $L$ | $O$ | EMForecaster (2025) | DLinear (2023) | TSMixer (2023) | PatchTST (2023) | Transformer (2017) | LSTM (1997) | ARIMA (1970) |
|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ED2022 | 336 | 96 | 10.50 | 11.10 | 12.99 | 12.89 | 14.80 | 14.14 | 21.65 |
| UH2022 | 336 | 96 | 6.87 | 6.60 | 7.59 | 8.91 | 7.92 | 8.64 | 9.66 |
| EMF2023 (University-5 w/ PTV) | 336 | 96 | 11.50 | 11.87 | 14.82 | 13.64 | 14.61 | 12.02 | 13.06 |

### 3.3 · Mean interval width (`sl_test/interval_width_mean`)

| Dataset | $L$ | $O$ | EMForecaster (2025) | DLinear (2023) | TSMixer (2023) | PatchTST (2023) | Transformer (2017) | LSTM (1997) | ARIMA (1970) |
|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ED2022 | 336 | 96 | 1.314 | 1.434 | 1.516 | 1.451 | 1.678 | 1.582 | 2.460 |
| UH2022 | 336 | 96 | 1.683 | 1.710 | 1.793 | 1.774 | 1.845 | 1.812 | 2.334 |
| EMF2023 (University-5 w/ PTV) | 336 | 96 | 1.941 | 1.935 | 2.202 | 2.001 | 2.112 | 1.996 | 2.446 |

## 4 · Model size

### 4.1 · Trainable weights (`parameters/sl/num_params`)

For multivariate logs this varies by dataset (different channel counts), not just by (model, L, O); shown here is the first value found per (model, L, O).

| Model | L336/O96 |
|:---|---:|
| EMForecaster (2025) | 2,416,907 |
| DLinear (2023) | 64,704 |
| TSMixer (2023) | 54,323 |
| PatchTST (2023) | 72,130 |
| Transformer (2017) | 372,738 |
| LSTM (1997) | 20,674 |

# Multivariate

- Models: EMForecaster (2025), DLinear (2023), TSMixer (2023), Transformer (2017), LSTM (1997), ARIMA (1970)
- Datasets: ED2022, UH2022, EMF2023 (University-5 w/ PTV)
- Grid: L336/O96
- Runs: 18 logs; each cell averages **1** replicate run(s)
- Conformal prediction: on for every run

## 1 · Forecast accuracy

**Bold** = best (lowest) in the row, <u>underline</u> = runner-up. `-` = not logged.

### 1.1 · MSE (`sl_test/loss`)

| Dataset | $L$ | $O$ | EMForecaster (2025) | DLinear (2023) | TSMixer (2023) | Transformer (2017) | LSTM (1997) | ARIMA (1970) |
|:---|---:|---:|---:|---:|---:|---:|---:|---:|
| ED2022 | 336 | 96 | **0.4788** | 0.5354 | 0.5526 | <u>0.4978</u> | 0.5600 | 1.0193 |
| UH2022 | 336 | 96 | **1.1236** | 1.1783 | 1.1992 | <u>1.1414</u> | 1.1887 | 1.6017 |
| EMF2023 (University-5 w/ PTV) | 336 | 96 | **1.4909** | 1.5548 | <u>1.4955</u> | 1.5056 | 1.5209 | 1.8090 |

### 1.2 · MAE (`sl_test/mae`)

| Dataset | $L$ | $O$ | EMForecaster (2025) | DLinear (2023) | TSMixer (2023) | Transformer (2017) | LSTM (1997) | ARIMA (1970) |
|:---|---:|---:|---:|---:|---:|---:|---:|---:|
| ED2022 | 336 | 96 | **0.3346** | 0.3771 | 0.3859 | <u>0.3559</u> | 0.4066 | 0.6075 |
| UH2022 | 336 | 96 | **0.4443** | 0.4829 | 0.4835 | <u>0.4613</u> | 0.4991 | 0.6807 |
| EMF2023 (University-5 w/ PTV) | 336 | 96 | **0.4397** | 0.5036 | <u>0.4495</u> | 0.4526 | 0.4675 | 0.6213 |

## 2 · Training cost

**Bold** = fastest in the row, <u>underline</u> = runner-up. `-` = not logged (ARIMA is not gradient-trained).

### 2.1 · Total training time (`parameters/running_time`)

| Dataset | $L$ | $O$ | EMForecaster (2025) | DLinear (2023) | TSMixer (2023) | Transformer (2017) | LSTM (1997) | ARIMA (1970) |
|:---|---:|---:|---:|---:|---:|---:|---:|---:|
| ED2022 | 336 | 96 | 35m 18s | **1m 6s** | <u>1m 11s</u> | 43m 10s | 2m 43s | - |
| UH2022 | 336 | 96 | 38m 29s | **1m 18s** | <u>1m 24s</u> | 35m 59s | 3m 19s | - |
| EMF2023 (University-5 w/ PTV) | 336 | 96 | 56m 32s | **48s** | <u>2m 12s</u> | 1h 44m 16s | 5m 59s | - |

### 2.2 · Per-epoch training time (`running_time / len(sl_train/loss)`)

| Dataset | $L$ | $O$ | EMForecaster (2025) | DLinear (2023) | TSMixer (2023) | Transformer (2017) | LSTM (1997) | ARIMA (1970) |
|:---|---:|---:|---:|---:|---:|---:|---:|---:|
| ED2022 | 336 | 96 | 21.2s | **0.7s** | <u>0.7s</u> | 27.3s | 1.6s | - |
| UH2022 | 336 | 96 | 26.8s | **0.8s** | <u>0.8s</u> | 34.8s | 2.0s | - |
| EMF2023 (University-5 w/ PTV) | 336 | 96 | 50.6s | **1.2s** | <u>1.3s</u> | 1m 4s | 3.6s | - |

### 2.3 · Mean epoch cost by model

Pooled over every run of that model in this section.

| Model | Mean / epoch | Estimation for 100 ep | Runs |
|:---|---:|---:|---:|
| DLinear (2023) | 0.9s | ~1m 27s | 3 |
| TSMixer (2023) | 1.0s | ~1m 36s | 3 |
| LSTM (1997) | 2.4s | ~4m 0s | 3 |
| EMForecaster (2025) | 32.9s | ~54m 49s | 3 |
| Transformer (2017) | 42.2s | ~1h 10m 19s | 3 |

## 3 · Conformal prediction

Not marked -- "best" coverage means closest to the target, not highest, and a narrower interval only counts if that target is met.

### 3.1 · Marginal coverage % (`sl_test/ic`)

| Dataset | $L$ | $O$ | EMForecaster (2025) | DLinear (2023) | TSMixer (2023) | Transformer (2017) | LSTM (1997) | ARIMA (1970) |
|:---|---:|---:|---:|---:|---:|---:|---:|---:|
| ED2022 | 336 | 96 | 84.75 | 84.89 | 85.39 | 85.45 | 85.31 | 86.43 |
| UH2022 | 336 | 96 | 86.94 | 87.16 | 87.01 | 86.72 | 86.68 | 84.27 |
| EMF2023 (University-5 w/ PTV) | 336 | 96 | 90.31 | 91.06 | 91.13 | 90.30 | 90.94 | 88.65 |

### 3.2 · Joint coverage % (`sl_test/jc`)

| Dataset | $L$ | $O$ | EMForecaster (2025) | DLinear (2023) | TSMixer (2023) | Transformer (2017) | LSTM (1997) | ARIMA (1970) |
|:---|---:|---:|---:|---:|---:|---:|---:|---:|
| ED2022 | 336 | 96 | 11.47 | 13.08 | 13.64 | 12.97 | 14.58 | 21.65 |
| UH2022 | 336 | 96 | 8.71 | 9.07 | 8.56 | 9.22 | 10.00 | 9.66 |
| EMF2023 (University-5 w/ PTV) | 336 | 96 | 10.67 | 13.81 | 13.80 | 12.21 | 13.00 | 13.06 |

### 3.3 · Mean interval width (`sl_test/interval_width_mean`)

| Dataset | $L$ | $O$ | EMForecaster (2025) | DLinear (2023) | TSMixer (2023) | Transformer (2017) | LSTM (1997) | ARIMA (1970) |
|:---|---:|---:|---:|---:|---:|---:|---:|---:|
| ED2022 | 336 | 96 | 1.273 | 1.448 | 1.516 | 1.366 | 1.570 | 2.460 |
| UH2022 | 336 | 96 | 1.655 | 1.765 | 1.789 | 1.716 | 1.805 | 2.334 |
| EMF2023 (University-5 w/ PTV) | 336 | 96 | 1.862 | 2.095 | 1.991 | 1.903 | 1.988 | 2.446 |

## 4 · Model size

### 4.1 · Trainable weights (`parameters/sl/num_params`)

For multivariate logs this varies by dataset (different channel counts), not just by (model, L, O); shown here is the first value found per (model, L, O).

| Model | L336/O96 |
|:---|---:|
| EMForecaster (2025) | 2,416,959 |
| DLinear (2023) | 64,704 |
| TSMixer (2023) | 56,065 |
| Transformer (2017) | 372,738 |
| LSTM (1997) | 20,726 |

# Multivariate vs Univariate

- Compares: Multivariate vs Per-channel
- Matched on: (dataset, L, O, model, conformal setting); only cells present in both trees appear below
- Delta: Δ is the univariate value relative to Multivariate, so negative Δ means the univariate value is lower; **bold** = better of the two where lower is better

## 1 · Forecast accuracy (Multivariate vs Per-channel)

18 matched cells, from `Multivariate` and `Univariate (Per Channel)`.

### 1.1 · MSE (`sl_test/loss`)

| Dataset | $L$ | $O$ | Model | Multivariate | Per-channel | Δ |
|:---|---:|---:|:---|---:|---:|---:|
| ED2022 | 336 | 96 | EMForecaster (2025) | **0.4788** | 0.4848 | +1.2% |
| ED2022 | 336 | 96 | DLinear (2023) | 0.5354 | **0.5321** | -0.6% |
| ED2022 | 336 | 96 | TSMixer (2023) | 0.5526 | **0.5484** | -0.8% |
| ED2022 | 336 | 96 | Transformer (2017) | **0.4978** | 0.6904 | +38.7% |
| ED2022 | 336 | 96 | LSTM (1997) | **0.5600** | 0.5785 | +3.3% |
| ED2022 | 336 | 96 | ARIMA (1970) | **1.0193** | 1.0194 | 0.0% |
| UH2022 | 336 | 96 | EMForecaster (2025) | **1.1236** | 1.1704 | +4.2% |
| UH2022 | 336 | 96 | DLinear (2023) | **1.1783** | 1.1900 | +1.0% |
| UH2022 | 336 | 96 | TSMixer (2023) | **1.1992** | 1.2932 | +7.8% |
| UH2022 | 336 | 96 | Transformer (2017) | **1.1414** | 1.2270 | +7.5% |
| UH2022 | 336 | 96 | LSTM (1997) | **1.1887** | 1.2180 | +2.5% |
| UH2022 | 336 | 96 | ARIMA (1970) | **1.6017** | 1.6017 | 0.0% |
| EMF2023 (University-5 w/ PTV) | 336 | 96 | EMForecaster (2025) | **1.4909** | 1.5054 | +1.0% |
| EMF2023 (University-5 w/ PTV) | 336 | 96 | DLinear (2023) | 1.5548 | **1.5163** | -2.5% |
| EMF2023 (University-5 w/ PTV) | 336 | 96 | TSMixer (2023) | **1.4955** | 1.5262 | +2.1% |
| EMF2023 (University-5 w/ PTV) | 336 | 96 | Transformer (2017) | **1.5056** | 1.5808 | +5.0% |
| EMF2023 (University-5 w/ PTV) | 336 | 96 | LSTM (1997) | **1.5209** | 1.5435 | +1.5% |
| EMF2023 (University-5 w/ PTV) | 336 | 96 | ARIMA (1970) | 1.8090 | **1.8089** | 0.0% |

### 1.2 · MAE (`sl_test/mae`)

| Dataset | $L$ | $O$ | Model | Multivariate | Per-channel | Δ |
|:---|---:|---:|:---|---:|---:|---:|
| ED2022 | 336 | 96 | EMForecaster (2025) | **0.3346** | 0.3428 | +2.4% |
| ED2022 | 336 | 96 | DLinear (2023) | **0.3771** | 0.3781 | +0.3% |
| ED2022 | 336 | 96 | TSMixer (2023) | **0.3859** | 0.3875 | +0.4% |
| ED2022 | 336 | 96 | Transformer (2017) | **0.3559** | 0.4429 | +24.4% |
| ED2022 | 336 | 96 | LSTM (1997) | **0.4066** | 0.4160 | +2.3% |
| ED2022 | 336 | 96 | ARIMA (1970) | **0.6075** | 0.6076 | 0.0% |
| UH2022 | 336 | 96 | EMForecaster (2025) | **0.4443** | 0.4585 | +3.2% |
| UH2022 | 336 | 96 | DLinear (2023) | **0.4829** | 0.4851 | +0.5% |
| UH2022 | 336 | 96 | TSMixer (2023) | **0.4835** | 0.4961 | +2.6% |
| UH2022 | 336 | 96 | Transformer (2017) | **0.4613** | 0.5075 | +10.0% |
| UH2022 | 336 | 96 | LSTM (1997) | **0.4991** | 0.5089 | +2.0% |
| UH2022 | 336 | 96 | ARIMA (1970) | **0.6807** | 0.6807 | 0.0% |
| EMF2023 (University-5 w/ PTV) | 336 | 96 | EMForecaster (2025) | **0.4397** | 0.4500 | +2.3% |
| EMF2023 (University-5 w/ PTV) | 336 | 96 | DLinear (2023) | 0.5036 | **0.4677** | -7.1% |
| EMF2023 (University-5 w/ PTV) | 336 | 96 | TSMixer (2023) | **0.4495** | 0.4591 | +2.1% |
| EMF2023 (University-5 w/ PTV) | 336 | 96 | Transformer (2017) | **0.4526** | 0.5006 | +10.6% |
| EMF2023 (University-5 w/ PTV) | 336 | 96 | LSTM (1997) | **0.4675** | 0.4795 | +2.6% |
| EMF2023 (University-5 w/ PTV) | 336 | 96 | ARIMA (1970) | 0.6213 | **0.6212** | 0.0% |

## 2 · Training cost (Multivariate vs Per-channel)

`Per-channel` is the mean cost of one channel's model; `all channels` multiplies it by the channel count. Δ compares that total against Multivariate.

### 2.1 · Total training time (`parameters/running_time`)

| Dataset | $L$ | $O$ | Model | Multivariate | Per-channel | Channels | All channels | Δ |
|:---|---:|---:|:---|---:|---:|---:|---:|---:|
| ED2022 | 336 | 96 | EMForecaster (2025) | 35m 18s | 1m 46s | 27 | 47m 32s | +34.7% |
| ED2022 | 336 | 96 | DLinear (2023) | 1m 6s | 56s | 27 | 25m 22s | +2206.1% |
| ED2022 | 336 | 96 | TSMixer (2023) | 1m 11s | 1m 11s | 27 | 31m 46s | +2584.5% |
| ED2022 | 336 | 96 | Transformer (2017) | 43m 10s | 3m 58s | 27 | 1h 47m 7s | +148.1% |
| ED2022 | 336 | 96 | LSTM (1997) | 2m 43s | 50s | 27 | 22m 26s | +725.8% |
| ED2022 | 336 | 96 | ARIMA (1970) | - | - | 27 | - | - |
| UH2022 | 336 | 96 | EMForecaster (2025) | 38m 29s | 1m 53s | 27 | 50m 38s | +31.6% |
| UH2022 | 336 | 96 | DLinear (2023) | 1m 18s | 1m 11s | 27 | 32m 6s | +2369.2% |
| UH2022 | 336 | 96 | TSMixer (2023) | 1m 24s | 1m 29s | 27 | 39m 51s | +2746.4% |
| UH2022 | 336 | 96 | Transformer (2017) | 35m 59s | 5m 1s | 27 | 2h 15m 23s | +276.2% |
| UH2022 | 336 | 96 | LSTM (1997) | 3m 19s | 56s | 27 | 25m 19s | +663.3% |
| UH2022 | 336 | 96 | ARIMA (1970) | - | - | 27 | - | - |
| EMF2023 (University-5 w/ PTV) | 336 | 96 | EMForecaster (2025) | 56m 32s | 3m 38s | 29 | 1h 45m 23s | +86.4% |
| EMF2023 (University-5 w/ PTV) | 336 | 96 | DLinear (2023) | 48s | 1m 40s | 29 | 48m 30s | +5962.5% |
| EMF2023 (University-5 w/ PTV) | 336 | 96 | TSMixer (2023) | 2m 12s | 2m 39s | 29 | 1h 16m 42s | +3386.4% |
| EMF2023 (University-5 w/ PTV) | 336 | 96 | Transformer (2017) | 1h 44m 16s | 8m 55s | 29 | 4h 18m 39s | +148.1% |
| EMF2023 (University-5 w/ PTV) | 336 | 96 | LSTM (1997) | 5m 59s | 1m 30s | 29 | 43m 19s | +624.0% |
| EMF2023 (University-5 w/ PTV) | 336 | 96 | ARIMA (1970) | - | - | 29 | - | - |

## 3 · Conformal prediction (Multivariate vs Per-channel)

Δ is in percentage points for coverage, relative for interval width. Coverage is not marked -- "best" means closest to the target.

### 3.1 · Marginal coverage % (`sl_test/ic`)

| Dataset | $L$ | $O$ | Model | Multivariate | Per-channel | Δ |
|:---|---:|---:|:---|---:|---:|---:|
| ED2022 | 336 | 96 | EMForecaster (2025) | 84.75 | 84.94 | +0.19 pp |
| ED2022 | 336 | 96 | DLinear (2023) | 84.89 | 84.64 | -0.25 pp |
| ED2022 | 336 | 96 | TSMixer (2023) | 85.39 | 85.59 | +0.20 pp |
| ED2022 | 336 | 96 | Transformer (2017) | 85.45 | 84.46 | -0.99 pp |
| ED2022 | 336 | 96 | LSTM (1997) | 85.31 | 85.11 | -0.20 pp |
| ED2022 | 336 | 96 | ARIMA (1970) | 86.43 | 86.43 | 0.00 pp |
| UH2022 | 336 | 96 | EMForecaster (2025) | 86.94 | 86.71 | -0.22 pp |
| UH2022 | 336 | 96 | DLinear (2023) | 87.16 | 86.24 | -0.92 pp |
| UH2022 | 336 | 96 | TSMixer (2023) | 87.01 | 86.70 | -0.31 pp |
| UH2022 | 336 | 96 | Transformer (2017) | 86.72 | 85.78 | -0.94 pp |
| UH2022 | 336 | 96 | LSTM (1997) | 86.68 | 85.82 | -0.86 pp |
| UH2022 | 336 | 96 | ARIMA (1970) | 84.27 | 84.28 | 0.00 pp |
| EMF2023 (University-5 w/ PTV) | 336 | 96 | EMForecaster (2025) | 90.31 | 90.46 | +0.15 pp |
| EMF2023 (University-5 w/ PTV) | 336 | 96 | DLinear (2023) | 91.06 | 90.76 | -0.31 pp |
| EMF2023 (University-5 w/ PTV) | 336 | 96 | TSMixer (2023) | 91.13 | 91.33 | +0.20 pp |
| EMF2023 (University-5 w/ PTV) | 336 | 96 | Transformer (2017) | 90.30 | 90.57 | +0.27 pp |
| EMF2023 (University-5 w/ PTV) | 336 | 96 | LSTM (1997) | 90.94 | 90.05 | -0.89 pp |
| EMF2023 (University-5 w/ PTV) | 336 | 96 | ARIMA (1970) | 88.65 | 88.65 | 0.00 pp |

### 3.2 · Joint coverage % (`sl_test/jc`)

| Dataset | $L$ | $O$ | Model | Multivariate | Per-channel | Δ |
|:---|---:|---:|:---|---:|---:|---:|
| ED2022 | 336 | 96 | EMForecaster (2025) | 11.47 | 10.50 | -0.98 pp |
| ED2022 | 336 | 96 | DLinear (2023) | 13.08 | 11.10 | -1.99 pp |
| ED2022 | 336 | 96 | TSMixer (2023) | 13.64 | 12.99 | -0.64 pp |
| ED2022 | 336 | 96 | Transformer (2017) | 12.97 | 14.80 | +1.83 pp |
| ED2022 | 336 | 96 | LSTM (1997) | 14.58 | 14.14 | -0.44 pp |
| ED2022 | 336 | 96 | ARIMA (1970) | 21.65 | 21.65 | 0.00 pp |
| UH2022 | 336 | 96 | EMForecaster (2025) | 8.71 | 6.87 | -1.83 pp |
| UH2022 | 336 | 96 | DLinear (2023) | 9.07 | 6.60 | -2.47 pp |
| UH2022 | 336 | 96 | TSMixer (2023) | 8.56 | 7.59 | -0.96 pp |
| UH2022 | 336 | 96 | Transformer (2017) | 9.22 | 7.92 | -1.30 pp |
| UH2022 | 336 | 96 | LSTM (1997) | 10.00 | 8.64 | -1.36 pp |
| UH2022 | 336 | 96 | ARIMA (1970) | 9.66 | 9.66 | 0.00 pp |
| EMF2023 (University-5 w/ PTV) | 336 | 96 | EMForecaster (2025) | 10.67 | 11.50 | +0.83 pp |
| EMF2023 (University-5 w/ PTV) | 336 | 96 | DLinear (2023) | 13.81 | 11.87 | -1.94 pp |
| EMF2023 (University-5 w/ PTV) | 336 | 96 | TSMixer (2023) | 13.80 | 14.82 | +1.02 pp |
| EMF2023 (University-5 w/ PTV) | 336 | 96 | Transformer (2017) | 12.21 | 14.61 | +2.40 pp |
| EMF2023 (University-5 w/ PTV) | 336 | 96 | LSTM (1997) | 13.00 | 12.02 | -0.98 pp |
| EMF2023 (University-5 w/ PTV) | 336 | 96 | ARIMA (1970) | 13.06 | 13.06 | 0.00 pp |

### 3.3 · Mean interval width (`sl_test/interval_width_mean`)

| Dataset | $L$ | $O$ | Model | Multivariate | Per-channel | Δ |
|:---|---:|---:|:---|---:|---:|---:|
| ED2022 | 336 | 96 | EMForecaster (2025) | **1.273** | 1.314 | +3.2% |
| ED2022 | 336 | 96 | DLinear (2023) | 1.448 | **1.434** | -1.0% |
| ED2022 | 336 | 96 | TSMixer (2023) | 1.516 | **1.516** | 0.0% |
| ED2022 | 336 | 96 | Transformer (2017) | **1.366** | 1.678 | +22.8% |
| ED2022 | 336 | 96 | LSTM (1997) | **1.570** | 1.582 | +0.7% |
| ED2022 | 336 | 96 | ARIMA (1970) | 2.460 | **2.460** | 0.0% |
| UH2022 | 336 | 96 | EMForecaster (2025) | **1.655** | 1.683 | +1.7% |
| UH2022 | 336 | 96 | DLinear (2023) | 1.765 | **1.710** | -3.1% |
| UH2022 | 336 | 96 | TSMixer (2023) | **1.789** | 1.793 | +0.3% |
| UH2022 | 336 | 96 | Transformer (2017) | **1.716** | 1.845 | +7.5% |
| UH2022 | 336 | 96 | LSTM (1997) | **1.805** | 1.812 | +0.4% |
| UH2022 | 336 | 96 | ARIMA (1970) | **2.334** | 2.334 | 0.0% |
| EMF2023 (University-5 w/ PTV) | 336 | 96 | EMForecaster (2025) | **1.862** | 1.941 | +4.2% |
| EMF2023 (University-5 w/ PTV) | 336 | 96 | DLinear (2023) | 2.095 | **1.935** | -7.6% |
| EMF2023 (University-5 w/ PTV) | 336 | 96 | TSMixer (2023) | **1.991** | 2.202 | +10.6% |
| EMF2023 (University-5 w/ PTV) | 336 | 96 | Transformer (2017) | **1.903** | 2.112 | +11.0% |
| EMF2023 (University-5 w/ PTV) | 336 | 96 | LSTM (1997) | **1.988** | 1.996 | +0.4% |
| EMF2023 (University-5 w/ PTV) | 336 | 96 | ARIMA (1970) | 2.446 | **2.446** | 0.0% |

## 4 · Scoreboard

### 4.1 · Matched-cell wins by metric

How many matched cells each pipeline wins, counted only where lower is better.

| Comparison | Metric | Multivariate | Univariate | Tie |
|:---|:---|---:|---:|---:|
| Multivariate vs Per-channel | MSE | 14 | 4 | 0 |
| Multivariate vs Per-channel | MAE | 16 | 2 | 0 |
| Multivariate vs Per-channel | Interval width | 12 | 6 | 0 |
