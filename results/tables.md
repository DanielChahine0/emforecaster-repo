# Univariate Series

- Models: EMForecaster (2025), DLinear (2023), TSMixer (2023), PatchTST (2023), Transformer (2017), LSTM (1997), ARIMA (1970)
- Datasets: DET, PTV, PTV2
- Grid: L192/O96, L192/O336, L336/O96, L336/O336
- Runs: 168 logs; each cell averages **1-5** replicate run(s)
- Conformal prediction: on for every run

## 1 · Forecast accuracy

**Bold** = best (lowest) in the row, <u>underline</u> = runner-up. `-` = not logged.

### 1.1 · MSE (`sl_test/loss`)

| Dataset | $L$ | $O$ | EMForecaster (2025) | DLinear (2023) | TSMixer (2023) | PatchTST (2023) | Transformer (2017) | LSTM (1997) | ARIMA (1970) |
|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| DET | 192 | 96 | **0.3135** | 0.4520 | <u>0.3625</u> | 0.3988 | 0.5986 | 0.3754 | 1.2371 |
| DET | 192 | 336 | **0.4261** | 0.5301 | <u>0.4602</u> | 0.5246 | 0.9783 | 0.4804 | 1.4194 |
| DET | 336 | 96 | **0.3019** | 0.3994 | <u>0.3552</u> | 0.3955 | 1.3149 | 0.4366 | 1.2371 |
| DET | 336 | 336 | **0.3950** | 0.5044 | <u>0.4585</u> | 0.5129 | 1.4857 | 0.5065 | 1.4194 |
| PTV | 192 | 96 | **0.2610** | 0.4703 | <u>0.3135</u> | 0.3564 | 0.6426 | 0.3743 | 1.1840 |
| PTV | 192 | 336 | **0.3224** | 0.4739 | <u>0.3672</u> | 0.4157 | 1.3925 | 0.4159 | 1.3061 |
| PTV | 336 | 96 | **0.2415** | 0.3542 | <u>0.3064</u> | 0.3285 | 1.2389 | 0.4095 | 1.1840 |
| PTV | 336 | 336 | **0.3181** | 0.4252 | <u>0.3838</u> | 0.3909 | 0.8120 | 0.4774 | 1.3061 |
| PTV2 | 192 | 96 | **0.3199** | 0.4003 | <u>0.3339</u> | 0.3658 | 0.3819 | 0.3488 | 1.0128 |
| PTV2 | 192 | 336 | **0.3574** | 0.4130 | <u>0.3601</u> | 0.4146 | 1.1660 | 0.3900 | 1.1346 |
| PTV2 | 336 | 96 | **0.3025** | 0.3312 | <u>0.3038</u> | 0.3428 | 0.8388 | 0.3372 | 1.0128 |
| PTV2 | 336 | 336 | <u>0.3375</u> | 0.3745 | **0.3348** | 0.4380 | 0.4592 | 0.3646 | 1.1346 |

### 1.2 · MAE (`sl_test/mae`)

| Dataset | $L$ | $O$ | EMForecaster (2025) | DLinear (2023) | TSMixer (2023) | PatchTST (2023) | Transformer (2017) | LSTM (1997) | ARIMA (1970) |
|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| DET | 192 | 96 | **0.3915** | 0.4961 | <u>0.4253</u> | 0.4527 | 0.5937 | 0.4572 | 0.8468 |
| DET | 192 | 336 | **0.4654** | 0.5318 | <u>0.4851</u> | 0.5232 | 0.7618 | 0.5108 | 0.9650 |
| DET | 336 | 96 | **0.3778** | 0.4450 | <u>0.4226</u> | 0.4532 | 0.9232 | 0.4872 | 0.8468 |
| DET | 336 | 336 | **0.4436** | 0.5106 | <u>0.4849</u> | 0.5151 | 1.0062 | 0.5327 | 0.9650 |
| PTV | 192 | 96 | **0.3740** | 0.5178 | <u>0.4036</u> | 0.4441 | 0.5832 | 0.4711 | 0.8374 |
| PTV | 192 | 336 | **0.4201** | 0.5069 | <u>0.4312</u> | 0.4723 | 0.9897 | 0.4942 | 0.9381 |
| PTV | 336 | 96 | **0.3585** | 0.4275 | <u>0.3953</u> | 0.4235 | 0.9093 | 0.4994 | 0.8374 |
| PTV | 336 | 336 | **0.4151** | 0.4760 | <u>0.4387</u> | 0.4542 | 0.7102 | 0.5395 | 0.9381 |
| PTV2 | 192 | 96 | **0.4372** | 0.4945 | <u>0.4449</u> | 0.4672 | 0.4871 | 0.4634 | 0.7957 |
| PTV2 | 192 | 336 | <u>0.4678</u> | 0.5030 | **0.4651** | 0.4952 | 0.8950 | 0.4947 | 0.8719 |
| PTV2 | 336 | 96 | <u>0.4258</u> | 0.4434 | **0.4223** | 0.4546 | 0.7231 | 0.4568 | 0.7957 |
| PTV2 | 336 | 336 | <u>0.4542</u> | 0.4755 | **0.4483** | 0.5092 | 0.5352 | 0.4812 | 0.8719 |

## 2 · Training cost

**Bold** = fastest in the row, <u>underline</u> = runner-up. `-` = not logged (ARIMA is not gradient-trained).

### 2.1 · Total training time (`parameters/running_time`)

| Dataset | $L$ | $O$ | EMForecaster (2025) | DLinear (2023) | TSMixer (2023) | PatchTST (2023) | Transformer (2017) | LSTM (1997) | ARIMA (1970) |
|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| DET | 192 | 96 | 1m 19s | **28s** | 38s | 41s | <u>35s</u> | 40s | - |
| DET | 192 | 336 | 54s | **30s** | 40s | 42s | <u>35s</u> | 41s | - |
| DET | 336 | 96 | 1m 27s | **39s** | <u>44s</u> | 53s | 49s | 56s | - |
| DET | 336 | 336 | 1m 12s | **31s** | <u>40s</u> | 48s | 1m 11s | 44s | - |
| PTV | 192 | 96 | 46s | **27s** | <u>35s</u> | 38s | 52s | 37s | - |
| PTV | 192 | 336 | 36s | **28s** | <u>35s</u> | 38s | 37s | 39s | - |
| PTV | 336 | 96 | 42s | **35s** | <u>36s</u> | 45s | 41s | 1m 39s | - |
| PTV | 336 | 336 | 41s | **28s** | <u>36s</u> | 41s | 1m 0s | 40s | - |
| PTV2 | 192 | 96 | 1m 3s | **41s** | <u>54s</u> | 1m 10s | 1m 9s | 1m 4s | - |
| PTV2 | 192 | 336 | 1m 9s | <u>42s</u> | 55s | **38s** | 1m 46s | 1m 10s | - |
| PTV2 | 336 | 96 | 1m 36s | **52s** | <u>58s</u> | 59s | 1m 8s | 1m 48s | - |
| PTV2 | 336 | 336 | 3m 37s | <u>44s</u> | 57s | **31s** | 2m 2s | 1m 18s | - |

### 2.2 · Per-epoch training time (`running_time / len(sl_train/loss)`)

| Dataset | $L$ | $O$ | EMForecaster (2025) | DLinear (2023) | TSMixer (2023) | PatchTST (2023) | Transformer (2017) | LSTM (1997) | ARIMA (1970) |
|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| DET | 192 | 96 | 0.8s | **0.3s** | <u>0.4s</u> | 0.4s | 0.7s | 0.4s | - |
| DET | 192 | 336 | 0.8s | **0.3s** | <u>0.4s</u> | 0.4s | 0.8s | 0.4s | - |
| DET | 336 | 96 | 1.2s | **0.4s** | <u>0.4s</u> | 0.5s | 1.0s | 0.6s | - |
| DET | 336 | 336 | 1.2s | **0.3s** | <u>0.4s</u> | 0.5s | 1.4s | 0.4s | - |
| PTV | 192 | 96 | 0.6s | **0.3s** | <u>0.3s</u> | 0.4s | 0.5s | 0.4s | - |
| PTV | 192 | 336 | 0.6s | **0.3s** | <u>0.3s</u> | 0.4s | 0.6s | 0.4s | - |
| PTV | 336 | 96 | 0.8s | **0.3s** | <u>0.4s</u> | 0.4s | 0.8s | 1.0s | - |
| PTV | 336 | 336 | 0.9s | **0.3s** | <u>0.4s</u> | 0.4s | 1.0s | 0.4s | - |
| PTV2 | 192 | 96 | 1.2s | **0.4s** | <u>0.5s</u> | 0.7s | 1.1s | 0.6s | - |
| PTV2 | 192 | 336 | 1.3s | **0.4s** | <u>0.6s</u> | 0.7s | 1.4s | 0.7s | - |
| PTV2 | 336 | 96 | 2.0s | **0.5s** | <u>0.6s</u> | 0.8s | 1.9s | 1.1s | - |
| PTV2 | 336 | 336 | 4.8s | **0.4s** | <u>0.6s</u> | 0.8s | 2.6s | 0.8s | - |

### 2.3 · Mean epoch cost by model

Pooled over every run of that model in this section.

| Model | Mean / epoch | Estimation for 100 ep | Runs |
|:---|---:|---:|---:|
| DLinear (2023) | 0.4s | ~39s | 24 |
| TSMixer (2023) | 0.5s | ~45s | 24 |
| PatchTST (2023) | 0.6s | ~56s | 24 |
| LSTM (1997) | 0.7s | ~1m 14s | 24 |
| Transformer (2017) | 1.2s | ~1m 59s | 24 |
| EMForecaster (2025) | 1.3s | ~2m 13s | 24 |

## 3 · Conformal prediction

Not marked -- "best" coverage means closest to the target, not highest, and a narrower interval only counts if that target is met.

### 3.1 · Marginal coverage % (`sl_test/ic`)

| Dataset | $L$ | $O$ | EMForecaster (2025) | DLinear (2023) | TSMixer (2023) | PatchTST (2023) | Transformer (2017) | LSTM (1997) | ARIMA (1970) |
|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| DET | 192 | 96 | 88.93 | 91.12 | 90.18 | 90.35 | 90.38 | 89.92 | 92.69 |
| DET | 192 | 336 | 88.38 | 90.57 | 89.47 | 89.57 | 88.46 | 89.76 | 92.16 |
| DET | 336 | 96 | 87.64 | 90.67 | 89.41 | 89.74 | 91.01 | 89.56 | 92.69 |
| DET | 336 | 336 | 87.93 | 90.54 | 89.17 | 90.17 | 91.92 | 89.75 | 92.16 |
| PTV | 192 | 96 | 87.06 | 87.19 | 90.71 | 89.43 | 78.75 | 88.39 | 86.92 |
| PTV | 192 | 336 | 85.14 | 86.38 | 88.21 | 87.13 | 81.54 | 87.88 | 82.95 |
| PTV | 336 | 96 | 88.16 | 88.31 | 92.57 | 89.64 | 81.31 | 88.22 | 86.92 |
| PTV | 336 | 336 | 86.06 | 86.05 | 89.45 | 86.90 | 83.68 | 87.15 | 82.95 |
| PTV2 | 192 | 96 | 90.43 | 89.48 | 90.76 | 89.76 | 89.98 | 89.99 | 85.34 |
| PTV2 | 192 | 336 | 90.06 | 89.97 | 90.45 | 90.28 | 84.27 | 89.32 | 84.60 |
| PTV2 | 336 | 96 | 90.71 | 91.06 | 91.34 | 91.21 | 86.67 | 90.09 | 85.34 |
| PTV2 | 336 | 336 | 90.73 | 90.57 | 90.99 | 90.36 | 88.04 | 89.55 | 84.60 |

### 3.2 · Joint coverage % (`sl_test/jc`)

| Dataset | $L$ | $O$ | EMForecaster (2025) | DLinear (2023) | TSMixer (2023) | PatchTST (2023) | Transformer (2017) | LSTM (1997) | ARIMA (1970) |
|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| DET | 192 | 96 | 19.04 | 39.10 | 22.51 | 29.88 | 31.91 | 28.50 | 51.33 |
| DET | 192 | 336 | 3.16 | 14.84 | 7.06 | 8.22 | 12.18 | 9.76 | 18.46 |
| DET | 336 | 96 | 10.76 | 43.08 | 17.10 | 33.59 | 50.71 | 31.47 | 51.33 |
| DET | 336 | 336 | 1.83 | 12.63 | 5.26 | 15.19 | 22.31 | 9.07 | 18.46 |
| PTV | 192 | 96 | 5.15 | 14.33 | 9.74 | 13.55 | 22.78 | 14.60 | 32.03 |
| PTV | 192 | 336 | 0.02 | 0.59 | 0.86 | 0.39 | 3.54 | 1.31 | 0.12 |
| PTV | 336 | 96 | 4.70 | 9.67 | 21.99 | 13.47 | 29.40 | 14.16 | 32.03 |
| PTV | 336 | 336 | 0.00 | 0.00 | 3.13 | 1.29 | 2.60 | 1.04 | 0.12 |
| PTV2 | 192 | 96 | 4.77 | 5.99 | 5.73 | 5.30 | 7.38 | 5.04 | 15.64 |
| PTV2 | 192 | 336 | 0.00 | 0.00 | 0.17 | 0.36 | 0.00 | 0.04 | 0.00 |
| PTV2 | 336 | 96 | 3.51 | 5.82 | 4.70 | 5.18 | 8.65 | 4.30 | 15.64 |
| PTV2 | 336 | 336 | 0.00 | 0.00 | 0.01 | 0.25 | 0.05 | 0.00 | 0.00 |

### 3.3 · Mean interval width (`sl_test/interval_width_mean`)

| Dataset | $L$ | $O$ | EMForecaster (2025) | DLinear (2023) | TSMixer (2023) | PatchTST (2023) | Transformer (2017) | LSTM (1997) | ARIMA (1970) |
|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| DET | 192 | 96 | 1.570 | 2.190 | 1.888 | 2.023 | 2.407 | 1.870 | 3.769 |
| DET | 192 | 336 | 1.930 | 2.429 | 2.190 | 2.376 | 3.027 | 2.223 | 3.981 |
| DET | 336 | 96 | 1.449 | 2.097 | 1.789 | 1.971 | 3.767 | 1.997 | 3.769 |
| DET | 336 | 336 | 1.817 | 2.403 | 2.154 | 2.425 | 4.162 | 2.262 | 3.981 |
| PTV | 192 | 96 | 1.408 | 1.949 | 1.759 | 1.817 | 1.945 | 1.782 | 3.077 |
| PTV | 192 | 336 | 1.524 | 1.895 | 1.760 | 1.836 | 3.077 | 1.852 | 3.099 |
| PTV | 336 | 96 | 1.374 | 1.718 | 1.881 | 1.733 | 2.876 | 1.881 | 3.077 |
| PTV | 336 | 336 | 1.534 | 1.778 | 1.909 | 1.782 | 2.499 | 1.964 | 3.099 |
| PTV2 | 192 | 96 | 1.840 | 2.021 | 1.915 | 1.941 | 1.977 | 1.897 | 2.955 |
| PTV2 | 192 | 336 | 1.930 | 2.082 | 1.974 | 2.125 | 3.158 | 1.969 | 3.119 |
| PTV2 | 336 | 96 | 1.785 | 1.908 | 1.845 | 1.952 | 2.654 | 1.862 | 2.955 |
| PTV2 | 336 | 336 | 1.893 | 2.010 | 1.930 | 2.179 | 2.066 | 1.904 | 3.119 |

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
| LSTM (1997) | 20,674 | 28,594 | 20,674 | 28,594 |

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

- Models: EMForecaster (2025), DLinear (2023), TSMixer (2023), PatchTST (2023), Transformer (2017), LSTM (1997), ARIMA (1970)
- Datasets: ED2022, UH2022, EMF2023 (University-5 w/ PTV)
- Grid: L336/O96
- Runs: 21 logs; each cell averages **1** replicate run(s)
- Conformal prediction: on for every run

## 1 · Forecast accuracy

**Bold** = best (lowest) in the row, <u>underline</u> = runner-up. `-` = not logged.

### 1.1 · MSE (`sl_test/loss`)

| Dataset | $L$ | $O$ | EMForecaster (2025) | DLinear (2023) | TSMixer (2023) | PatchTST (2023) | Transformer (2017) | LSTM (1997) | ARIMA (1970) |
|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ED2022 | 336 | 96 | **0.4788** | 0.5354 | 0.5526 | 0.5439 | <u>0.4978</u> | 0.5600 | 1.0193 |
| UH2022 | 336 | 96 | **1.1236** | 1.1783 | 1.1992 | 1.1741 | <u>1.1414</u> | 1.1887 | 1.6017 |
| EMF2023 (University-5 w/ PTV) | 336 | 96 | **1.4909** | 1.5548 | <u>1.4955</u> | 1.5147 | 1.5056 | 1.5209 | 1.8090 |

### 1.2 · MAE (`sl_test/mae`)

| Dataset | $L$ | $O$ | EMForecaster (2025) | DLinear (2023) | TSMixer (2023) | PatchTST (2023) | Transformer (2017) | LSTM (1997) | ARIMA (1970) |
|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ED2022 | 336 | 96 | **0.3346** | 0.3771 | 0.3859 | 0.3868 | <u>0.3559</u> | 0.4066 | 0.6075 |
| UH2022 | 336 | 96 | **0.4443** | 0.4829 | 0.4835 | 0.4791 | <u>0.4613</u> | 0.4991 | 0.6807 |
| EMF2023 (University-5 w/ PTV) | 336 | 96 | **0.4397** | 0.5036 | <u>0.4495</u> | 0.4581 | 0.4526 | 0.4675 | 0.6213 |

## 2 · Training cost

**Bold** = fastest in the row, <u>underline</u> = runner-up. `-` = not logged (ARIMA is not gradient-trained).

### 2.1 · Total training time (`parameters/running_time`)

| Dataset | $L$ | $O$ | EMForecaster (2025) | DLinear (2023) | TSMixer (2023) | PatchTST (2023) | Transformer (2017) | LSTM (1997) | ARIMA (1970) |
|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ED2022 | 336 | 96 | 35m 18s | **1m 6s** | <u>1m 11s</u> | 2m 11s | 43m 10s | 2m 43s | - |
| UH2022 | 336 | 96 | 38m 29s | **1m 18s** | <u>1m 24s</u> | 2m 38s | 35m 59s | 3m 19s | - |
| EMF2023 (University-5 w/ PTV) | 336 | 96 | 56m 32s | **48s** | <u>2m 12s</u> | 4m 43s | 1h 44m 16s | 5m 59s | - |

### 2.2 · Per-epoch training time (`running_time / len(sl_train/loss)`)

| Dataset | $L$ | $O$ | EMForecaster (2025) | DLinear (2023) | TSMixer (2023) | PatchTST (2023) | Transformer (2017) | LSTM (1997) | ARIMA (1970) |
|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ED2022 | 336 | 96 | 21.2s | **0.7s** | <u>0.7s</u> | 1.3s | 27.3s | 1.6s | - |
| UH2022 | 336 | 96 | 26.8s | **0.8s** | <u>0.8s</u> | 1.6s | 34.8s | 2.0s | - |
| EMF2023 (University-5 w/ PTV) | 336 | 96 | 50.6s | **1.2s** | <u>1.3s</u> | 2.8s | 1m 4s | 3.6s | - |

### 2.3 · Mean epoch cost by model

Pooled over every run of that model in this section.

| Model | Mean / epoch | Estimation for 100 ep | Runs |
|:---|---:|---:|---:|
| DLinear (2023) | 0.9s | ~1m 27s | 3 |
| TSMixer (2023) | 1.0s | ~1m 36s | 3 |
| PatchTST (2023) | 1.9s | ~3m 11s | 3 |
| LSTM (1997) | 2.4s | ~4m 0s | 3 |
| EMForecaster (2025) | 32.9s | ~54m 49s | 3 |
| Transformer (2017) | 42.2s | ~1h 10m 19s | 3 |

## 3 · Conformal prediction

Not marked -- "best" coverage means closest to the target, not highest, and a narrower interval only counts if that target is met.

### 3.1 · Marginal coverage % (`sl_test/ic`)

| Dataset | $L$ | $O$ | EMForecaster (2025) | DLinear (2023) | TSMixer (2023) | PatchTST (2023) | Transformer (2017) | LSTM (1997) | ARIMA (1970) |
|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ED2022 | 336 | 96 | 84.75 | 84.89 | 85.39 | 85.06 | 85.45 | 85.31 | 86.43 |
| UH2022 | 336 | 96 | 86.94 | 87.16 | 87.01 | 86.77 | 86.72 | 86.68 | 84.27 |
| EMF2023 (University-5 w/ PTV) | 336 | 96 | 90.31 | 91.06 | 91.13 | 90.99 | 90.30 | 90.94 | 88.65 |

### 3.2 · Joint coverage % (`sl_test/jc`)

| Dataset | $L$ | $O$ | EMForecaster (2025) | DLinear (2023) | TSMixer (2023) | PatchTST (2023) | Transformer (2017) | LSTM (1997) | ARIMA (1970) |
|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ED2022 | 336 | 96 | 11.47 | 13.08 | 13.64 | 14.09 | 12.97 | 14.58 | 21.65 |
| UH2022 | 336 | 96 | 8.71 | 9.07 | 8.56 | 9.63 | 9.22 | 10.00 | 9.66 |
| EMF2023 (University-5 w/ PTV) | 336 | 96 | 10.67 | 13.81 | 13.80 | 12.87 | 12.21 | 13.00 | 13.06 |

### 3.3 · Mean interval width (`sl_test/interval_width_mean`)

| Dataset | $L$ | $O$ | EMForecaster (2025) | DLinear (2023) | TSMixer (2023) | PatchTST (2023) | Transformer (2017) | LSTM (1997) | ARIMA (1970) |
|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ED2022 | 336 | 96 | 1.273 | 1.448 | 1.516 | 1.480 | 1.366 | 1.570 | 2.460 |
| UH2022 | 336 | 96 | 1.655 | 1.765 | 1.789 | 1.770 | 1.716 | 1.805 | 2.334 |
| EMF2023 (University-5 w/ PTV) | 336 | 96 | 1.862 | 2.095 | 1.991 | 2.006 | 1.903 | 1.988 | 2.446 |

## 4 · Model size

### 4.1 · Trainable weights (`parameters/sl/num_params`)

For multivariate logs this varies by dataset (different channel counts), not just by (model, L, O); shown here is the first value found per (model, L, O).

| Model | L336/O96 |
|:---|---:|
| EMForecaster (2025) | 2,416,959 |
| DLinear (2023) | 64,704 |
| TSMixer (2023) | 56,065 |
| PatchTST (2023) | 72,182 |
| Transformer (2017) | 372,738 |
| LSTM (1997) | 20,726 |

# Cluster (k-Grouped Channels)

- Models: EMForecaster (2025), DLinear (2023), Transformer (2017), LSTM (1997)
- Datasets: ED2022, UH2022, EMF2023 (University-5 w/ PTV)
- Grid: L336/O96
- Channel clusters: k = 1, 9 (each k is its own cell, never averaged together)
- Runs: 21 logs; each cell averages **1** replicate run(s)
- Conformal prediction: on for every run

## 1 · Forecast accuracy

**Bold** = best (lowest) in the row, <u>underline</u> = runner-up. `-` = not logged.

### 1.1 · MSE (`sl_test/loss`)

| Dataset | $L$ | $O$ | $k$ | EMForecaster (2025) | DLinear (2023) | Transformer (2017) | LSTM (1997) |
|:---|---:|---:|---:|---:|---:|---:|---:|
| ED2022 | 336 | 96 | 1 | - | 0.6641 | <u>0.5541</u> | **0.5449** |
| ED2022 | 336 | 96 | 9 | <u>0.4905</u> | 0.5923 | **0.4873** | 0.5499 |
| UH2022 | 336 | 96 | 1 | - | 1.4913 | <u>1.3022</u> | **1.2336** |
| UH2022 | 336 | 96 | 9 | <u>1.1838</u> | 1.3412 | **1.1533** | 1.1938 |
| EMF2023 (University-5 w/ PTV) | 336 | 96 | 1 | - | 1.7097 | <u>1.6641</u> | **1.5251** |
| EMF2023 (University-5 w/ PTV) | 336 | 96 | 9 | 1.5190 | 1.6135 | **1.5091** | <u>1.5132</u> |

### 1.2 · MAE (`sl_test/mae`)

| Dataset | $L$ | $O$ | $k$ | EMForecaster (2025) | DLinear (2023) | Transformer (2017) | LSTM (1997) |
|:---|---:|---:|---:|---:|---:|---:|---:|
| ED2022 | 336 | 96 | 1 | - | 0.5021 | <u>0.3956</u> | **0.3949** |
| ED2022 | 336 | 96 | 9 | <u>0.3505</u> | 0.4254 | **0.3480** | 0.3967 |
| UH2022 | 336 | 96 | 1 | - | 0.6552 | <u>0.5179</u> | **0.4947** |
| UH2022 | 336 | 96 | 9 | <u>0.4688</u> | 0.5568 | **0.4655** | 0.4916 |
| EMF2023 (University-5 w/ PTV) | 336 | 96 | 1 | - | 0.5965 | <u>0.5562</u> | **0.4652** |
| EMF2023 (University-5 w/ PTV) | 336 | 96 | 9 | 0.4629 | 0.5033 | **0.4555** | <u>0.4590</u> |

## 2 · Training cost

**Bold** = fastest in the row, <u>underline</u> = runner-up.

### 2.1 · Total training time (`parameters/running_time`)

| Dataset | $L$ | $O$ | $k$ | EMForecaster (2025) | DLinear (2023) | Transformer (2017) | LSTM (1997) |
|:---|---:|---:|---:|---:|---:|---:|---:|
| ED2022 | 336 | 96 | 1 | - | **1m 9s** | 13m 44s | <u>2m 8s</u> |
| ED2022 | 336 | 96 | 9 | 7m 10s | **1m 2s** | 23m 53s | <u>1m 31s</u> |
| UH2022 | 336 | 96 | 1 | - | **1m 21s** | 13m 53s | <u>2m 4s</u> |
| UH2022 | 336 | 96 | 9 | 9m 0s | **1m 14s** | 16m 15s | <u>1m 50s</u> |
| EMF2023 (University-5 w/ PTV) | 336 | 96 | 1 | - | **2m 27s** | 13m 1s | <u>3m 15s</u> |
| EMF2023 (University-5 w/ PTV) | 336 | 96 | 9 | 17m 56s | **2m 5s** | 33m 1s | <u>3m 17s</u> |

### 2.2 · Per-epoch training time (`running_time / len(sl_train/loss)`)

| Dataset | $L$ | $O$ | $k$ | EMForecaster (2025) | DLinear (2023) | Transformer (2017) | LSTM (1997) |
|:---|---:|---:|---:|---:|---:|---:|---:|
| ED2022 | 336 | 96 | 1 | - | **0.7s** | 15.3s | <u>1.3s</u> |
| ED2022 | 336 | 96 | 9 | 13.0s | **0.6s** | 15.4s | <u>0.9s</u> |
| UH2022 | 336 | 96 | 1 | - | **0.8s** | 18.9s | <u>1.2s</u> |
| UH2022 | 336 | 96 | 9 | 16.9s | **0.7s** | 19.9s | <u>1.1s</u> |
| EMF2023 (University-5 w/ PTV) | 336 | 96 | 1 | - | **1.7s** | 37.2s | <u>2.1s</u> |
| EMF2023 (University-5 w/ PTV) | 336 | 96 | 9 | 32.6s | **1.2s** | 38.8s | <u>2.0s</u> |

### 2.3 · Mean epoch cost by model

Pooled over every run of that model in this section.

| Model | Mean / epoch | Estimation for 100 ep | Runs |
|:---|---:|---:|---:|
| DLinear (2023) (k=9) | 0.9s | ~1m 27s | 3 |
| DLinear (2023) (k=1) | 1.1s | ~1m 46s | 3 |
| LSTM (1997) (k=9) | 1.3s | ~2m 13s | 3 |
| LSTM (1997) (k=1) | 1.5s | ~2m 33s | 3 |
| EMForecaster (2025) (k=9) | 20.8s | ~34m 44s | 3 |
| Transformer (2017) (k=1) | 23.8s | ~39m 39s | 3 |
| Transformer (2017) (k=9) | 24.7s | ~41m 12s | 3 |

## 3 · Conformal prediction

Not marked -- "best" coverage means closest to the target, not highest, and a narrower interval only counts if that target is met.

### 3.1 · Marginal coverage % (`sl_test/ic`)

| Dataset | $L$ | $O$ | $k$ | EMForecaster (2025) | DLinear (2023) | Transformer (2017) | LSTM (1997) |
|:---|---:|---:|---:|---:|---:|---:|---:|
| ED2022 | 336 | 96 | 1 | - | 87.37 | 85.08 | 84.95 |
| ED2022 | 336 | 96 | 9 | 85.43 | 84.08 | 85.74 | 85.33 |
| UH2022 | 336 | 96 | 1 | - | 82.12 | 85.95 | 87.15 |
| UH2022 | 336 | 96 | 9 | 86.92 | 85.76 | 86.21 | 85.97 |
| EMF2023 (University-5 w/ PTV) | 336 | 96 | 1 | - | 94.76 | 90.87 | 90.55 |
| EMF2023 (University-5 w/ PTV) | 336 | 96 | 9 | 90.97 | 91.29 | 90.15 | 90.33 |

### 3.2 · Joint coverage % (`sl_test/jc`)

| Dataset | $L$ | $O$ | $k$ | EMForecaster (2025) | DLinear (2023) | Transformer (2017) | LSTM (1997) |
|:---|---:|---:|---:|---:|---:|---:|---:|
| ED2022 | 336 | 96 | 1 | - | 5.25 | 14.69 | 15.19 |
| ED2022 | 336 | 96 | 9 | 12.45 | 12.09 | 12.82 | 15.01 |
| UH2022 | 336 | 96 | 1 | - | 1.07 | 9.24 | 9.37 |
| UH2022 | 336 | 96 | 9 | 7.65 | 4.75 | 7.04 | 8.05 |
| EMF2023 (University-5 w/ PTV) | 336 | 96 | 1 | - | 19.27 | 16.39 | 16.42 |
| EMF2023 (University-5 w/ PTV) | 336 | 96 | 9 | 14.60 | 19.16 | 11.80 | 11.64 |

### 3.3 · Mean interval width (`sl_test/interval_width_mean`)

| Dataset | $L$ | $O$ | $k$ | EMForecaster (2025) | DLinear (2023) | Transformer (2017) | LSTM (1997) |
|:---|---:|---:|---:|---:|---:|---:|---:|
| ED2022 | 336 | 96 | 1 | - | 1.917 | 1.531 | 1.510 |
| ED2022 | 336 | 96 | 9 | 1.367 | 1.643 | 1.368 | 1.530 |
| UH2022 | 336 | 96 | 1 | - | 2.112 | 1.785 | 1.768 |
| UH2022 | 336 | 96 | 9 | 1.728 | 1.910 | 1.693 | 1.766 |
| EMF2023 (University-5 w/ PTV) | 336 | 96 | 1 | - | 2.863 | 2.520 | 2.218 |
| EMF2023 (University-5 w/ PTV) | 336 | 96 | 9 | 2.109 | 2.690 | 1.917 | 1.909 |

## 4 · Model size

### 4.1 · Trainable weights (`parameters/sl/num_params`)

For multivariate logs this varies by dataset (different channel counts), not just by (model, k, L, O); shown here is the first value found per (model, k, L, O).

| Model | L336/O96 |
|:---|---:|
| EMForecaster (2025) (k=9) | 283,175,679 |
| DLinear (2023) (k=1) | 47,034,432 |
| DLinear (2023) (k=9) | 8,843,328 |
| Transformer (2017) (k=1) | 23,857,602 |
| Transformer (2017) (k=9) | 4,762,050 |
| LSTM (1997) (k=1) | 2,259,638 |
| LSTM (1997) (k=9) | 441,014 |

# Multivariate vs Univariate

- Compares: Multivariate vs Per-channel
- Matched on: (dataset, L, O, model, conformal setting); only cells present in both trees appear below
- Delta: Δ is the univariate value relative to Multivariate, so negative Δ means the univariate value is lower; **bold** = better of the two where lower is better

## 1 · Forecast accuracy (Multivariate vs Per-channel)

21 matched cells, from `Multivariate` and `Univariate (Per Channel)`.

### 1.1 · MSE (`sl_test/loss`)

| Dataset | $L$ | $O$ | Model | Multivariate | Per-channel | Δ |
|:---|---:|---:|:---|---:|---:|---:|
| ED2022 | 336 | 96 | EMForecaster (2025) | **0.4788** | 0.4848 | +1.2% |
| ED2022 | 336 | 96 | DLinear (2023) | 0.5354 | **0.5321** | -0.6% |
| ED2022 | 336 | 96 | TSMixer (2023) | 0.5526 | **0.5484** | -0.8% |
| ED2022 | 336 | 96 | PatchTST (2023) | 0.5439 | **0.5419** | -0.4% |
| ED2022 | 336 | 96 | Transformer (2017) | **0.4978** | 0.6904 | +38.7% |
| ED2022 | 336 | 96 | LSTM (1997) | **0.5600** | 0.5785 | +3.3% |
| ED2022 | 336 | 96 | ARIMA (1970) | **1.0193** | 1.0194 | 0.0% |
| UH2022 | 336 | 96 | EMForecaster (2025) | **1.1236** | 1.1704 | +4.2% |
| UH2022 | 336 | 96 | DLinear (2023) | **1.1783** | 1.1900 | +1.0% |
| UH2022 | 336 | 96 | TSMixer (2023) | **1.1992** | 1.2932 | +7.8% |
| UH2022 | 336 | 96 | PatchTST (2023) | **1.1741** | 1.2623 | +7.5% |
| UH2022 | 336 | 96 | Transformer (2017) | **1.1414** | 1.2270 | +7.5% |
| UH2022 | 336 | 96 | LSTM (1997) | **1.1887** | 1.2180 | +2.5% |
| UH2022 | 336 | 96 | ARIMA (1970) | **1.6017** | 1.6017 | 0.0% |
| EMF2023 (University-5 w/ PTV) | 336 | 96 | EMForecaster (2025) | **1.4909** | 1.5054 | +1.0% |
| EMF2023 (University-5 w/ PTV) | 336 | 96 | DLinear (2023) | 1.5548 | **1.5163** | -2.5% |
| EMF2023 (University-5 w/ PTV) | 336 | 96 | TSMixer (2023) | **1.4955** | 1.5262 | +2.1% |
| EMF2023 (University-5 w/ PTV) | 336 | 96 | PatchTST (2023) | **1.5147** | 1.5266 | +0.8% |
| EMF2023 (University-5 w/ PTV) | 336 | 96 | Transformer (2017) | **1.5056** | 1.5808 | +5.0% |
| EMF2023 (University-5 w/ PTV) | 336 | 96 | LSTM (1997) | **1.5209** | 1.5435 | +1.5% |
| EMF2023 (University-5 w/ PTV) | 336 | 96 | ARIMA (1970) | 1.8090 | **1.8089** | 0.0% |

### 1.2 · MAE (`sl_test/mae`)

| Dataset | $L$ | $O$ | Model | Multivariate | Per-channel | Δ |
|:---|---:|---:|:---|---:|---:|---:|
| ED2022 | 336 | 96 | EMForecaster (2025) | **0.3346** | 0.3428 | +2.4% |
| ED2022 | 336 | 96 | DLinear (2023) | **0.3771** | 0.3781 | +0.3% |
| ED2022 | 336 | 96 | TSMixer (2023) | **0.3859** | 0.3875 | +0.4% |
| ED2022 | 336 | 96 | PatchTST (2023) | 0.3868 | **0.3817** | -1.3% |
| ED2022 | 336 | 96 | Transformer (2017) | **0.3559** | 0.4429 | +24.4% |
| ED2022 | 336 | 96 | LSTM (1997) | **0.4066** | 0.4160 | +2.3% |
| ED2022 | 336 | 96 | ARIMA (1970) | **0.6075** | 0.6076 | 0.0% |
| UH2022 | 336 | 96 | EMForecaster (2025) | **0.4443** | 0.4585 | +3.2% |
| UH2022 | 336 | 96 | DLinear (2023) | **0.4829** | 0.4851 | +0.5% |
| UH2022 | 336 | 96 | TSMixer (2023) | **0.4835** | 0.4961 | +2.6% |
| UH2022 | 336 | 96 | PatchTST (2023) | **0.4791** | 0.4988 | +4.1% |
| UH2022 | 336 | 96 | Transformer (2017) | **0.4613** | 0.5075 | +10.0% |
| UH2022 | 336 | 96 | LSTM (1997) | **0.4991** | 0.5089 | +2.0% |
| UH2022 | 336 | 96 | ARIMA (1970) | **0.6807** | 0.6807 | 0.0% |
| EMF2023 (University-5 w/ PTV) | 336 | 96 | EMForecaster (2025) | **0.4397** | 0.4500 | +2.3% |
| EMF2023 (University-5 w/ PTV) | 336 | 96 | DLinear (2023) | 0.5036 | **0.4677** | -7.1% |
| EMF2023 (University-5 w/ PTV) | 336 | 96 | TSMixer (2023) | **0.4495** | 0.4591 | +2.1% |
| EMF2023 (University-5 w/ PTV) | 336 | 96 | PatchTST (2023) | **0.4581** | 0.4701 | +2.6% |
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
| ED2022 | 336 | 96 | PatchTST (2023) | 2m 11s | 48s | 27 | 21m 38s | +890.8% |
| ED2022 | 336 | 96 | Transformer (2017) | 43m 10s | 3m 58s | 27 | 1h 47m 7s | +148.1% |
| ED2022 | 336 | 96 | LSTM (1997) | 2m 43s | 50s | 27 | 22m 26s | +725.8% |
| ED2022 | 336 | 96 | ARIMA (1970) | - | - | 27 | - | - |
| UH2022 | 336 | 96 | EMForecaster (2025) | 38m 29s | 1m 53s | 27 | 50m 38s | +31.6% |
| UH2022 | 336 | 96 | DLinear (2023) | 1m 18s | 1m 11s | 27 | 32m 6s | +2369.2% |
| UH2022 | 336 | 96 | TSMixer (2023) | 1m 24s | 1m 29s | 27 | 39m 51s | +2746.4% |
| UH2022 | 336 | 96 | PatchTST (2023) | 2m 38s | 55s | 27 | 24m 44s | +839.2% |
| UH2022 | 336 | 96 | Transformer (2017) | 35m 59s | 5m 1s | 27 | 2h 15m 23s | +276.2% |
| UH2022 | 336 | 96 | LSTM (1997) | 3m 19s | 56s | 27 | 25m 19s | +663.3% |
| UH2022 | 336 | 96 | ARIMA (1970) | - | - | 27 | - | - |
| EMF2023 (University-5 w/ PTV) | 336 | 96 | EMForecaster (2025) | 56m 32s | 3m 38s | 29 | 1h 45m 23s | +86.4% |
| EMF2023 (University-5 w/ PTV) | 336 | 96 | DLinear (2023) | 48s | 1m 40s | 29 | 48m 30s | +5962.5% |
| EMF2023 (University-5 w/ PTV) | 336 | 96 | TSMixer (2023) | 2m 12s | 2m 39s | 29 | 1h 16m 42s | +3386.4% |
| EMF2023 (University-5 w/ PTV) | 336 | 96 | PatchTST (2023) | 4m 43s | 1m 35s | 29 | 45m 55s | +873.5% |
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
| ED2022 | 336 | 96 | PatchTST (2023) | 85.06 | 85.05 | -0.01 pp |
| ED2022 | 336 | 96 | Transformer (2017) | 85.45 | 84.46 | -0.99 pp |
| ED2022 | 336 | 96 | LSTM (1997) | 85.31 | 85.11 | -0.20 pp |
| ED2022 | 336 | 96 | ARIMA (1970) | 86.43 | 86.43 | 0.00 pp |
| UH2022 | 336 | 96 | EMForecaster (2025) | 86.94 | 86.71 | -0.22 pp |
| UH2022 | 336 | 96 | DLinear (2023) | 87.16 | 86.24 | -0.92 pp |
| UH2022 | 336 | 96 | TSMixer (2023) | 87.01 | 86.70 | -0.31 pp |
| UH2022 | 336 | 96 | PatchTST (2023) | 86.77 | 85.97 | -0.80 pp |
| UH2022 | 336 | 96 | Transformer (2017) | 86.72 | 85.78 | -0.94 pp |
| UH2022 | 336 | 96 | LSTM (1997) | 86.68 | 85.82 | -0.86 pp |
| UH2022 | 336 | 96 | ARIMA (1970) | 84.27 | 84.28 | 0.00 pp |
| EMF2023 (University-5 w/ PTV) | 336 | 96 | EMForecaster (2025) | 90.31 | 90.46 | +0.15 pp |
| EMF2023 (University-5 w/ PTV) | 336 | 96 | DLinear (2023) | 91.06 | 90.76 | -0.31 pp |
| EMF2023 (University-5 w/ PTV) | 336 | 96 | TSMixer (2023) | 91.13 | 91.33 | +0.20 pp |
| EMF2023 (University-5 w/ PTV) | 336 | 96 | PatchTST (2023) | 90.99 | 90.66 | -0.33 pp |
| EMF2023 (University-5 w/ PTV) | 336 | 96 | Transformer (2017) | 90.30 | 90.57 | +0.27 pp |
| EMF2023 (University-5 w/ PTV) | 336 | 96 | LSTM (1997) | 90.94 | 90.05 | -0.89 pp |
| EMF2023 (University-5 w/ PTV) | 336 | 96 | ARIMA (1970) | 88.65 | 88.65 | 0.00 pp |

### 3.2 · Joint coverage % (`sl_test/jc`)

| Dataset | $L$ | $O$ | Model | Multivariate | Per-channel | Δ |
|:---|---:|---:|:---|---:|---:|---:|
| ED2022 | 336 | 96 | EMForecaster (2025) | 11.47 | 10.50 | -0.98 pp |
| ED2022 | 336 | 96 | DLinear (2023) | 13.08 | 11.10 | -1.99 pp |
| ED2022 | 336 | 96 | TSMixer (2023) | 13.64 | 12.99 | -0.64 pp |
| ED2022 | 336 | 96 | PatchTST (2023) | 14.09 | 12.89 | -1.20 pp |
| ED2022 | 336 | 96 | Transformer (2017) | 12.97 | 14.80 | +1.83 pp |
| ED2022 | 336 | 96 | LSTM (1997) | 14.58 | 14.14 | -0.44 pp |
| ED2022 | 336 | 96 | ARIMA (1970) | 21.65 | 21.65 | 0.00 pp |
| UH2022 | 336 | 96 | EMForecaster (2025) | 8.71 | 6.87 | -1.83 pp |
| UH2022 | 336 | 96 | DLinear (2023) | 9.07 | 6.60 | -2.47 pp |
| UH2022 | 336 | 96 | TSMixer (2023) | 8.56 | 7.59 | -0.96 pp |
| UH2022 | 336 | 96 | PatchTST (2023) | 9.63 | 8.91 | -0.72 pp |
| UH2022 | 336 | 96 | Transformer (2017) | 9.22 | 7.92 | -1.30 pp |
| UH2022 | 336 | 96 | LSTM (1997) | 10.00 | 8.64 | -1.36 pp |
| UH2022 | 336 | 96 | ARIMA (1970) | 9.66 | 9.66 | 0.00 pp |
| EMF2023 (University-5 w/ PTV) | 336 | 96 | EMForecaster (2025) | 10.67 | 11.50 | +0.83 pp |
| EMF2023 (University-5 w/ PTV) | 336 | 96 | DLinear (2023) | 13.81 | 11.87 | -1.94 pp |
| EMF2023 (University-5 w/ PTV) | 336 | 96 | TSMixer (2023) | 13.80 | 14.82 | +1.02 pp |
| EMF2023 (University-5 w/ PTV) | 336 | 96 | PatchTST (2023) | 12.87 | 13.64 | +0.78 pp |
| EMF2023 (University-5 w/ PTV) | 336 | 96 | Transformer (2017) | 12.21 | 14.61 | +2.40 pp |
| EMF2023 (University-5 w/ PTV) | 336 | 96 | LSTM (1997) | 13.00 | 12.02 | -0.98 pp |
| EMF2023 (University-5 w/ PTV) | 336 | 96 | ARIMA (1970) | 13.06 | 13.06 | 0.00 pp |

### 3.3 · Mean interval width (`sl_test/interval_width_mean`)

| Dataset | $L$ | $O$ | Model | Multivariate | Per-channel | Δ |
|:---|---:|---:|:---|---:|---:|---:|
| ED2022 | 336 | 96 | EMForecaster (2025) | **1.273** | 1.314 | +3.2% |
| ED2022 | 336 | 96 | DLinear (2023) | 1.448 | **1.434** | -1.0% |
| ED2022 | 336 | 96 | TSMixer (2023) | 1.516 | **1.516** | 0.0% |
| ED2022 | 336 | 96 | PatchTST (2023) | 1.480 | **1.451** | -2.0% |
| ED2022 | 336 | 96 | Transformer (2017) | **1.366** | 1.678 | +22.8% |
| ED2022 | 336 | 96 | LSTM (1997) | **1.570** | 1.582 | +0.7% |
| ED2022 | 336 | 96 | ARIMA (1970) | 2.460 | **2.460** | 0.0% |
| UH2022 | 336 | 96 | EMForecaster (2025) | **1.655** | 1.683 | +1.7% |
| UH2022 | 336 | 96 | DLinear (2023) | 1.765 | **1.710** | -3.1% |
| UH2022 | 336 | 96 | TSMixer (2023) | **1.789** | 1.793 | +0.3% |
| UH2022 | 336 | 96 | PatchTST (2023) | **1.770** | 1.774 | +0.2% |
| UH2022 | 336 | 96 | Transformer (2017) | **1.716** | 1.845 | +7.5% |
| UH2022 | 336 | 96 | LSTM (1997) | **1.805** | 1.812 | +0.4% |
| UH2022 | 336 | 96 | ARIMA (1970) | **2.334** | 2.334 | 0.0% |
| EMF2023 (University-5 w/ PTV) | 336 | 96 | EMForecaster (2025) | **1.862** | 1.941 | +4.2% |
| EMF2023 (University-5 w/ PTV) | 336 | 96 | DLinear (2023) | 2.095 | **1.935** | -7.6% |
| EMF2023 (University-5 w/ PTV) | 336 | 96 | TSMixer (2023) | **1.991** | 2.202 | +10.6% |
| EMF2023 (University-5 w/ PTV) | 336 | 96 | PatchTST (2023) | 2.006 | **2.001** | -0.3% |
| EMF2023 (University-5 w/ PTV) | 336 | 96 | Transformer (2017) | **1.903** | 2.112 | +11.0% |
| EMF2023 (University-5 w/ PTV) | 336 | 96 | LSTM (1997) | **1.988** | 1.996 | +0.4% |
| EMF2023 (University-5 w/ PTV) | 336 | 96 | ARIMA (1970) | 2.446 | **2.446** | 0.0% |

## 4 · Scoreboard

### 4.1 · Matched-cell wins by metric

How many matched cells each pipeline wins, counted only where lower is better.

| Comparison | Metric | Multivariate | Univariate | Tie |
|:---|:---|---:|---:|---:|
| Multivariate vs Per-channel | MSE | 16 | 5 | 0 |
| Multivariate vs Per-channel | MAE | 18 | 3 | 0 |
| Multivariate vs Per-channel | Interval width | 13 | 8 | 0 |

# Ultimate Comparison

- Configuration: L336/O96 only -- the one window every pipeline shares
- Pipelines: Per-channel · Cluster (k=1) · Cluster (k=9) · Multivariate
- Datasets: ED2022, UH2022, EMF2023 (University-5 w/ PTV)
- Models: EMForecaster (2025), DLinear (2023), TSMixer (2023), PatchTST (2023), Transformer (2017), LSTM (1997), ARIMA (1970)
- Caveat: `Cluster (k=1)` is not a re-run of `Multivariate` -- it still replaces the model's own head with one channel-mixing ClusteredHead, which changes the weight count (see each tree's Model size table). Every row is a whole strategy, not the same network under a different loader.
- Conformal prediction: on for every run

## 1 · Every pipeline side by side

Read **across** a row to rank the models under one pipeline (**bold** = best, <u>underline</u> = runner-up); read **down** a dataset block to see what the same model does per channel, jointly, or with a k-cluster head. `-` = that combination was not run (the cluster grid only covers the four models that have a head to cluster).

### 1.1 · MSE (`sl_test/loss`)

| Dataset | Pipeline | EMForecaster (2025) | DLinear (2023) | TSMixer (2023) | PatchTST (2023) | Transformer (2017) | LSTM (1997) | ARIMA (1970) |
|:---|:---|---:|---:|---:|---:|---:|---:|---:|
| ED2022 | Per-channel | **0.4848** | <u>0.5321</u> | 0.5484 | 0.5419 | 0.6904 | 0.5785 | 1.0194 |
| ED2022 | Cluster (k=1) | - | 0.6641 | - | - | <u>0.5541</u> | **0.5449** | - |
| ED2022 | Cluster (k=9) | <u>0.4905</u> | 0.5923 | - | - | **0.4873** | 0.5499 | - |
| ED2022 | Multivariate | **0.4788** | 0.5354 | 0.5526 | 0.5439 | <u>0.4978</u> | 0.5600 | 1.0193 |
| UH2022 | Per-channel | **1.1704** | <u>1.1900</u> | 1.2932 | 1.2623 | 1.2270 | 1.2180 | 1.6017 |
| UH2022 | Cluster (k=1) | - | 1.4913 | - | - | <u>1.3022</u> | **1.2336** | - |
| UH2022 | Cluster (k=9) | <u>1.1838</u> | 1.3412 | - | - | **1.1533** | 1.1938 | - |
| UH2022 | Multivariate | **1.1236** | 1.1783 | 1.1992 | 1.1741 | <u>1.1414</u> | 1.1887 | 1.6017 |
| EMF2023 (University-5 w/ PTV) | Per-channel | **1.5054** | <u>1.5163</u> | 1.5262 | 1.5266 | 1.5808 | 1.5435 | 1.8089 |
| EMF2023 (University-5 w/ PTV) | Cluster (k=1) | - | 1.7097 | - | - | <u>1.6641</u> | **1.5251** | - |
| EMF2023 (University-5 w/ PTV) | Cluster (k=9) | 1.5190 | 1.6135 | - | - | **1.5091** | <u>1.5132</u> | - |
| EMF2023 (University-5 w/ PTV) | Multivariate | **1.4909** | 1.5548 | <u>1.4955</u> | 1.5147 | 1.5056 | 1.5209 | 1.8090 |

### 1.2 · MAE (`sl_test/mae`)

| Dataset | Pipeline | EMForecaster (2025) | DLinear (2023) | TSMixer (2023) | PatchTST (2023) | Transformer (2017) | LSTM (1997) | ARIMA (1970) |
|:---|:---|---:|---:|---:|---:|---:|---:|---:|
| ED2022 | Per-channel | **0.3428** | <u>0.3781</u> | 0.3875 | 0.3817 | 0.4429 | 0.4160 | 0.6076 |
| ED2022 | Cluster (k=1) | - | 0.5021 | - | - | <u>0.3956</u> | **0.3949** | - |
| ED2022 | Cluster (k=9) | <u>0.3505</u> | 0.4254 | - | - | **0.3480** | 0.3967 | - |
| ED2022 | Multivariate | **0.3346** | 0.3771 | 0.3859 | 0.3868 | <u>0.3559</u> | 0.4066 | 0.6075 |
| UH2022 | Per-channel | **0.4585** | <u>0.4851</u> | 0.4961 | 0.4988 | 0.5075 | 0.5089 | 0.6807 |
| UH2022 | Cluster (k=1) | - | 0.6552 | - | - | <u>0.5179</u> | **0.4947** | - |
| UH2022 | Cluster (k=9) | <u>0.4688</u> | 0.5568 | - | - | **0.4655** | 0.4916 | - |
| UH2022 | Multivariate | **0.4443** | 0.4829 | 0.4835 | 0.4791 | <u>0.4613</u> | 0.4991 | 0.6807 |
| EMF2023 (University-5 w/ PTV) | Per-channel | **0.4500** | 0.4677 | <u>0.4591</u> | 0.4701 | 0.5006 | 0.4795 | 0.6212 |
| EMF2023 (University-5 w/ PTV) | Cluster (k=1) | - | 0.5965 | - | - | <u>0.5562</u> | **0.4652** | - |
| EMF2023 (University-5 w/ PTV) | Cluster (k=9) | 0.4629 | 0.5033 | - | - | **0.4555** | <u>0.4590</u> | - |
| EMF2023 (University-5 w/ PTV) | Multivariate | **0.4397** | 0.5036 | <u>0.4495</u> | 0.4581 | 0.4526 | 0.4675 | 0.6213 |

### 1.3 · Total training time (`parameters/running_time`)

Cost of the whole strategy, so the per-channel row is the mean single-channel run times the channel count (ED2022 C=27, UH2022 C=27, EMF2023 (University-5 w/ PTV) C=29); every other row is one model's single training run. ARIMA is not gradient-trained and logs no time.

| Dataset | Pipeline | EMForecaster (2025) | DLinear (2023) | TSMixer (2023) | PatchTST (2023) | Transformer (2017) | LSTM (1997) | ARIMA (1970) |
|:---|:---|---:|---:|---:|---:|---:|---:|---:|
| ED2022 | Per-channel | 47m 32s | 25m 22s | 31m 46s | **21m 38s** | 1h 47m 7s | <u>22m 26s</u> | - |
| ED2022 | Cluster (k=1) | - | **1m 9s** | - | - | 13m 44s | <u>2m 8s</u> | - |
| ED2022 | Cluster (k=9) | 7m 10s | **1m 2s** | - | - | 23m 53s | <u>1m 31s</u> | - |
| ED2022 | Multivariate | 35m 18s | **1m 6s** | <u>1m 11s</u> | 2m 11s | 43m 10s | 2m 43s | - |
| UH2022 | Per-channel | 50m 38s | 32m 6s | 39m 51s | **24m 44s** | 2h 15m 23s | <u>25m 19s</u> | - |
| UH2022 | Cluster (k=1) | - | **1m 21s** | - | - | 13m 53s | <u>2m 4s</u> | - |
| UH2022 | Cluster (k=9) | 9m 0s | **1m 14s** | - | - | 16m 15s | <u>1m 50s</u> | - |
| UH2022 | Multivariate | 38m 29s | **1m 18s** | <u>1m 24s</u> | 2m 38s | 35m 59s | 3m 19s | - |
| EMF2023 (University-5 w/ PTV) | Per-channel | 1h 45m 23s | 48m 30s | 1h 16m 42s | <u>45m 55s</u> | 4h 18m 39s | **43m 19s** | - |
| EMF2023 (University-5 w/ PTV) | Cluster (k=1) | - | **2m 27s** | - | - | 13m 1s | <u>3m 15s</u> | - |
| EMF2023 (University-5 w/ PTV) | Cluster (k=9) | 17m 56s | **2m 5s** | - | - | 33m 1s | <u>3m 17s</u> | - |
| EMF2023 (University-5 w/ PTV) | Multivariate | 56m 32s | **48s** | <u>2m 12s</u> | 4m 43s | 1h 44m 16s | 5m 59s | - |
