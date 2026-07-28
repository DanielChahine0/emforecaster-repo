# Multivariate

---

## 2 · Point-Forecasting Performance (with conformal prediction)

### Table A | MSE (`sl_test/loss`)

**Bold** = best (lowest) value in the row, among populated cells.

| Dataset | $L$ | $O$ | EMForecaster (2025) | DLinear (2023) | TSMixer (2023) | PatchTST (2023) | Transformer (2017) | LSTM (1997) | ARIMA (1970) |
|---|---|---|---|---|---|---|---|---|---|
| ED2022 | 192 | 96 | **0.4866** | 0.5358 | 0.5858 | 0.5996 | 0.4976 | 0.5579 | 1.0194 |
| ED2022 | 192 | 336 | **0.5526** | 0.6048 | 0.5989 | 0.8315 | 0.9551 | 0.6067 | 1.1384 |
| ED2022 | 336 | 96 | **0.4788** | 0.5354 | 0.5525 | 0.5444 | 0.5027 | 0.5600 | 1.0193 |
| ED2022 | 336 | 336 | **0.5406** | 0.6002 | 0.6002 | 0.6326 | 0.9826 | 0.6335 | 1.1381 |
| UH2022 | 192 | 96 | **1.1322** | 1.1903 | 1.2432 | 1.1852 | 1.1471 | 1.1802 | 1.6017 |
| UH2022 | 192 | 336 | **1.1822** | 1.2456 | 1.2298 | 1.2944 | 1.5511 | 1.2128 | 1.7232 |
| UH2022 | 336 | 96 | **1.1236** | 1.1783 | 1.1992 | 1.1717 | 1.1432 | 1.1887 | 1.6017 |
| UH2022 | 336 | 336 | **1.1706** | 1.2276 | 1.2132 | 1.2112 | 1.5720 | 1.2077 | 1.7231 |
| EMF_2023_UNIVERSITY_5_WITH_PTV | 192 | 96 | **1.4979** | 1.5651 | 1.5096 | 1.5247 | 1.5095 | 1.5196 | 1.7990 |
| EMF_2023_UNIVERSITY_5_WITH_PTV | 192 | 336 | 0.6144 | 0.6625 | **0.6122** | 0.6576 | 0.8630 | 0.6273 | 0.9350 |
| EMF_2023_UNIVERSITY_5_WITH_PTV | 336 | 96 | **1.4909** | 1.5548 | 1.4955 | 1.5147 | 1.5080 | 1.5209 | 1.8090 |
| EMF_2023_UNIVERSITY_5_WITH_PTV | 336 | 336 | **0.5963** | 0.6406 | 0.5982 | 0.6109 | 0.8568 | 0.6087 | 0.9302 |

### Table A2 | MAE (`sl_test/mae`)

| Dataset | $L$ | $O$ | EMForecaster (2025) | DLinear (2023) | TSMixer (2023) | PatchTST (2023) | Transformer (2017) | LSTM (1997) | ARIMA (1970) |
|---|---|---|---|---|---|---|---|---|---|
| ED2022 | 192 | 96 | **0.3422** | 0.3805 | 0.4000 | 0.4257 | 0.3531 | 0.3985 | 0.6081 |
| ED2022 | 192 | 336 | **0.3741** | 0.4125 | 0.4029 | 0.5490 | 0.6025 | 0.4167 | 0.6722 |
| ED2022 | 336 | 96 | **0.3346** | 0.3771 | 0.3858 | 0.3870 | 0.3592 | 0.4066 | 0.6075 |
| ED2022 | 336 | 336 | **0.3684** | 0.4064 | 0.4043 | 0.4375 | 0.6134 | 0.4373 | 0.6714 |
| UH2022 | 192 | 96 | **0.4495** | 0.4896 | 0.4904 | 0.4852 | 0.4604 | 0.4859 | 0.6809 |
| UH2022 | 192 | 336 | **0.4714** | 0.5153 | 0.4959 | 0.5502 | 0.6686 | 0.4946 | 0.7326 |
| UH2022 | 336 | 96 | **0.4443** | 0.4829 | 0.4835 | 0.4777 | 0.4614 | 0.4991 | 0.6807 |
| UH2022 | 336 | 336 | **0.4645** | 0.5050 | 0.4890 | 0.4972 | 0.6782 | 0.4940 | 0.7323 |
| EMF_2023_UNIVERSITY_5_WITH_PTV | 192 | 96 | **0.4467** | 0.5078 | 0.4575 | 0.4718 | 0.4555 | 0.4647 | 0.6226 |
| EMF_2023_UNIVERSITY_5_WITH_PTV | 192 | 336 | **0.4544** | 0.5082 | 0.4603 | 0.4926 | 0.6035 | 0.4672 | 0.6442 |
| EMF_2023_UNIVERSITY_5_WITH_PTV | 336 | 96 | **0.4397** | 0.5036 | 0.4495 | 0.4581 | 0.4581 | 0.4675 | 0.6213 |
| EMF_2023_UNIVERSITY_5_WITH_PTV | 336 | 336 | **0.4474** | 0.4977 | 0.4512 | 0.4589 | 0.6032 | 0.4618 | 0.6431 |

### Table B | Training Time (`parameters/running_time`)

**Bold** = fastest in the row. `X` = not logged (e.g. ARIMA is not gradient-trained).

| Dataset | $L$ | $O$ | EMForecaster (2025) | DLinear (2023) | TSMixer (2023) | PatchTST (2023) | Transformer (2017) | LSTM (1997) | ARIMA (1970) |
|---|---|---|---|---|---|---|---|---|---|
| ED2022 | 192 | 96 | 10m 10s | **27s** | 27s | 2m 26s | 6m 13s | 56s | X |
| ED2022 | 192 | 336 | 9m 2s | **46s** | 46s | 1m 20s | 3m 56s | 1m 6s | X |
| ED2022 | 336 | 96 | 14m 12s | **38s** | 41s | 1m 48s | 19m 33s | 1m 17s | X |
| ED2022 | 336 | 336 | 10m 7s | **52s** | 56s | 1m 52s | 13m 45s | 1m 24s | X |
| UH2022 | 192 | 96 | 12m 51s | **36s** | 36s | 2m 30s | 7m 45s | 1m 7s | X |
| UH2022 | 192 | 336 | 9m 7s | **58s** | 1m 15s | 1m 35s | 6m 50s | 1m 19s | X |
| UH2022 | 336 | 96 | 14m 34s | **47s** | 1m 7s | 2m 15s | 15m 15s | 1m 33s | X |
| UH2022 | 336 | 336 | 13m 46s | **1m 9s** | 1m 28s | 2m 21s | 10m 8s | 1m 42s | X |
| EMF_2023_UNIVERSITY_5_WITH_PTV | 192 | 96 | 21m 4s | **38s** | 1m 30s | 5m 50s | 15m 12s | 4m 7s | X |
| EMF_2023_UNIVERSITY_5_WITH_PTV | 192 | 336 | 25m 21s | **52s** | 2m 15s | 5m 33s | 16m 26s | 5m 12s | X |
| EMF_2023_UNIVERSITY_5_WITH_PTV | 336 | 96 | 21m 42s | **46s** | 1m 54s | 4m 53s | 36m 15s | 4m 0s | X |
| EMF_2023_UNIVERSITY_5_WITH_PTV | 336 | 336 | 29m 14s | **1m 0s** | 1m 58s | 5m 4s | 36m 16s | 3m 9s | X |

### Table C | Per-Epoch Training Time (`running_time / len(sl_train/loss)`, per run)

| Dataset | $L$ | $O$ | EMForecaster (2025) | DLinear (2023) | TSMixer (2023) | PatchTST (2023) | Transformer (2017) | LSTM (1997) | ARIMA (1970) |
|---|---|---|---|---|---|---|---|---|---|
| ED2022 | 192 | 96 | 6.1s | **0.3s** | 0.3s | 1.5s | 4.8s | 0.6s | X |
| ED2022 | 192 | 336 | 6.1s | **0.5s** | 0.5s | 0.8s | 6.4s | 0.7s | X |
| ED2022 | 336 | 96 | 8.5s | **0.4s** | 0.4s | 1.1s | 11.7s | 0.8s | X |
| ED2022 | 336 | 336 | 8.1s | **0.5s** | 0.6s | 1.1s | 15.3s | 0.8s | X |
| UH2022 | 192 | 96 | 7.7s | **0.4s** | 0.4s | 1.5s | 5.9s | 0.7s | X |
| UH2022 | 192 | 336 | 7.7s | **0.6s** | 0.8s | 0.9s | 8.5s | 0.8s | X |
| UH2022 | 336 | 96 | 10.2s | **0.5s** | 0.7s | 1.4s | 14.8s | 0.9s | X |
| UH2022 | 336 | 336 | 10.3s | **0.7s** | 0.9s | 1.4s | 19.0s | 1.0s | X |
| EMF_2023_UNIVERSITY_5_WITH_PTV | 192 | 96 | 14.7s | **0.9s** | 0.9s | 3.5s | 11.3s | 2.5s | X |
| EMF_2023_UNIVERSITY_5_WITH_PTV | 192 | 336 | 15.2s | **1.2s** | 1.4s | 3.3s | 16.4s | 3.2s | X |
| EMF_2023_UNIVERSITY_5_WITH_PTV | 336 | 96 | 19.4s | **1.1s** | 1.1s | 2.9s | 29.0s | 2.4s | X |
| EMF_2023_UNIVERSITY_5_WITH_PTV | 336 | 336 | 19.9s | **1.4s** | 1.5s | 3.0s | 37.5s | 1.9s | X |

#### Per-model summary

| Model | Mean / epoch | Estimation for 100 ep |
|---|---:|---:|
| DLinear (2023) | 0.7s | ~1m 10s |
| TSMixer (2023) | 0.8s | ~1m 17s |
| LSTM (1997) | 1.3s | ~2m 15s |
| PatchTST (2023) | 1.9s | ~3m 7s |
| EMForecaster (2025) | 11.2s | ~18m 36s |
| Transformer (2017) | 15.0s | ~25m 5s |

### Table D | Marginal Coverage % (`sl_test/ic`)

Not bolded -- "best" means closest to the target coverage, not highest.

| Dataset | $L$ | $O$ | EMForecaster (2025) | DLinear (2023) | TSMixer (2023) | PatchTST (2023) | Transformer (2017) | LSTM (1997) | ARIMA (1970) |
|---|---|---|---|---|---|---|---|---|---|
| ED2022 | 192 | 96 | 85.00 | 85.20 | 85.25 | 85.32 | 85.54 | 85.14 | 86.40 |
| ED2022 | 192 | 336 | 84.88 | 84.96 | 84.93 | 85.32 | 85.62 | 84.85 | 86.30 |
| ED2022 | 336 | 96 | 84.75 | 84.89 | 85.39 | 85.04 | 85.36 | 85.31 | 86.43 |
| ED2022 | 336 | 336 | 84.57 | 84.66 | 84.96 | 84.91 | 85.74 | 85.09 | 86.34 |
| UH2022 | 192 | 96 | 86.80 | 87.02 | 86.82 | 86.72 | 86.77 | 86.88 | 84.23 |
| UH2022 | 192 | 336 | 86.77 | 86.80 | 86.69 | 86.21 | 85.09 | 86.66 | 83.75 |
| UH2022 | 336 | 96 | 86.94 | 87.16 | 87.01 | 86.84 | 86.72 | 86.68 | 84.27 |
| UH2022 | 336 | 336 | 86.97 | 86.96 | 87.07 | 86.98 | 85.46 | 87.04 | 83.80 |
| EMF_2023_UNIVERSITY_5_WITH_PTV | 192 | 96 | 90.12 | 90.76 | 90.74 | 91.14 | 90.21 | 90.34 | 88.59 |
| EMF_2023_UNIVERSITY_5_WITH_PTV | 192 | 336 | 90.61 | 90.82 | 90.90 | 90.93 | 90.86 | 90.46 | 87.98 |
| EMF_2023_UNIVERSITY_5_WITH_PTV | 336 | 96 | 90.31 | 91.06 | 91.13 | 90.99 | 90.58 | 90.94 | 88.65 |
| EMF_2023_UNIVERSITY_5_WITH_PTV | 336 | 336 | 91.07 | 91.05 | 91.57 | 90.99 | 90.90 | 91.15 | 88.05 |

### Table E | Joint Coverage % (`sl_test/jc`)

| Dataset | $L$ | $O$ | EMForecaster (2025) | DLinear (2023) | TSMixer (2023) | PatchTST (2023) | Transformer (2017) | LSTM (1997) | ARIMA (1970) |
|---|---|---|---|---|---|---|---|---|---|
| ED2022 | 192 | 96 | 12.33 | 12.72 | 12.49 | 13.79 | 13.63 | 13.76 | 21.70 |
| ED2022 | 192 | 336 | 3.46 | 3.18 | 3.83 | 4.66 | 4.46 | 3.68 | 4.71 |
| ED2022 | 336 | 96 | 11.47 | 13.08 | 13.63 | 14.05 | 12.86 | 14.58 | 21.65 |
| ED2022 | 336 | 336 | 3.62 | 3.94 | 4.93 | 4.33 | 5.49 | 3.95 | 4.66 |
| UH2022 | 192 | 96 | 8.91 | 8.30 | 9.17 | 10.05 | 9.04 | 9.32 | 9.61 |
| UH2022 | 192 | 336 | 1.52 | 0.81 | 1.45 | 1.90 | 1.15 | 1.21 | 0.74 |
| UH2022 | 336 | 96 | 8.71 | 9.07 | 8.56 | 9.67 | 8.98 | 10.00 | 9.66 |
| UH2022 | 336 | 336 | 1.29 | 1.23 | 1.32 | 2.07 | 1.93 | 1.36 | 0.75 |
| EMF_2023_UNIVERSITY_5_WITH_PTV | 192 | 96 | 9.71 | 13.81 | 13.09 | 13.55 | 11.18 | 12.17 | 13.08 |
| EMF_2023_UNIVERSITY_5_WITH_PTV | 192 | 336 | 2.15 | 3.48 | 3.66 | 3.09 | 2.97 | 1.61 | 2.20 |
| EMF_2023_UNIVERSITY_5_WITH_PTV | 336 | 96 | 10.67 | 13.81 | 13.80 | 12.87 | 13.43 | 13.00 | 13.06 |
| EMF_2023_UNIVERSITY_5_WITH_PTV | 336 | 336 | 2.70 | 4.01 | 4.67 | 2.68 | 2.96 | 2.43 | 2.20 |

### Table F | Mean Interval Width (`sl_test/interval_width_mean`)

Not bolded -- narrower is only better when the target coverage (Table D) is met.

| Dataset | $L$ | $O$ | EMForecaster (2025) | DLinear (2023) | TSMixer (2023) | PatchTST (2023) | Transformer (2017) | LSTM (1997) | ARIMA (1970) |
|---|---|---|---|---|---|---|---|---|---|
| ED2022 | 192 | 96 | 1.302 | 1.463 | 1.551 | 1.642 | 1.358 | 1.516 | 2.454 |
| ED2022 | 192 | 336 | 1.440 | 1.606 | 1.575 | 2.126 | 2.344 | 1.581 | 2.620 |
| ED2022 | 336 | 96 | 1.273 | 1.448 | 1.516 | 1.480 | 1.370 | 1.570 | 2.460 |
| ED2022 | 336 | 336 | 1.414 | 1.595 | 1.606 | 1.665 | 2.407 | 1.674 | 2.624 |
| UH2022 | 192 | 96 | 1.674 | 1.787 | 1.825 | 1.794 | 1.717 | 1.785 | 2.334 |
| UH2022 | 192 | 336 | 1.780 | 1.865 | 1.846 | 1.944 | 2.368 | 1.831 | 2.475 |
| UH2022 | 336 | 96 | 1.655 | 1.765 | 1.789 | 1.768 | 1.716 | 1.805 | 2.334 |
| UH2022 | 336 | 336 | 1.758 | 1.838 | 1.836 | 1.844 | 2.395 | 1.832 | 2.476 |
| EMF_2023_UNIVERSITY_5_WITH_PTV | 192 | 96 | 1.857 | 2.076 | 1.982 | 2.056 | 1.904 | 1.940 | 2.445 |
| EMF_2023_UNIVERSITY_5_WITH_PTV | 192 | 336 | 1.957 | 2.097 | 2.034 | 2.083 | 2.589 | 1.976 | 2.471 |
| EMF_2023_UNIVERSITY_5_WITH_PTV | 336 | 96 | 1.862 | 2.095 | 1.991 | 2.006 | 1.952 | 1.988 | 2.446 |
| EMF_2023_UNIVERSITY_5_WITH_PTV | 336 | 336 | 1.988 | 2.097 | 2.105 | 2.008 | 2.570 | 1.990 | 2.471 |

Every cell in this section averages **1** replicate run(s).

---

## Model sizes

Weights come from `parameters/sl/num_params` in each log. For multivariate logs this can vary by dataset (different channel counts), not just by (model, L, O); this table shows the first value found per (model, L, O) -- cross-check if that matters for your use case.

| Model | L192 / O96 | L192 / O336 | L336 / O96 | L336 / O336 |
|---|---:|---:|---:|---:|
| EMForecaster (2025) | 1,495,287 | 4,444,647 | 2,416,959 | 7,578,159 |
| DLinear (2023) | 37,056 | 129,696 | 64,704 | 226,464 |
| TSMixer (2023) | 32,881 | 79,201 | 56,065 | 136,945 |
| PatchTST (2023) | 44,246 | 136,646 | 72,182 | 233,702 |
| Transformer (2017) | 130,530 | 179,250 | 372,738 | 456,018 |
| LSTM (1997) | 20,726 | 28,646 | 20,726 | 28,646 |

# Univariate

---

## 2 · Point-Forecasting Performance (with conformal prediction)

### Table A | MSE (`sl_test/loss`)

**Bold** = best (lowest) value in the row, among populated cells.

| Dataset | $L$ | $O$ | EMForecaster (2025) | DLinear (2023) | TSMixer (2023) | PatchTST (2023) | Transformer (2017) |
|---|---|---|---|---|---|---|---|
| DET | 192 | 96 | **0.3135** | 0.4520 | 0.3625 | 0.3988 | 0.6018 |
| DET | 192 | 336 | **0.4261** | 0.5301 | 0.4602 | 0.5246 | 1.4640 |
| DET | 336 | 96 | **0.3039** | 0.3997 | 0.3578 | 0.3988 | 0.5970 |
| DET | 336 | 336 | **0.3950** | 0.5044 | 0.4585 | 0.5129 | 1.4857 |
| PTV | 192 | 96 | **0.2610** | 0.4703 | 0.3135 | 0.3564 | 0.6332 |
| PTV | 192 | 336 | **0.3224** | 0.4739 | 0.3672 | 0.4157 | 1.3922 |
| PTV | 336 | 96 | **0.2359** | 0.3547 | 0.3084 | 0.3501 | 0.6799 |
| PTV | 336 | 336 | **0.3181** | 0.4252 | 0.3838 | 0.3909 | 0.5751 |
| PTV2 | 192 | 96 | **0.3199** | 0.4003 | 0.3339 | 0.3658 | 0.3642 |
| PTV2 | 192 | 336 | **0.3574** | 0.4130 | 0.3601 | 0.4146 | 1.1660 |
| PTV2 | 336 | 96 | **0.2975** | 0.3312 | 0.3043 | 0.3249 | 0.3777 |
| PTV2 | 336 | 336 | 0.3375 | 0.3745 | **0.3348** | 0.4380 | 0.4592 |

### Table A2 | MAE (`sl_test/mae`)

| Dataset | $L$ | $O$ | EMForecaster (2025) | DLinear (2023) | TSMixer (2023) | PatchTST (2023) | Transformer (2017) |
|---|---|---|---|---|---|---|---|
| DET | 192 | 96 | **0.3915** | 0.4961 | 0.4253 | 0.4527 | 0.5947 |
| DET | 192 | 336 | **0.4654** | 0.5318 | 0.4851 | 0.5232 | 0.9980 |
| DET | 336 | 96 | **0.3797** | 0.4452 | 0.4246 | 0.4653 | 0.5738 |
| DET | 336 | 336 | **0.4436** | 0.5106 | 0.4849 | 0.5151 | 1.0063 |
| PTV | 192 | 96 | **0.3740** | 0.5178 | 0.4036 | 0.4441 | 0.5774 |
| PTV | 192 | 336 | **0.4201** | 0.5069 | 0.4312 | 0.4723 | 0.9895 |
| PTV | 336 | 96 | **0.3525** | 0.4279 | 0.3954 | 0.4523 | 0.6027 |
| PTV | 336 | 336 | **0.4151** | 0.4760 | 0.4387 | 0.4542 | 0.5895 |
| PTV2 | 192 | 96 | **0.4372** | 0.4945 | 0.4449 | 0.4672 | 0.4756 |
| PTV2 | 192 | 336 | 0.4678 | 0.5030 | **0.4651** | 0.4952 | 0.8950 |
| PTV2 | 336 | 96 | **0.4220** | 0.4434 | 0.4227 | 0.4361 | 0.4840 |
| PTV2 | 336 | 336 | 0.4542 | 0.4755 | **0.4483** | 0.5092 | 0.5352 |

### Table B | Training Time (`parameters/running_time`)

**Bold** = fastest in the row. `X` = not logged (e.g. ARIMA is not gradient-trained).

| Dataset | $L$ | $O$ | EMForecaster (2025) | DLinear (2023) | TSMixer (2023) | PatchTST (2023) | Transformer (2017) |
|---|---|---|---|---|---|---|---|
| DET | 192 | 96 | 1m 7s | **30s** | 44s | 1m 2s | 35s |
| DET | 192 | 336 | 47s | 32s | 46s | 55s | **26s** |
| DET | 336 | 96 | 1m 5s | **33s** | 45s | 49s | 1m 12s |
| DET | 336 | 336 | 50s | **34s** | 45s | 1m 9s | 49s |
| PTV | 192 | 96 | 41s | **30s** | 39s | 52s | 46s |
| PTV | 192 | 336 | 36s | 30s | 40s | 51s | **29s** |
| PTV | 336 | 96 | 42s | **29s** | 40s | 45s | 52s |
| PTV | 336 | 336 | 35s | **30s** | 41s | 45s | 59s |
| PTV2 | 192 | 96 | 54s | **45s** | 1m 5s | 1m 26s | 1m 4s |
| PTV2 | 192 | 336 | 1m 0s | 54s | 1m 15s | **45s** | 1m 20s |
| PTV2 | 336 | 96 | 1m 2s | **44s** | 1m 13s | 1m 6s | 1m 11s |
| PTV2 | 336 | 336 | 1m 4s | 56s | 1m 9s | **33s** | 1m 25s |

### Table C | Per-Epoch Training Time (`running_time / len(sl_train/loss)`, per run)

| Dataset | $L$ | $O$ | EMForecaster (2025) | DLinear (2023) | TSMixer (2023) | PatchTST (2023) | Transformer (2017) |
|---|---|---|---|---|---|---|---|
| DET | 192 | 96 | 0.7s | **0.3s** | 0.4s | 0.6s | 0.6s |
| DET | 192 | 336 | 0.7s | **0.3s** | 0.5s | 0.6s | 0.7s |
| DET | 336 | 96 | 0.8s | **0.3s** | 0.5s | 0.5s | 0.8s |
| DET | 336 | 336 | 0.8s | **0.3s** | 0.5s | 0.7s | 1.0s |
| PTV | 192 | 96 | 0.5s | **0.3s** | 0.4s | 0.5s | 0.5s |
| PTV | 192 | 336 | 0.6s | **0.3s** | 0.4s | 0.5s | 0.5s |
| PTV | 336 | 96 | 0.7s | **0.3s** | 0.4s | 0.5s | 0.7s |
| PTV | 336 | 336 | 0.7s | **0.3s** | 0.4s | 0.5s | 0.7s |
| PTV2 | 192 | 96 | 1.0s | **0.5s** | 0.7s | 0.9s | 0.9s |
| PTV2 | 192 | 336 | 1.1s | **0.5s** | 0.8s | 0.8s | 1.0s |
| PTV2 | 336 | 96 | 1.5s | **0.4s** | 0.7s | 0.8s | 1.4s |
| PTV2 | 336 | 336 | 1.4s | **0.6s** | 0.7s | 0.8s | 1.8s |

#### Per-model summary

| Model | Mean / epoch | Estimation for 100 ep |
|---|---:|---:|
| DLinear (2023) | 0.4s | ~37s |
| TSMixer (2023) | 0.5s | ~52s |
| PatchTST (2023) | 0.6s | ~1m 3s |
| Transformer (2017) | 0.9s | ~1m 28s |
| EMForecaster (2025) | 0.9s | ~1m 29s |

### Table D | Marginal Coverage % (`sl_test/ic`)

Not bolded -- "best" means closest to the target coverage, not highest.

| Dataset | $L$ | $O$ | EMForecaster (2025) | DLinear (2023) | TSMixer (2023) | PatchTST (2023) | Transformer (2017) |
|---|---|---|---|---|---|---|---|
| DET | 192 | 96 | 88.93 | 91.12 | 90.18 | 90.35 | 93.25 |
| DET | 192 | 336 | 88.38 | 90.57 | 89.47 | 89.57 | 92.14 |
| DET | 336 | 96 | 87.54 | 90.68 | 89.40 | 89.80 | 88.17 |
| DET | 336 | 336 | 87.93 | 90.54 | 89.17 | 90.17 | 91.92 |
| PTV | 192 | 96 | 87.06 | 87.19 | 90.71 | 89.43 | 79.24 |
| PTV | 192 | 336 | 85.14 | 86.38 | 88.21 | 87.13 | 81.56 |
| PTV | 336 | 96 | 88.44 | 88.31 | 92.94 | 88.84 | 79.29 |
| PTV | 336 | 336 | 86.06 | 86.05 | 89.45 | 86.90 | 83.19 |
| PTV2 | 192 | 96 | 90.43 | 89.48 | 90.76 | 89.76 | 90.69 |
| PTV2 | 192 | 336 | 90.06 | 89.97 | 90.45 | 90.28 | 84.27 |
| PTV2 | 336 | 96 | 91.20 | 91.06 | 91.36 | 91.89 | 89.30 |
| PTV2 | 336 | 336 | 90.73 | 90.57 | 90.99 | 90.36 | 88.04 |

### Table E | Joint Coverage % (`sl_test/jc`)

| Dataset | $L$ | $O$ | EMForecaster (2025) | DLinear (2023) | TSMixer (2023) | PatchTST (2023) | Transformer (2017) |
|---|---|---|---|---|---|---|---|
| DET | 192 | 96 | 19.04 | 39.10 | 22.51 | 29.88 | 45.61 |
| DET | 192 | 336 | 3.16 | 14.84 | 7.06 | 8.22 | 22.44 |
| DET | 336 | 96 | 10.16 | 43.13 | 16.83 | 29.60 | 32.29 |
| DET | 336 | 336 | 1.83 | 12.63 | 5.26 | 15.19 | 22.31 |
| PTV | 192 | 96 | 5.15 | 14.33 | 9.74 | 13.55 | 22.68 |
| PTV | 192 | 336 | 0.02 | 0.59 | 0.86 | 0.39 | 3.60 |
| PTV | 336 | 96 | 4.90 | 9.66 | 25.92 | 12.22 | 27.50 |
| PTV | 336 | 336 | 0.00 | 0.00 | 3.13 | 1.29 | 2.29 |
| PTV2 | 192 | 96 | 4.77 | 5.99 | 5.73 | 5.30 | 7.22 |
| PTV2 | 192 | 336 | 0.00 | 0.00 | 0.17 | 0.36 | 0.00 |
| PTV2 | 336 | 96 | 3.86 | 5.83 | 4.71 | 6.68 | 5.87 |
| PTV2 | 336 | 336 | 0.00 | 0.00 | 0.01 | 0.25 | 0.05 |

### Table F | Mean Interval Width (`sl_test/interval_width_mean`)

Not bolded -- narrower is only better when the target coverage (Table D) is met.

| Dataset | $L$ | $O$ | EMForecaster (2025) | DLinear (2023) | TSMixer (2023) | PatchTST (2023) | Transformer (2017) |
|---|---|---|---|---|---|---|---|
| DET | 192 | 96 | 1.570 | 2.190 | 1.888 | 2.023 | 2.672 |
| DET | 192 | 336 | 1.930 | 2.429 | 2.190 | 2.376 | 4.133 |
| DET | 336 | 96 | 1.450 | 2.098 | 1.793 | 1.924 | 2.302 |
| DET | 336 | 336 | 1.817 | 2.403 | 2.154 | 2.425 | 4.162 |
| PTV | 192 | 96 | 1.408 | 1.949 | 1.759 | 1.817 | 1.940 |
| PTV | 192 | 336 | 1.524 | 1.895 | 1.760 | 1.836 | 3.077 |
| PTV | 336 | 96 | 1.373 | 1.720 | 1.923 | 1.717 | 2.067 |
| PTV | 336 | 336 | 1.534 | 1.778 | 1.909 | 1.782 | 2.062 |
| PTV2 | 192 | 96 | 1.840 | 2.021 | 1.915 | 1.941 | 1.972 |
| PTV2 | 192 | 336 | 1.930 | 2.082 | 1.974 | 2.125 | 3.158 |
| PTV2 | 336 | 96 | 1.802 | 1.908 | 1.847 | 1.953 | 1.936 |
| PTV2 | 336 | 336 | 1.893 | 2.010 | 1.930 | 2.179 | 2.066 |

Every cell in this section averages **1** replicate run(s).

---

## Model sizes

Weights come from `parameters/sl/num_params` in each log. For multivariate logs this can vary by dataset (different channel counts), not just by (model, L, O); this table shows the first value found per (model, L, O) -- cross-check if that matters for your use case.

| Model | L192 / O96 | L192 / O336 | L336 / O96 | L336 / O336 |
|---|---:|---:|---:|---:|
| EMForecaster (2025) | 1,495,235 | 4,444,595 | 2,416,907 | 7,578,107 |
| DLinear (2023) | 37,056 | 129,696 | 64,704 | 226,464 |
| TSMixer (2023) | 31,139 | 77,459 | 54,323 | 135,203 |
| PatchTST (2023) | 44,194 | 136,594 | 72,130 | 233,650 |
| Transformer (2017) | 130,528 | 179,248 | 372,736 | 456,016 |

# Univariate Merged

---

## 1 · Point-Forecasting Performance (without conformal prediction)

### Table A | MSE (`sl_test/loss`)

**Bold** = best (lowest) value in the row, among populated cells.

| Dataset | $L$ | $O$ | EMForecaster (2025) | DLinear (2023) | TSMixer (2023) | PatchTST (2023) | Transformer (2017) | LSTM (1997) |
|---|---|---|---|---|---|---|---|---|
| DET | 192 | 96 |  | 0.4524 | 0.3678 | 0.4153 | 1.0383 | **0.3404** |
| DET | 192 | 336 |  | 0.5306 | **0.4583** | 0.5373 | 1.1477 | 0.4948 |
| DET | 336 | 96 |  | 0.3995 | 0.3560 | 0.3963 | 1.0601 | **0.3500** |
| DET | 336 | 336 |  | 0.5046 | **0.4619** | 0.5127 | 1.4810 | 0.4834 |
| PTV | 192 | 96 |  | 0.4710 | **0.3212** | 0.3772 | 1.0271 | 0.3501 |
| PTV | 192 | 336 |  | 0.4747 | **0.3657** | 0.4538 | 1.3877 | 0.5308 |
| PTV | 336 | 96 |  | 0.3545 | **0.3074** | 0.3263 | 1.0398 | 0.3514 |
| PTV | 336 | 336 |  | 0.4256 | **0.3846** | 0.3876 | 1.1085 | 0.5473 |
| PTV2 | 192 | 96 |  | 0.4003 | 0.3348 | 0.3994 | 0.3746 | **0.3293** |
| PTV2 | 192 | 336 |  | 0.4130 | **0.3594** | 0.4486 | 0.8356 | 0.3627 |
| PTV2 | 336 | 96 |  | 0.3312 | **0.3038** | 0.3428 | 0.7625 | 0.3170 |
| PTV2 | 336 | 336 |  | 0.3745 | **0.3354** | 0.4141 | 0.8474 | 0.3542 |

### Table A2 | MAE (`sl_test/mae`)

| Dataset | $L$ | $O$ | EMForecaster (2025) | DLinear (2023) | TSMixer (2023) | PatchTST (2023) | Transformer (2017) | LSTM (1997) |
|---|---|---|---|---|---|---|---|---|
| DET | 192 | 96 |  | 0.4963 | 0.4288 | 0.4625 | 0.7953 | **0.4186** |
| DET | 192 | 336 |  | 0.5321 | **0.4839** | 0.5305 | 0.8408 | 0.5274 |
| DET | 336 | 96 |  | 0.4451 | **0.4229** | 0.4587 | 0.8056 | 0.4255 |
| DET | 336 | 336 |  | 0.5107 | **0.4859** | 0.5211 | 1.0047 | 0.5196 |
| PTV | 192 | 96 |  | 0.5182 | **0.4093** | 0.4618 | 0.7953 | 0.4372 |
| PTV | 192 | 336 |  | 0.5074 | **0.4303** | 0.4931 | 0.9876 | 0.5372 |
| PTV | 336 | 96 |  | 0.4278 | **0.3946** | 0.4273 | 0.7953 | 0.4357 |
| PTV | 336 | 336 |  | 0.4763 | **0.4386** | 0.4552 | 0.8558 | 0.5443 |
| PTV2 | 192 | 96 |  | 0.4945 | **0.4455** | 0.4944 | 0.4835 | 0.4495 |
| PTV2 | 192 | 336 |  | 0.5030 | **0.4646** | 0.5182 | 0.7287 | 0.4748 |
| PTV2 | 336 | 96 |  | 0.4434 | **0.4224** | 0.4478 | 0.6827 | 0.4410 |
| PTV2 | 336 | 336 |  | 0.4755 | **0.4487** | 0.4964 | 0.7357 | 0.4708 |

### Table B | Training Time (`parameters/running_time`)

**Bold** = fastest in the row. `X` = not logged (e.g. ARIMA is not gradient-trained).

| Dataset | $L$ | $O$ | EMForecaster (2025) | DLinear (2023) | TSMixer (2023) | PatchTST (2023) | Transformer (2017) | LSTM (1997) |
|---|---|---|---|---|---|---|---|---|
| DET | 192 | 96 | X | **36s** | 1m 11s | 1m 42s | 5m 16s | 2m 3s |
| DET | 192 | 336 | X | **1m 1s** | 1m 10s | 1m 34s | 6m 40s | 2m 22s |
| DET | 336 | 96 | X | **39s** | 1m 26s | 2m 40s | 9m 40s | 2m 4s |
| DET | 336 | 336 | X | **1m 8s** | 1m 29s | 3m 0s | 8m 12s | 2m 17s |
| PTV | 192 | 96 | X | **22s** | 42s | 54s | 4m 4s | 1m 16s |
| PTV | 192 | 336 | X | **38s** | 50s | 39s | 1m 20s | 1m 26s |
| PTV | 336 | 96 | X | **28s** | 54s | 2m 3s | 6m 42s | 1m 18s |
| PTV | 336 | 336 | X | **54s** | 1m 8s | 1m 40s | 5m 28s | 1m 24s |
| PTV2 | 192 | 96 | X | **54s** | 1m 56s | 2m 14s | 9m 32s | 4m 27s |
| PTV2 | 192 | 336 | X | 1m 58s | 2m 37s | **1m 12s** | 13m 58s | 5m 21s |
| PTV2 | 336 | 96 | X | **1m 12s** | 2m 38s | 3m 56s | 17m 22s | 4m 41s |
| PTV2 | 336 | 336 | X | 2m 53s | **2m 32s** | 2m 40s | 15m 12s | 5m 22s |

### Table C | Per-Epoch Training Time (`running_time / len(sl_train/loss)`, per run)

| Dataset | $L$ | $O$ | EMForecaster (2025) | DLinear (2023) | TSMixer (2023) | PatchTST (2023) | Transformer (2017) | LSTM (1997) |
|---|---|---|---|---|---|---|---|---|
| DET | 192 | 96 | X | **0.4s** | 0.7s | 1.0s | 7.0s | 1.2s |
| DET | 192 | 336 | X | **0.6s** | 0.7s | 1.3s | 10.0s | 1.4s |
| DET | 336 | 96 | X | **0.4s** | 0.9s | 1.6s | 17.2s | 1.2s |
| DET | 336 | 336 | X | **0.7s** | 0.9s | 1.8s | 19.6s | 1.4s |
| PTV | 192 | 96 | X | **0.2s** | 0.4s | 0.5s | 3.8s | 0.8s |
| PTV | 192 | 336 | X | **0.4s** | 0.5s | 0.5s | 5.5s | 0.9s |
| PTV | 336 | 96 | X | **0.3s** | 0.5s | 1.2s | 10.3s | 0.8s |
| PTV | 336 | 336 | X | **0.5s** | 0.7s | 1.0s | 12.0s | 0.8s |
| PTV2 | 192 | 96 | X | **0.5s** | 1.2s | 1.7s | 12.3s | 2.7s |
| PTV2 | 192 | 336 | X | **1.2s** | 1.6s | 1.6s | 19.6s | 3.2s |
| PTV2 | 336 | 96 | X | **0.7s** | 1.6s | 3.2s | 36.7s | 2.8s |
| PTV2 | 336 | 336 | X | 1.7s | **1.5s** | 3.7s | 39.6s | 3.2s |

#### Per-model summary

| Model | Mean / epoch | Estimation for 100 ep |
|---|---:|---:|
| DLinear (2023) | 0.6s | ~1m 4s |
| TSMixer (2023) | 0.9s | ~1m 33s |
| PatchTST (2023) | 1.6s | ~2m 41s |
| LSTM (1997) | 1.7s | ~2m 50s |
| Transformer (2017) | 16.1s | ~26m 54s |

Every cell in this section averages **2** replicate run(s).

---

## 2 · Point-Forecasting Performance (with conformal prediction)

### Table A | MSE (`sl_test/loss`)

**Bold** = best (lowest) value in the row, among populated cells.

| Dataset | $L$ | $O$ | EMForecaster (2025) | DLinear (2023) | TSMixer (2023) | PatchTST (2023) | Transformer (2017) | LSTM (1997) |
|---|---|---|---|---|---|---|---|---|
| DET | 192 | 96 | **0.3124** |  |  |  | 1.0383 | 0.3404 |
| DET | 192 | 336 | **0.4239** |  |  |  | 1.1477 | 0.4949 |
| DET | 336 | 96 | **0.3009** |  |  |  | 1.0601 | 0.3500 |
| DET | 336 | 336 | **0.3966** |  |  |  | 1.4810 | 0.4834 |
| PTV | 192 | 96 | **0.2614** |  |  |  | 1.0271 | 0.3501 |
| PTV | 192 | 336 | **0.3198** |  |  |  | 1.3877 | 0.5307 |
| PTV | 336 | 96 | **0.2435** |  |  |  | 1.0398 | 0.3514 |
| PTV | 336 | 336 | **0.3137** |  |  |  | 1.1085 | 0.5473 |
| PTV2 | 192 | 96 | **0.3220** |  |  |  | 0.3746 | 0.3293 |
| PTV2 | 192 | 336 | **0.3586** |  |  |  | 0.8356 | 0.3626 |
| PTV2 | 336 | 96 | **0.2999** |  |  |  | 0.7625 | 0.3169 |
| PTV2 | 336 | 336 | **0.3387** |  |  |  | 0.8474 | 0.3541 |

### Table A2 | MAE (`sl_test/mae`)

| Dataset | $L$ | $O$ | EMForecaster (2025) | DLinear (2023) | TSMixer (2023) | PatchTST (2023) | Transformer (2017) | LSTM (1997) |
|---|---|---|---|---|---|---|---|---|
| DET | 192 | 96 | **0.3904** |  |  |  | 0.7953 | 0.4186 |
| DET | 192 | 336 | **0.4636** |  |  |  | 0.8408 | 0.5274 |
| DET | 336 | 96 | **0.3785** |  |  |  | 0.8056 | 0.4255 |
| DET | 336 | 336 | **0.4459** |  |  |  | 1.0047 | 0.5196 |
| PTV | 192 | 96 | **0.3754** |  |  |  | 0.7953 | 0.4372 |
| PTV | 192 | 336 | **0.4185** |  |  |  | 0.9876 | 0.5372 |
| PTV | 336 | 96 | **0.3591** |  |  |  | 0.7953 | 0.4357 |
| PTV | 336 | 336 | **0.4117** |  |  |  | 0.8558 | 0.5443 |
| PTV2 | 192 | 96 | **0.4387** |  |  |  | 0.4835 | 0.4495 |
| PTV2 | 192 | 336 | **0.4680** |  |  |  | 0.7287 | 0.4748 |
| PTV2 | 336 | 96 | **0.4230** |  |  |  | 0.6827 | 0.4410 |
| PTV2 | 336 | 336 | **0.4547** |  |  |  | 0.7357 | 0.4708 |

### Table B | Training Time (`parameters/running_time`)

**Bold** = fastest in the row. `X` = not logged (e.g. ARIMA is not gradient-trained).

| Dataset | $L$ | $O$ | EMForecaster (2025) | DLinear (2023) | TSMixer (2023) | PatchTST (2023) | Transformer (2017) | LSTM (1997) |
|---|---|---|---|---|---|---|---|---|
| DET | 192 | 96 | 11m 25s | X | X | X | 4m 22s | **2m 26s** |
| DET | 192 | 336 | 10m 14s | X | X | X | 5m 22s | **2m 30s** |
| DET | 336 | 96 | 19m 56s | X | X | X | 8m 28s | **2m 23s** |
| DET | 336 | 336 | 14m 41s | X | X | X | 8m 21s | **2m 30s** |
| PTV | 192 | 96 | 6m 27s | X | X | X | 4m 17s | **1m 24s** |
| PTV | 192 | 336 | 5m 18s | X | X | X | **1m 31s** | 1m 32s |
| PTV | 336 | 96 | 7m 6s | X | X | X | 7m 16s | **1m 24s** |
| PTV | 336 | 336 | 7m 58s | X | X | X | 5m 37s | **1m 33s** |
| PTV2 | 192 | 96 | 17m 39s | X | X | X | 9m 9s | **4m 40s** |
| PTV2 | 192 | 336 | 15m 40s | X | X | X | 12m 32s | **5m 32s** |
| PTV2 | 336 | 96 | 30m 13s | X | X | X | 15m 45s | **4m 49s** |
| PTV2 | 336 | 336 | 30m 12s | X | X | X | 15m 50s | **5m 36s** |

### Table C | Per-Epoch Training Time (`running_time / len(sl_train/loss)`, per run)

| Dataset | $L$ | $O$ | EMForecaster (2025) | DLinear (2023) | TSMixer (2023) | PatchTST (2023) | Transformer (2017) | LSTM (1997) |
|---|---|---|---|---|---|---|---|---|
| DET | 192 | 96 | 7.7s | X | X | X | 5.9s | **1.5s** |
| DET | 192 | 336 | 9.2s | X | X | X | 8.6s | **1.5s** |
| DET | 336 | 96 | 16.1s | X | X | X | 15.2s | **1.4s** |
| DET | 336 | 336 | 12.7s | X | X | X | 19.6s | **1.5s** |
| PTV | 192 | 96 | 4.8s | X | X | X | 4.0s | **0.8s** |
| PTV | 192 | 336 | 5.1s | X | X | X | 6.3s | **0.9s** |
| PTV | 336 | 96 | 7.4s | X | X | X | 11.0s | **0.8s** |
| PTV | 336 | 336 | 9.5s | X | X | X | 12.6s | **0.9s** |
| PTV2 | 192 | 96 | 19.5s | X | X | X | 11.8s | **2.8s** |
| PTV2 | 192 | 336 | 20.4s | X | X | X | 17.4s | **3.3s** |
| PTV2 | 336 | 96 | 39.0s | X | X | X | 33.2s | **2.9s** |
| PTV2 | 336 | 336 | 38.1s | X | X | X | 40.9s | **3.4s** |

#### Per-model summary

| Model | Mean / epoch | Estimation for 100 ep |
|---|---:|---:|
| LSTM (1997) | 1.8s | ~3m 2s |
| Transformer (2017) | 15.5s | ~25m 54s |
| EMForecaster (2025) | 15.8s | ~26m 20s |

### Table D | Marginal Coverage % (`sl_test/ic`)

Not bolded -- "best" means closest to the target coverage, not highest.

| Dataset | $L$ | $O$ | EMForecaster (2025) | DLinear (2023) | TSMixer (2023) | PatchTST (2023) | Transformer (2017) | LSTM (1997) |
|---|---|---|---|---|---|---|---|---|
| DET | 192 | 96 | 90.65 |  |  |  | 92.38 | 88.79 |
| DET | 192 | 336 | 91.20 |  |  |  | 90.20 | 89.21 |
| DET | 336 | 96 | 90.86 |  |  |  | 89.86 | 87.51 |
| DET | 336 | 336 | 91.31 |  |  |  | 92.24 | 88.81 |
| PTV | 192 | 96 | 88.45 |  |  |  | 80.73 | 83.30 |
| PTV | 192 | 336 | 87.21 |  |  |  | 81.64 | 81.15 |
| PTV | 336 | 96 | 88.91 |  |  |  | 79.98 | 82.84 |
| PTV | 336 | 336 | 87.80 |  |  |  | 82.97 | 80.74 |
| PTV2 | 192 | 96 | 87.27 |  |  |  | 90.45 | 89.85 |
| PTV2 | 192 | 336 | 85.81 |  |  |  | 90.88 | 89.67 |
| PTV2 | 336 | 96 | 87.18 |  |  |  | 88.04 | 90.12 |
| PTV2 | 336 | 336 | 86.10 |  |  |  | 90.67 | 89.71 |

### Table E | Joint Coverage % (`sl_test/jc`)

| Dataset | $L$ | $O$ | EMForecaster (2025) | DLinear (2023) | TSMixer (2023) | PatchTST (2023) | Transformer (2017) | LSTM (1997) |
|---|---|---|---|---|---|---|---|---|
| DET | 192 | 96 | 40.89 |  |  |  | 50.41 | 19.11 |
| DET | 192 | 336 | 20.59 |  |  |  | 18.47 | 2.06 |
| DET | 336 | 96 | 39.63 |  |  |  | 43.42 | 13.69 |
| DET | 336 | 336 | 20.39 |  |  |  | 22.80 | 1.17 |
| PTV | 192 | 96 | 23.78 |  |  |  | 28.43 | 7.38 |
| PTV | 192 | 336 | 4.66 |  |  |  | 2.79 | 0.26 |
| PTV | 336 | 96 | 24.57 |  |  |  | 29.18 | 7.48 |
| PTV | 336 | 336 | 5.84 |  |  |  | 3.67 | 0.33 |
| PTV2 | 192 | 96 | 20.14 |  |  |  | 7.14 | 4.27 |
| PTV2 | 192 | 336 | 1.45 |  |  |  | 7.97 | 0.04 |
| PTV2 | 336 | 96 | 18.93 |  |  |  | 10.34 | 3.12 |
| PTV2 | 336 | 336 | 1.80 |  |  |  | 6.98 | 0.01 |

### Table F | Mean Interval Width (`sl_test/interval_width_mean`)

Not bolded -- narrower is only better when the target coverage (Table D) is met.

| Dataset | $L$ | $O$ | EMForecaster (2025) | DLinear (2023) | TSMixer (2023) | PatchTST (2023) | Transformer (2017) | LSTM (1997) |
|---|---|---|---|---|---|---|---|---|
| DET | 192 | 96 | 3.389 |  |  |  | 3.375 | 1.712 |
| DET | 192 | 336 | 4.531 |  |  |  | 3.457 | 2.170 |
| DET | 336 | 96 | 3.430 |  |  |  | 3.252 | 1.665 |
| DET | 336 | 336 | 4.602 |  |  |  | 4.172 | 2.122 |
| PTV | 192 | 96 | 2.973 |  |  |  | 2.575 | 1.573 |
| PTV | 192 | 336 | 3.814 |  |  |  | 3.078 | 1.868 |
| PTV | 336 | 96 | 2.968 |  |  |  | 2.525 | 1.548 |
| PTV | 336 | 336 | 3.831 |  |  |  | 2.827 | 1.884 |
| PTV2 | 192 | 96 | 3.185 |  |  |  | 1.986 | 1.827 |
| PTV2 | 192 | 336 | 3.934 |  |  |  | 3.157 | 1.913 |
| PTV2 | 336 | 96 | 3.145 |  |  |  | 2.594 | 1.790 |
| PTV2 | 336 | 336 | 3.949 |  |  |  | 3.156 | 1.878 |

Every cell in this section averages **2** replicate run(s).

---

## Model sizes

Weights come from `parameters/sl/num_params` in each log. For multivariate logs this can vary by dataset (different channel counts), not just by (model, L, O); this table shows the first value found per (model, L, O) -- cross-check if that matters for your use case.

| Model | L192 / O96 | L192 / O336 | L336 / O96 | L336 / O336 |
|---|---:|---:|---:|---:|
| EMForecaster (2025) | 1,495,235 | 4,444,595 | 2,416,907 | 7,578,107 |
| DLinear (2023) | 37,056 | 129,696 | 64,704 | 226,464 |
| TSMixer (2023) | 31,139 | 77,459 | 54,323 | 135,203 |
| PatchTST (2023) | 44,194 | 136,594 | 72,130 | 233,650 |
| Transformer (2017) | 130,528 | 179,248 | 372,736 | 456,016 |
| LSTM (1997) | 32,096 | 40,016 | 50,528 | 58,448 |

# Univariate Per Channel

---

## 2 · Point-Forecasting Performance (with conformal prediction)

### Table A | MSE (`sl_test/loss`)

**Bold** = best (lowest) value in the row, among populated cells.

| Dataset | $L$ | $O$ | EMForecaster (2025) | DLinear (2023) | TSMixer (2023) | PatchTST (2023) | Transformer (2017) | LSTM (1997) | ARIMA (1970) |
|---|---|---|---|---|---|---|---|---|---|
| ED2022 | 336 | 96 | **0.4848** | 0.5321 | 0.5484 | 0.5419 | 0.6127 | 0.5785 | 1.0194 |
| UH2022 | 336 | 96 | **1.1704** | 1.1900 | 1.2932 | 1.2623 | 1.2710 | 1.2180 | 1.6017 |
| EMF_2023_UNIVERSITY_5_WITH_PTV | 336 | 96 | **1.5052** | 1.5163 | 1.5262 | 1.5266 | 1.5723 | 1.5435 | 1.8089 |

### Table A2 | MAE (`sl_test/mae`)

| Dataset | $L$ | $O$ | EMForecaster (2025) | DLinear (2023) | TSMixer (2023) | PatchTST (2023) | Transformer (2017) | LSTM (1997) | ARIMA (1970) |
|---|---|---|---|---|---|---|---|---|---|
| ED2022 | 336 | 96 | **0.3428** | 0.3781 | 0.3875 | 0.3817 | 0.4344 | 0.4160 | 0.6076 |
| UH2022 | 336 | 96 | **0.4586** | 0.4851 | 0.4961 | 0.4988 | 0.5099 | 0.5089 | 0.6807 |
| EMF_2023_UNIVERSITY_5_WITH_PTV | 336 | 96 | **0.4498** | 0.4677 | 0.4591 | 0.4701 | 0.4904 | 0.4795 | 0.6212 |

### Table B | Training Time (`parameters/running_time`)

**Bold** = fastest in the row. `X` = not logged (e.g. ARIMA is not gradient-trained).

| Dataset | $L$ | $O$ | EMForecaster (2025) | DLinear (2023) | TSMixer (2023) | PatchTST (2023) | Transformer (2017) | LSTM (1997) | ARIMA (1970) |
|---|---|---|---|---|---|---|---|---|---|
| ED2022 | 336 | 96 | 1m 51s | **41s** | 42s | 1m 27s | 2m 55s | 44s | X |
| UH2022 | 336 | 96 | 2m 6s | 53s | **41s** | 1m 46s | 3m 40s | 52s | X |
| EMF_2023_UNIVERSITY_5_WITH_PTV | 336 | 96 | 3m 39s | **54s** | 1m 5s | 2m 10s | 6m 45s | 1m 24s | X |

### Table C | Per-Epoch Training Time (`running_time / len(sl_train/loss)`, per run)

| Dataset | $L$ | $O$ | EMForecaster (2025) | DLinear (2023) | TSMixer (2023) | PatchTST (2023) | Transformer (2017) | LSTM (1997) | ARIMA (1970) |
|---|---|---|---|---|---|---|---|---|---|
| ED2022 | 336 | 96 | 2.2s | 0.4s | **0.4s** | 0.9s | 3.5s | 0.4s | X |
| UH2022 | 336 | 96 | 2.9s | 0.6s | **0.4s** | 1.2s | 5.0s | 0.5s | X |
| EMF_2023_UNIVERSITY_5_WITH_PTV | 336 | 96 | 5.2s | 0.7s | **0.7s** | 1.5s | 9.0s | 0.9s | X |

#### Per-model summary

| Model | Mean / epoch | Estimation for 100 ep |
|---|---:|---:|
| TSMixer (2023) | 0.5s | ~50s |
| DLinear (2023) | 0.6s | ~56s |
| LSTM (1997) | 0.6s | ~1m 1s |
| PatchTST (2023) | 1.2s | ~2m 1s |
| EMForecaster (2025) | 3.4s | ~5m 44s |
| Transformer (2017) | 5.8s | ~9m 43s |

### Table D | Marginal Coverage % (`sl_test/ic`)

Not bolded -- "best" means closest to the target coverage, not highest.

| Dataset | $L$ | $O$ | EMForecaster (2025) | DLinear (2023) | TSMixer (2023) | PatchTST (2023) | Transformer (2017) | LSTM (1997) | ARIMA (1970) |
|---|---|---|---|---|---|---|---|---|---|
| ED2022 | 336 | 96 | 84.94 | 84.64 | 85.59 | 85.05 | 84.58 | 85.11 | 86.43 |
| UH2022 | 336 | 96 | 86.72 | 86.24 | 86.70 | 85.97 | 85.81 | 85.82 | 84.28 |
| EMF_2023_UNIVERSITY_5_WITH_PTV | 336 | 96 | 90.45 | 90.76 | 91.33 | 90.66 | 90.35 | 90.05 | 88.65 |

### Table E | Joint Coverage % (`sl_test/jc`)

| Dataset | $L$ | $O$ | EMForecaster (2025) | DLinear (2023) | TSMixer (2023) | PatchTST (2023) | Transformer (2017) | LSTM (1997) | ARIMA (1970) |
|---|---|---|---|---|---|---|---|---|---|
| ED2022 | 336 | 96 | 10.48 | 11.10 | 12.99 | 12.89 | 12.93 | 14.13 | 21.65 |
| UH2022 | 336 | 96 | 6.89 | 6.60 | 7.59 | 8.91 | 8.01 | 8.64 | 9.66 |
| EMF_2023_UNIVERSITY_5_WITH_PTV | 336 | 96 | 11.57 | 11.87 | 14.82 | 13.64 | 13.67 | 12.01 | 13.06 |

### Table F | Mean Interval Width (`sl_test/interval_width_mean`)

Not bolded -- narrower is only better when the target coverage (Table D) is met.

| Dataset | $L$ | $O$ | EMForecaster (2025) | DLinear (2023) | TSMixer (2023) | PatchTST (2023) | Transformer (2017) | LSTM (1997) | ARIMA (1970) |
|---|---|---|---|---|---|---|---|---|---|
| ED2022 | 336 | 96 | 1.315 | 1.434 | 1.516 | 1.451 | 1.637 | 1.582 | 2.460 |
| UH2022 | 336 | 96 | 1.684 | 1.710 | 1.793 | 1.774 | 1.822 | 1.812 | 2.334 |
| EMF_2023_UNIVERSITY_5_WITH_PTV | 336 | 96 | 1.941 | 1.935 | 2.202 | 2.001 | 2.056 | 1.996 | 2.446 |

Replicate counts per cell in this section range from **27** to **30**, so rows mix cells with different amounts of averaging.

---

## Model sizes

Weights come from `parameters/sl/num_params` in each log. For multivariate logs this can vary by dataset (different channel counts), not just by (model, L, O); this table shows the first value found per (model, L, O) -- cross-check if that matters for your use case.

| Model | L336 / O96 |
|---|---:|
| EMForecaster (2025) | 2,416,907 |
| DLinear (2023) | 64,704 |
| TSMixer (2023) | 54,323 |
| PatchTST (2023) | 72,130 |
| Transformer (2017) | 372,736 |
| LSTM (1997) | 20,674 |
