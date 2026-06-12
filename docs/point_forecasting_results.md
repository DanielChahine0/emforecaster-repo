# Point-Forecasting Performance (without conformal prediction)

Test-set metrics, averaged over seeds.
Datasets: **DET** = `rf_emf_det`, **PTV** = `rf_emf_ptv`, **PTV2** = `rf_emf_ptv2`.
$L$ = input/lookback length (`seq_len`), $O$ = forecast horizon (`pred_len`).

**Bold** = best (lowest) value in the row. Values rounded to 4 decimals.

## Table A | MSE

| Dataset | $L$ | $O$ | EMForecaster | DLinear | PatchTST | TSMixer | LSTM | ARIMA |
|---|---|---|---|---|---|---|---|---|
| DET  | 192 | 96  | **0.3111** | 0.4517 | 0.4400 | 0.3648 | 0.3404 | 1.2371 |
| DET  | 192 | 336 | **0.4229** | 0.5303 | 0.5440 | 0.4639 | 0.4949 | 1.4194 |
| DET  | 336 | 96  | **0.3009** | 0.3993 | 0.3957 | 0.3551 | 0.3500 | 1.2371 |
| DET  | 336 | 336 | **0.3940** | 0.5048 | 0.5111 | 0.4602 | 0.4834 | 1.4194 |
| PTV  | 192 | 96  | **0.2614** | 0.4710 | 0.3772 | 0.3212 | 0.3501 | 1.1840 |
| PTV  | 192 | 336 | **0.3198** | 0.4747 | 0.4538 | 0.3657 | 0.5307 | 1.3061 |
| PTV  | 336 | 96  | **0.2435** | 0.3545 | 0.3263 | 0.3074 | 0.3514 | 1.1840 |
| PTV  | 336 | 336 | **0.3137** | 0.4256 | 0.3876 | 0.3846 | 0.5473 | 1.3061 |
| PTV2 | 192 | 96  | **0.3220** | 0.4003 | 0.3994 | 0.3348 | 0.3293 | 1.0128 |
| PTV2 | 192 | 336 | **0.3586** | 0.4130 | 0.4486 | 0.3594 | 0.3626 | 1.1346 |
| PTV2 | 336 | 96  | **0.2999** | 0.3312 | 0.3428 | 0.3038 | 0.3169 | 1.0128 |
| PTV2 | 336 | 336 | 0.3387 | 0.3745 | 0.4141 | **0.3354** | 0.3541 | 1.1346 |

## Table B | MAE

| Dataset | $L$ | $O$ | EMForecaster | DLinear | PatchTST | TSMixer | LSTM | ARIMA |
|---|---|---|---|---|---|---|---|---|
| DET  | 192 | 96  | **0.3890** | 0.4959 | 0.4880 | 0.4270 | 0.4186 | 0.8468 |
| DET  | 192 | 336 | **0.4638** | 0.5319 | 0.5372 | 0.4867 | 0.5274 | 0.9650 |
| DET  | 336 | 96  | **0.3773** | 0.4450 | 0.4541 | 0.4225 | 0.4255 | 0.8468 |
| DET  | 336 | 336 | **0.4443** | 0.5108 | 0.5167 | 0.4845 | 0.5196 | 0.9650 |
| PTV  | 192 | 96  | **0.3754** | 0.5182 | 0.4618 | 0.4093 | 0.4372 | 0.8374 |
| PTV  | 192 | 336 | **0.4185** | 0.5074 | 0.4931 | 0.4303 | 0.5372 | 0.9381 |
| PTV  | 336 | 96  | **0.3591** | 0.4278 | 0.4273 | 0.3946 | 0.4357 | 0.8374 |
| PTV  | 336 | 336 | **0.4117** | 0.4763 | 0.4552 | 0.4386 | 0.5443 | 0.9381 |
| PTV2 | 192 | 96  | **0.4387** | 0.4945 | 0.4944 | 0.4455 | 0.4495 | 0.7957 |
| PTV2 | 192 | 336 | 0.4680 | 0.5030 | 0.5182 | **0.4646** | 0.4748 | 0.8719 |
| PTV2 | 336 | 96  | 0.4230 | 0.4434 | 0.4478 | **0.4224** | 0.4410 | 0.7957 |
| PTV2 | 336 | 336 | 0.4547 | 0.4755 | 0.4964 | **0.4487** | 0.4708 | 0.8719 |

## Notes

- **MSE:** EMForecaster is best in 11 of 12 settings; TSMixer edges it out on PTV2 / L=336 / O=336 (0.3354 vs 0.3387).
- **MAE:** EMForecaster is best in 9 of 12 settings; TSMixer takes the remaining three, all on PTV2, L=192 / O=336 (0.4646 vs 0.4680), L=336 / O=96 (0.4224 vs 0.4230), and L=336 / O=336 (0.4487 vs 0.4547). The first two are decided only at the 4th decimal place.
- ARIMA values are markedly higher than the deep models across the board.
- ARIMA does not use a learned lookback, so its L=192 and L=336 rows are identical for a given (dataset, O).
- Seed aggregation: DET deep-model rows average 5 seeds; all other deep-model rows average 2 seeds (`69421`, `69422`).
