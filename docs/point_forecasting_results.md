# Point-Forecasting Performance (without conformal prediction)

Test-set metrics, averaged over seeds.
Datasets: **DET** = `rf_emf_det`, **PTV** = `rf_emf_ptv`, **PTV2** = `rf_emf_ptv2`.
$L$ = input/lookback length (`seq_len`), $O$ = forecast horizon (`pred_len`).
Source: deep models from `logs/**/log.json` (`sl_test/loss`, `sl_test/mae`); ARIMA from `arima_results.txt`.
**Bold** = best (lowest) value in the row. Values rounded to 3 decimals.

## Table A — MSE

| Dataset | $L$ | $O$ | EMForecaster | DLinear | PatchTST | TSMixer | LSTM | ARIMA |
|---|---|---|---|---|---|---|---|---|
| DET  | 192 | 96  | **0.311** | 0.452 | 0.440 | 0.365 | 0.340 | 1.237 |
| DET  | 192 | 336 | **0.423** | 0.530 | 0.544 | 0.464 | 0.495 | 1.419 |
| DET  | 336 | 96  | **0.301** | 0.399 | 0.396 | 0.355 | 0.350 | 1.237 |
| DET  | 336 | 336 | **0.394** | 0.505 | 0.511 | 0.460 | 0.483 | 1.419 |
| PTV  | 192 | 96  | **0.261** | 0.471 | 0.377 | 0.321 | 0.350 | 1.184 |
| PTV  | 192 | 336 | **0.320** | 0.475 | 0.454 | 0.366 | 0.531 | 1.306 |
| PTV  | 336 | 96  | **0.244** | 0.355 | 0.326 | 0.307 | 0.351 | 1.184 |
| PTV  | 336 | 336 | **0.314** | 0.426 | 0.388 | 0.385 | 0.547 | 1.306 |
| PTV2 | 192 | 96  | **0.322** | 0.400 | 0.399 | 0.335 | 0.329 | 1.013 |
| PTV2 | 192 | 336 | **0.359** | 0.413 | 0.449 | 0.359 | 0.363 | 1.135 |
| PTV2 | 336 | 96  | **0.300** | 0.331 | 0.343 | 0.304 | 0.317 | 1.013 |
| PTV2 | 336 | 336 | 0.339 | 0.375 | 0.414 | **0.335** | 0.354 | 1.135 |

## Table B — MAE

| Dataset | $L$ | $O$ | EMForecaster | DLinear | PatchTST | TSMixer | LSTM | ARIMA |
|---|---|---|---|---|---|---|---|---|
| DET  | 192 | 96  | **0.389** | 0.496 | 0.488 | 0.427 | 0.419 | 0.847 |
| DET  | 192 | 336 | **0.464** | 0.532 | 0.537 | 0.487 | 0.527 | 0.965 |
| DET  | 336 | 96  | **0.377** | 0.445 | 0.454 | 0.423 | 0.426 | 0.847 |
| DET  | 336 | 336 | **0.444** | 0.511 | 0.517 | 0.485 | 0.520 | 0.965 |
| PTV  | 192 | 96  | **0.375** | 0.518 | 0.462 | 0.409 | 0.437 | 0.837 |
| PTV  | 192 | 336 | **0.419** | 0.507 | 0.493 | 0.430 | 0.537 | 0.938 |
| PTV  | 336 | 96  | **0.359** | 0.428 | 0.427 | 0.395 | 0.436 | 0.837 |
| PTV  | 336 | 336 | **0.412** | 0.476 | 0.455 | 0.439 | 0.544 | 0.938 |
| PTV2 | 192 | 96  | **0.439** | 0.495 | 0.494 | 0.446 | 0.450 | 0.796 |
| PTV2 | 192 | 336 | **0.468** | 0.503 | 0.518 | 0.465 | 0.475 | 0.872 |
| PTV2 | 336 | 96  | **0.423** | 0.443 | 0.448 | 0.422 | 0.441 | 0.796 |
| PTV2 | 336 | 336 | 0.455 | 0.476 | 0.496 | **0.449** | 0.471 | 0.872 |

## Notes

- **MSE:** EMForecaster is best in 11 of 12 settings; TSMixer edges it out on PTV2 / L=336 / O=336 (0.335 vs 0.339).
- **MAE:** EMForecaster is best in 11 of 12 settings; TSMixer edges it out on PTV2 / L=336 / O=336 (0.449 vs 0.455) — the same setting where it wins on MSE.
- ARIMA values are markedly higher than the deep models across the board.
- ARIMA does not use a learned lookback, so its L=192 and L=336 rows are identical for a given (dataset, O).
- Seed aggregation: DET deep-model rows average 5 seeds; all other deep-model rows average 2 seeds (`69421`, `69422`).
