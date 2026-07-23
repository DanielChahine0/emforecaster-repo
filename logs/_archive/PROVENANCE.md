# Archive provenance

## 2026-07-23 — accidental deletion of the previous archive

The previous `logs/_archive/` tree (`2026-07-22/` and `2026-07-22-pass2-revinless/`)
was deleted on 2026-07-23. It was never tracked by git, so it is unrecoverable.
`logs/grid_report_multivariate.json` was deleted in the same operation and is also
unrecoverable (it was untracked). `logs/grid_run_2026-07-22.log` was tracked and has
been restored.

### What survives

The `revin: False` RF_EMF_Transformer sweep is **still recoverable from git history**
(branch `main`, i.e. any commit before `454c50c`), because those `log.json` files were
tracked. Retrieve with:

    git show main:logs/multivariate/RF_EMF_Transformer/<ds>/<cfg>_cp/69421/log.json

| cell | revin=False (main) | revin=True (reported) | ic False → True |
|---|---|---|---|
| ed2022/L192_O96   | 0.8330 | 0.4976 | 82.42 → 85.54 |
| ed2022/L192_O336  | 1.2907 | 0.9551 | 85.25 → 85.62 |
| ed2022/L336_O96   | 0.8435 | 0.5027 | 83.34 → 85.36 |
| ed2022/L336_O336  | 1.2891 | 0.9826 | 85.28 → 85.74 |
| uh2022/L192_O96   | 1.3897 | 1.1471 | 85.28 → 86.77 |
| uh2022/L192_O336  | 1.7947 | 1.5511 | 83.26 → 85.09 |
| uh2022/L336_O96   | 1.3881 | 1.1432 | 85.22 → 86.72 |
| uh2022/L336_O336  | 1.7940 | 1.5720 | 83.28 → 85.46 |
| emf2023/L192_O96  | 1.6102 | 1.5095 | 87.00 → 90.21 |
| emf2023/L192_O336 | 0.9673 | 0.8630 | 88.05 → 90.86 |
| emf2023/L336_O96  | 1.6221 | 1.5080 | 87.16 → 90.58 |
| emf2023/L336_O336 | (never committed) | 0.8568 | — |

### What is lost

`2026-07-22-pass2-revinless/` was a **second, identical-config rerun** of the 8
ed2022/uh2022 Transformer cells. Its only unique value was as evidence that the
pipeline is not seed-deterministic. Numbers extracted before deletion:

| cell | pass 1 | pass 2 | delta | epochs |
|---|---|---|---|---|
| ed2022/L192_O96  | 0.868138 | 0.833015 | +4.046% | 94 vs 76 |
| uh2022/L192_O96  | 1.404332 | 1.389693 | +1.042% | 83 vs 100 |
| ed2022/L192_O336 | 1.2906526 | 1.2906523 | 1e-7 | — |
| uh2022/L192_O336 | diverges at epoch 0, converges to identical MSE | | | |
| ed2022/L336_O96, ed2022/L336_O336, uh2022/L336_O96, uh2022/L336_O336 | bit-identical trajectories | | | |

Both passes used seed 69421 and identical configs (`revin=False`, `batch_size=128`;
`accum_steps` None vs 1 is a mathematical no-op). The two divergent cells start
differing in the 6th decimal at epoch 0 and amplify through early stopping into
different stopping epochs and different reported MSE.

Root enablers, all in `forecasting/multivariate/exp/exp.py:122-127`:
`torch.backends.cuda.matmul.allow_tf32 = True`, `torch.backends.cudnn.allow_tf32 = True`,
`torch.set_float32_matmul_precision("high")`, `torch.backends.cudnn.benchmark = True`,
and `torch.compile` enabled for RF_EMF_Transformer and PatchTST only. There is no
`torch.use_deterministic_algorithms` call anywhere in the repo.

This measurement should be redone as part of the planned multi-seed sweep rather than
reconstructed — a 4% spread matters because EMForecaster's winning margin is smaller
than that in several cells.

## 2026-07-23 — cells archived as invalid

28 distinct cells were moved out of `logs/multivariate/` so that `grid.py`'s
skip-if-`log.json`-exists check will actually re-run them.

### `2026-07-23-transformer-dead-encoder/` — 12 cells

All RF_EMF_Transformer cells, every dataset and config. The encoder never trains at
any horizon. In `forecasting/multivariate/models/rf_emf_baselines.py`:

- `EncoderBlock(seq_len, pred_len, d_model, num_heads)` is called against the
  signature `(seq_len, d_ff, d_model, num_heads, ff_dropout)`, so `pred_len` binds to
  `d_ff`. Confirmed in the checkpoints: `enc.0.ff.0.weight` is `(336,2)` at O336 and
  `(96,2)` at O96. Configured `d_ff: 256` and `ff_dropout: 0.2` never reach the model.
- `nn.LayerNorm(d_model)` with `d_model=2` outputs exactly `(±1, ∓1)` — a sign function
  with a.e. zero Jacobian. Measured gradient norm into `fc1.weight` is 4.5e-08 and into
  `enc.0.ff.0.weight` 1.3e-08 in both O96 and O336 checkpoints.
- The second residual adds `attn_out` rather than `attn_norm`, and one `self.norm`
  module is reused for both add-and-norm sites.

Because `d_ff = pred_len` shifts the RNG stream, whether the sign bit ever flips is a
deterministic function of `pred_len`: threshold `|x*|` = 0.104σ at P=96 (alive) vs
5.24σ at P=336 (dead), identical for L=192 and L=336. Loading the six O336 checkpoints
gives forecast std **across the batch of 0.000000** — every window yields the identical
336-vector, with `corr(forecast mean, input window mean)` = 0.999. A 60-seed sweep puts
the dead-encoder rate at 35–48% at *both* horizons, so the O96 cells are luck, not
training.

The `revin: False -> True` change was **not** the cause; the dead rate is the same with
and without RevIN. The comment claiming otherwise in
`forecasting/multivariate/jobs/exp/rf_emf_mv/transformer/args.yaml` is wrong.

### `2026-07-23-emf2023-mixed-support/` — 16 cells

All emf2023 cells for DLinear, EMForecaster, PatchTST and TSMixer. (The 4
RF_EMF_Transformer/emf2023 cells are in the archive above and need both fixes.)

O96 and O336 are not evaluated on the same data. `utils/dataloading.py:343` sets
`test_idx = [len(data) - num_test - seq_len, len(data)]`, independent of `pred_len`,
but `utils/datasets.py:56-62` then drops any window start whose `seq_len + pred_len`
span touches the outage mask. emf2023 has a 10,013-row flat-filled gap at global rows
63273–73285, so every pre-outage O336 start is masked while the O96 starts survive:

- L192_O96 → 5346 windows, first start row 62777
- L192_O336 → 4897 windows, first start row 73286
- L336_O96 → 5202 windows, first start row 62633
- L336_O336 → 4753 windows, first start row 73286

The O96 test set therefore includes 209 March windows the O336 set never sees. Within
them, row 63044 / channel 3547.5 MHz reads 0.598226 against a train σ of 0.00187 →
z = +316.87. It falls in the target of 76 O96 windows and contributes 0.513 MSE alone.
A zero-forecast with no model reproduces the whole inversion (1.8795 O96 vs 0.9666
O336); restricted to the shared post-outage support it vanishes (0.9779 vs 0.9666).

The spike is genuine data — verified present verbatim in the unprocessed source file,
not a sentinel, NaN fill or unit change.

Fix before re-running: compute valid test origins once using the grid's `max(pred_len)`
and reuse that origin set for every horizon.

## Not archived — verified clean

The 32 ed2022/uh2022 cells for DLinear, PatchTST, TSMixer and EMForecaster remain in
`logs/multivariate/` (4 models × 2 datasets × 4 configs). The 2026-07-21 → 2026-07-22 dataloading changes are a provable
no-op on those two datasets (both processed CSVs have zero missing values, so the
outage masking path never activates). They do not need re-running on provenance
grounds. They *will* need re-running if the DLinear learning rate (`lr: 1e-5`, 10×
below every other model) is corrected, or under a multi-seed protocol.
