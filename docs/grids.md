# Experiment grids

The detailed reference for grid mode (`python main.py --grid`).
For a short overview and quickstart, see the [root README](../README.md).
The grids themselves are defined in [`forecasting/grids.py`](../forecasting/grids.py).

## Contents

- [The grids at a glance](#the-grids-at-a-glance)
- [Flags](#flags)
- [Notes on each grid](#notes-on-each-grid)
- [Seeds and reproducibility](#seeds-and-reproducibility)
- [Running the per-channel grid faster with CUDA MPS](#running-the-per-channel-grid-faster-with-cuda-mps)
- [More examples](#more-examples)

## The grids at a glance

A grid is every model x dataset x window combination (plus a `k` dimension for the cluster grids).
Each combination is one **cell**.
The exit code is non-zero if any cell failed.

| `--setup` | What one cell trains | Datasets | Windows (`LxO`) | Models | Cells |
|---|---|---|---|---|---|
| `multivariate` | One model over all 27/29 frequency bands jointly | `ED2022`, `UH2022`, `EMF_2023_university_5_with_PTV` | `192x96`, `192x336`, `336x96`, `336x336` | 7 | 84 |
| `univariate` | One independent model per band, averaged back into one number | same as `multivariate` | same | same 7 | 84 |
| `univariate_series` | One model on a single exposure series | `rf_emf_det`, `rf_emf_ptv`, `rf_emf_ptv2` | same | 7 | 84 |
| `cluster` | The multivariate network with CCM channel clustering, at each `k` | same as `multivariate` | same | 6 | 360 (`k` in 1, 2, 3, 5, 9) |
| `voip` | Like `multivariate`, on VoIP traces | `VOIP_MOBILE`, `VOIP_FIXED` | `48x12`, `24x6` | 8 | 32 |
| `voip_perchannel` | Like `univariate`, on VoIP traces | same as `voip` | same | 8 | 32 |
| `voip_cluster` | Like `cluster`, on VoIP traces | same as `voip` | same | 6 | 96 (`k` in 1, 2, 3, 6) |

The seven models are EMForecaster, DLinear, PatchTST, TSMixer, Transformer, LSTM and ARIMA.
The cluster grids drop ARIMA (6 models).
The VoIP grids add Persistence (8 models).

If `--setup` is omitted, you are prompted to pick one of the four EMF grids (`u`/`m`/`s`/`c`).
The VoIP grids must be named explicitly.

Defined is not the same as run.
Of the four EMF windows, the live `logs/<setup>/` trees hold only the `336x96` benchmark for `multivariate`, `univariate` and `cluster`.
The L-ladder rungs at other windows sit under `logs/_archive/` (the cluster ones are upstream-era) and are not benchmark runs.
So the three-paradigm EMF comparison exists at `336x96` alone, while `univariate_series` has all four windows.

## Flags

| Flag | Applies to | What it does |
|---|---|---|
| `--setup <name>` | all grids | Which grid to run (see the table above). |
| `--model <list>` | all grids | Only these models. Comma-separated, case-insensitive, matched against the display label or the on-disk `model_id` (`--model dlinear`, `--model RF_EMF_Transformer`). |
| `--dataset <list>` | all grids | Only these datasets. Comma-separated, case-insensitive, matched against the dataset name or its `logs/` folder slug (`--dataset ED2022`, `--dataset emf2023`). |
| `--window <list>` | all grids | Only these windows. |
| `--k <list>` | `cluster`, `voip_cluster` | Only these cluster counts. |
| `--seeds <spec>` | all grids | Train each cell on more than one seed. |
| `--jobs <int>` | all grids | How many trainings run at once. |
| `--dry_run` | all grids | List the cells and their target log directories without training anything. |

`--model`, `--dataset`, `--window` and `--k` keep exactly the cells the full sweep would have run, so a trial run's results drop into the same tables.

### `--window`

Each window is written as `LxO` (`336x96`), `L_O` (`336_96`), or the `logs/` directory form `L336_O96`.
Comma-separate to select several, for example `--window 336x96,336x336`.
A malformed spec is a clean error up front rather than a mid-run failure.
A well-formed window outside the grid's own list runs as a one-off, so `--window 720x96` works without editing `forecasting/grids.py`.

### `--k`

A single value (`--k 5`) or a comma-separated list (`--k 1,2`).
Duplicates collapse and the values run in ascending order.
A requested `k` need not be one of the grid's own `k_values`.
`k = num_channels` on the EMF grid (one cluster per band) is not swept, but `--k 27` reaches it, or any other count, as a one-off.
An off-grid value is logged as `[NOTE ] ... running as a one-off` so it is never mistaken for part of the sweep.
The `k_values` actually covered are recorded in `logs/grid_report_<setup>.json`.
Passing `--k` to a grid without a `k` dimension is a clean error.

### `--seeds`

The default is one seed, the canonical `69421`.
A count counts up from the canonical seed, so `--seeds 3` runs `69421`, `69422` and `69423`.
A comma-separated list selects those seeds exactly, for example `--seeds 69422,69423` to top up a grid that already has `69421`.
A lone number is always a count, so a single explicit seed needs a trailing comma.
`--seeds 69421,` runs that seed alone, while a bare `--seeds 69421` plans 69,421 seeds.

Resume is per `(cell, seed)`, so adding seeds to a finished sweep re-runs only the new ones; the existing runs are skipped, not retrained.
Seeds run seed-major (the whole grid at one seed, then the next), so an interrupted multi-seed sweep leaves a complete grid at one seed rather than a handful of cells with error bars.
On the per-channel grid a seed multiplies *band* runs: a second seed on the full `--setup univariate` grid is another ~1,700 trainings.

### `--jobs`

On the per-channel grid (`--setup univariate`), this is how many band trainings run at once (default 8).
On every other grid it is how many whole cells run at once (default 1, the in-process sequential driver).
Each concurrent training runs in its own subprocess.

Expect well under a linear speedup.
Measured on an RTX 5090, `--jobs 4` on a DLinear/PatchTST/TSMixer cluster sweep slowed each cell 3.15x and netted only 1.2x, because these models issue many small kernels and concurrent processes just time-slice the GPU.
Cheap models (DLinear) parallelize better than heavy ones (PatchTST).
Size `--jobs` by the peak memory of the heaviest model in the sweep.
EMForecaster and the Transformer need most of the card for one cell, so keep `--jobs` at 1-2 on a grid containing them, and lower it if the GPU runs out of memory.

## Notes on each grid

### `multivariate`

One model per cell, forecasting all 27/29 frequency bands jointly.

PatchTST left this sweep on 2026-07-24 and rejoined it on 2026-08-03.
Its pre-drop cells are not on disk: commit `e84faec4` deleted `logs/multivariate/PatchTST/` along with the other three windows of every model, leaving only the `L336_O96` slice.
`git checkout e84faec4^ -- logs/multivariate` restores the old tree.

### `univariate` (per channel)

The per-band counterpart of the multivariate grid: same models, datasets and windows.
Each cell trains one independent model per frequency band and averages the bands back into one number per cell, so its cells are directly comparable to the multivariate ones.
It is driven by `forecasting/perchannel.py`, which fans the band trainings out to a process pool.
It shares the multivariate grid's model list outright (`UNIVARIATE_MODELS is MV_MODELS`) and reuses the same `rf_emf_mv/*` job configs one band at a time.
Its runs are written under `logs/univariate (per channel)/`, so they never mix with the single-series results under `logs/univariate/`.

### `univariate_series`

The original single-series sweep over the DET/PTV monitoring datasets (`rf_emf_det`, `rf_emf_ptv`, `rf_emf_ptv2`).
It has the same seven models, but its own job configs under `forecasting/univariate/jobs/exp/rf_emf/`, because it is the only grid that runs the `forecasting.univariate` package (see `grid.SETUP_PACKAGE`).

Its LSTM is `rf_emf/lstm` (model_id `LSTM`, which builds `RecurrentModel(backbone_id="LSTM")`), the same model the other grids run.
It is *not* the frozen `rf_emf/baselines/lstm` (`RF_EMF_LSTM`) that produced the published "LSTM (1997)" column, so do not merge those two columns.

### `cluster`

The CCM channel-clustering sweep: same datasets and windows as `multivariate`, with the learned, soft channel clustering of the [Channel Clustering Module (NeurIPS 2024)](https://arxiv.org/abs/2404.01340) threaded through this repo's own model implementations.
Runs are written to `logs/cluster/<model>/<dataset>/L<L>_O<O>_cp/k<k>/<seed>/`.

#### How it works

The clustering modules, `Cluster_assigner` and `Cluster_wise_linear`, are the paper's own code.
They are imported verbatim from the vendored `TimeSeriesCCM-main/` tree (`forecasting/multivariate/utils/ccm_upstream.py`) and orchestrated by `forecasting/multivariate/models/ccm.py`.

1. Each channel's window is projected to `d_model`.
2. It is scored by cosine similarity against `k` trainable cluster embeddings.
3. The scores are sinkhorn-normalized into a `[num_channels, k]` distribution.
4. A linear map the model applies to each channel on its own is swapped for `k` parallel cluster-wise copies, mixed by that assignment.
5. The cluster embeddings are refreshed from each training batch by cross-attention.

The objective is `forecast_loss + beta * similarity_loss` (`cluster.beta`, default `0.3` from upstream).
The similarity term pulls the assignment toward the data's own channel-similarity structure.

#### Models

The six models are every gradient-trained model of the multivariate grid.
DLinear, PatchTST and TSMixer each have a per-channel temporal linear for CCM to route: DLinear's seasonal/trend linears, PatchTST's flatten head, and TSMixer's temporal MLPs and head.
These are the same places upstream's own wrappers put `Cluster_wise_linear`.
For EMForecaster, the Transformer and LSTM, which upstream has no version of, CCM clusters the Linear read-out their channels share, following upstream's PatchTST precedent.

All six are in `MV_MODELS`, so every cluster cell has same-window multivariate and per-channel baselines beside it.
The cluster cell is the multivariate cell's exact network (same config, same init, same RevIN), differing only in the clustering module.
That is what makes a cluster-vs-multivariate (or vs per-channel) comparison attributable to the *paradigm* rather than to implementation drift.

ARIMA and Persistence (per-band by construction, with no gradient path) are rejected at model build instead of being silently adapted.
So is TimesNet: it embeds all channels jointly from the first layer, so there is nothing per-channel to route, and it never had a multivariate baseline.

#### Divergences from the paper's released code

Only the clustering *layers* run from `TimeSeriesCCM-main/`; its model wrappers and Exp pipeline do not.

1. **The networks are ours.**
   Until 2026-08-25 the grid ran upstream's own `DLinearC`/`PatchTSTC`/`TSMixerC`/`TimesNetC`, whose implementations differ from the multivariate cells'.
   For example, `DLinearC` hardcodes a decomposition kernel of 25 where the job config sets `moving_avg: 3`, and skips the repo's `1/seq_len` constant init.
   Runs from that era are not comparable to current ones and must not share a logs tree with them.
   The upstream-era runs that were kept, the L-ladder rungs among them, sit under `logs/_archive/`, and `logs/cluster/` holds only runs of the native implementation.
   TimesNet left the grid with the wrappers.
2. **PatchTST's encoder is not clustered.**
   Upstream's `PatchTSTC` passes the cluster embeddings into every encoder layer, but the cluster-token attention that would use them is commented out in the release (`TimeSeriesCCM-main/models/patch_layer.py:355-367`).
   So upstream's encoder ignores them and in effect only the head is clustered, as in ours.
   Our encoder stays byte-identical to the multivariate cell's; upstream's layers still build the unused cluster-attention modules, which ours do not.
3. **The similarity term is trained on but kept out of the reported loss.**
   It is excluded from `sl_{train,val,test}/loss` and from the best-model / early-stopping metric, which stay the pure forecast loss.
   It is logged separately as `sl_train/similarity_loss`.
   Upstream (`Exp_CCM.train`) early-stops on the combined value, but at 27 channels the similarity term runs 15-250 against an MSE around 0.5, so combining them hands model selection to the regularizer.
   On an ED2022 `L336_O96` sweep, upstream's rule early-stopped `k=2` at epoch 27 having selected epoch 7, while the plain DLinear baseline kept improving for all 100 epochs.
   Keeping them separate also means a cluster cell's loss is comparable to every other cell in every grid.
   See `compute_aux_loss` in `forecasting/multivariate/utils/models.py`.
4. **The cluster-embedding refresh is pooled over the batch.**
   Upstream's refresh returns a per-example `[batch, k, d_model]` tensor that the released wrappers write back into the `[k, d_model]` slot, which is fatal on training batch two as released.
   `CCMForecaster` pools it over the batch (the assigner's own batch-mean convention) and holds it in a registered buffer, so it checkpoints and moves with the module.
5. **Channel similarity runs along the channel axis.**
   Upstream's batch-wise similarity matrix only has the right shape in its M4/stock setups; the same kernel is applied along the channel axis instead (`CCMForecaster.channel_similarity`).

#### The `k=1` baseline

The `k=1` cell, part of the sweep, is the controlled no-clustering baseline.
At `k=1` sinkhorn returns an all-ones assignment, so every cluster-wise linear collapses to exactly one shared channel-independent map, and the similarity loss becomes an additive constant with no gradient.
Every difference between `k=1` and `k>1` is then attributable to the clustering itself.
Because the backbone is the multivariate implementation, the plain multivariate column is a fair comparison too.

`k=1` matches the multivariate cell within seed noise, not bit for bit.
The residual difference is the assigner, since all six models already share their per-channel maps across channels (PatchTST runs `channel_independent: True`).
The assigner's parameters are drawn after the backbone's, which starts from the multivariate cell's exact init.
But they, and the random draws the assigner and the similarity loss make each step, shift every later draw from the global RNG.
For PatchTST, TSMixer, EMForecaster and LSTM that moves their dropout masks.
DLinear and the Transformer have no dropout, and the data loaders shuffle from a generator of their own, so their gap is floating-point rounding alone (the one-copy blend is an extra matmul, TF32 on the GPU, and no run requests deterministic kernels).

#### Two evaluation protocols; the fixed one is primary

Upstream's assigner pools the cluster membership over the windows in a batch, in evaluation as well as in training.
So a cluster cell's test forecast depends on which windows share its batch: up to `batch_size - 1` others.
On the sequential test loader, the later ones among them (up to `seq_len + pred_len - 1`) have lookbacks that overlap its target.

So after the best checkpoint is restored, a cluster run also fits one **fixed** membership per run.
It is the mean of every training window's assigner scores, then upstream's sinkhorn: its batch mean with the whole training split as the batch.
Validation and test are scored under that fixed membership, with split-conformal recalibrated on validation under the same membership.

- `sl_val_fixed/*` and `sl_test_fixed/*` are the primary cluster result.
  `sl_val_fixed/loss` is the validation loss to pick `k` by.
- `sl_val/*` and `sl_test/*` keep upstream's batch protocol, unchanged, as a sensitivity row.

The batch protocol pools over the evaluation micro-batch, which is not the same for every model.
On the EMF grid it is 128 windows for EMForecaster (the 128-window micro-batches with 4-step gradient accumulation that `grid._memory_profile` gives it at `L336` and on every EMF2023 window) and for the Transformer (its job config's batch), against 512 for DLinear, PatchTST, TSMixer and LSTM.
On VoIP every model pools 64 (`VOIP_OVERRIDES`).
The fixed protocol does not depend on the batch size, which is one more reason it is the primary one.
The checkpoint's epoch is still the one training chose on the batch protocol's validation loss.

At `k=1` the membership is all ones either way, so the two protocols give the same forecasts.
The test metrics agree exactly, and the validation loss up to summation order, because the validation loader shuffles.

For cluster runs trained before the fixed protocol existed, rescore the saved checkpoints on the CPU without retraining:

```bash
python -m forecasting.ccm_rescore logs/cluster --mode fixed
```

It first checks that its rebuild of the run reproduces the logged `sl_test/{loss,mse,mae}`, then writes an `eval_fixed.json` sidecar beside the run.
It never edits `log.json`.

#### History

This grid replaced a KMeans-partition head on 2026-08-06.
The k1/k9 cells in commits `2ebc7f08` / `3d2de887` (deleted in `f24cf7a2`) came from that head and are not resumable against the current config.

### `voip`, `voip_perchannel`, `voip_cluster`

The multivariate, per-channel and CCM-clustering paradigms over the pooled VoIP datasets (`VOIP_MOBILE`, `VOIP_FIXED`; 6 channels).
They use windows `L48_O12` and `L24_O6`, and `k_values = [1, 2, 3, 6]`, from the `k = 1` baseline to one cluster per channel.
`voip_cluster` carries the same six models as `cluster`.

They run the same drivers, job configs and log trees as their EMF counterparts.
A VoIP cell lands in `logs/multivariate/<model>/voip_mobile/...` beside the EMF ones, one dataset slug apart.

They add a **Persistence** column (`rf_emf_mv/persistence`, `exp.no_train`), which the EMF grids deliberately do not carry.
Four of the six VoIP channels repeat their previous value in 80-99% of steps, so without a last-value baseline a low error cannot be told apart from having learned nothing.

Every cell of these three grids also takes `grids.VOIP_OVERRIDES` on top of its job config: `sl.batch_size` 64, `data.train_split` 0.6, `data.val_split` 0.2.
The validation split is wider because a 10% validation split of a 400-row trace leaves too few windows to calibrate a conformal interval on.
The batch is smaller because a pooled VoIP train split is ~1,900 windows, where a 512 batch would be four optimizer steps per epoch.

## Seeds and reproducibility

In single-job mode `main.py` iterates over every seed in the config's `exp.seed_list`, building a fresh `Experiment` per seed.
Grid mode ignores `seed_list` and pins the seeds itself (the canonical `69421` unless `--seeds` asks for more), so a cell's log path is stable and the resume check can find it.

A seed pins the initialization, not the GPU arithmetic: no run requests deterministic kernels.
The logged multivariate Transformer run at ED2022 `L336_O96`, seed `69421` (MSE 0.5060 after 99 epochs), does not reproduce on today's GPU stack.
Reruns at the code that logged it and at current code agree with each other at MSE 0.4950 after 79 epochs.
So the Transformer cells carry about 2% GPU run-to-run noise on top of their known seed collapse.
Only `69421` trains; at `69422` and `69423` the multivariate Transformer is dead from init on all three EMF datasets.

## Running the per-channel grid faster with CUDA MPS

A per-channel band is a tiny single-channel model, so its kernels are small and short.
With several bands in flight the GPU time-slices between their processes: `nvidia-smi` reports ~97% utilization while the card draws about 58% of its power limit and most of its memory sits idle.
NVIDIA's Multi-Process Service (MPS) lets those processes' kernels run on the GPU at the same time instead.
It changes scheduling, not arithmetic: bands rerun under it are bit-identical.
On the RTX 5090 it measured 1.19x faster per band at `--jobs 8`, then a further 1.28x in total throughput at `--jobs 12` (Transformer bands, each still well under the 1.5x slowdown that would cancel the extra concurrency).

Start the daemon on a private pipe directory, so only processes you give that directory attach and nobody else's GPU jobs are affected on a shared machine:

```bash
export CUDA_MPS_PIPE_DIRECTORY=/tmp/mps-$USER/pipe CUDA_MPS_LOG_DIRECTORY=/tmp/mps-$USER/log
mkdir -p "$CUDA_MPS_PIPE_DIRECTORY" "$CUDA_MPS_LOG_DIRECTORY"
nvidia-cuda-mps-control -d                     # start (serves only this pipe directory)
python main.py --grid --setup univariate --window 336x96 --jobs 12
echo quit | nvidia-cuda-mps-control            # stop it when the sweep is done
```

It only pays for many small concurrent processes.
A single multivariate or cluster cell gains nothing, and neither does ARIMA, which fits on the CPU.
The levers that would be faster still (a larger batch, mixed precision, `torch.compile`, fewer epochs) all change the numbers, so they are not used on benchmark runs.

Each run records in its `log.json` whether it ran under MPS, as `parameters/exp/cuda_mps` (true when a daemon was listening on the pipe directory the run was given).
It also records that directory, as `parameters/exp/cuda_mps_pipe_dir`, so its timing is read against runs that shared the GPU the same way.

## More examples

```bash
# Full multivariate sweep, all four windows
python main.py --grid --setup multivariate

# Per-band grid, one window only, 4 concurrent band trainings
python main.py --grid --setup univariate --window 336x96 --jobs 4

# Preview the single-series sweep for two windows without training
python main.py --grid --setup univariate_series --window 192x96,336x96 --dry_run

# Single-series sweep on one window, three seeds (69421-69423) for error bars
python main.py --grid --setup univariate_series --window 336x96 --seeds 3

# Cluster sweep, one window, every k in the grid (90 cells)
python main.py --grid --setup cluster --window 336x96

# Cluster at k=1 only (one cluster for every channel), the controlled baseline
python main.py --grid --setup cluster --window 336x96 --k 1

# Trial run: DLinear+CCM on one dataset/window, sweeping k (4 cells).
# --k 1 matches the plain DLinear cell below within seed noise.
python main.py --grid --setup cluster \
    --model DLinear --dataset ed2022 --window 336x96 --k 1,2,4,6

# The plain DLinear cell it sits beside, if not already on disk
python main.py --grid --setup multivariate \
    --model DLinear --dataset ed2022 --window 336x96

# Rebuild the comparison tables (adds a k column on the cluster tree)
python results/make.py extract && python results/make.py tables
```
