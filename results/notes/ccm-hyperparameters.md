# Clustering (CCM) runs — exact hyperparameters and where they live

Generated 2026-08-11 from the repo source and from the frozen `args.yaml` files
under `logs/cluster/`. Revised 2026-08-14: the lookback ladder was restored (§6)
and the excluded `beta = 0.05` group documented (§8), taking the published set
from 105 runs to 132.

The cluster grid replaces the network with CCM ("From Similarity to Superiority:
Channel Clustering for Time Series Forecasting", NeurIPS 2024), imported verbatim
from the vendored `TimeSeriesCCM-main/` tree. Everything else about a run (data,
optimizer, conformal recalibration, logging) is the multivariate grid's.

---

## 0. Short answer: config-driven or hardcoded upstream?

**Both, split along a clean line.** `build_ccm_args`
(`forecasting/multivariate/models/ccm.py:62-114`) builds upstream's flat argparse
`Namespace` from our nested config, so everything upstream reads *through `args`*
is ours. Everything upstream writes as a literal inside its own layers is theirs,
and no config field can reach it.

| Comes from **our** config (`args.yaml` → `build_ccm_args`) | Hardcoded in **`TimeSeriesCCM-main/`**, not configurable |
| --- | --- |
| `in_len` / `out_len` (336 / 96) | sinkhorn ε = 0.05, single normalisation pass |
| `data_dim` (27 / 29), `batch_size` (512) | concrete-Bernoulli temperature 0.07 |
| `n_cluster` = `cluster.k_clusters` | cosine-similarity channel scoring (L2-norm, `dim=1`) |
| `d_model`, `d_ff` (via `_backbone_dims`) | cluster-embedding init `kaiming_uniform_(a=√5)` |
| `n_heads`, `n_layers`, `dropout`, `attn_dropout` | 1-head cross-attention embedding refresh |
| `patch_len` / `stride` (16 / 8) | DLinear decomposition kernel = **25** |
| `individual = "c"` (set by the adapter) | PatchTST RevIN(`affine=True`, `subtract_last=False`) |
| `cluster.beta` — applied repo-side in `compute_aux_loss`, never passed to upstream | |
| | PatchTST `pe='zeros'`, `learn_pe=True`, `norm='BatchNorm'` |
| | `Flatten_Head(head_dropout=0.0)`, `d_k=d_v=None` |
| | TSMixer's RevIN calls (commented out upstream) |
| | k parallel `nn.Linear` head (`Cluster_wise_linear`) |

A third, smaller category: **repo-side literals in `build_ccm_args`** that upstream
requires but our config has no field for — `padding_patch="end"`, `max_seq_len=1024`,
`pre_norm=False`, `pretrain_head=False`, `freq="h"`.

The consequence that matters most: because several config fields are read by a
*different name* upstream (or not at all), **the raw `args.yaml` overstates what
was used**. See [§4](#4-per-backbone-hyperparameters-as-actually-consumed-by-ccm)
and [§7](#7-gotchas-when-quoting-these-numbers).

---

## 1. Files to look at

A cluster run's hyperparameters come from four places, in this order of precedence:

1. `TimeSeriesCCM-main/models/*` — upstream constants, hardcoded
2. `forecasting/multivariate/jobs/exp/rf_emf_mv/<model>/args.yaml` — backbone / optimizer / data
3. `forecasting/multivariate/config/config.py` — pydantic defaults for anything the job YAML omits (this is where several of the values that actually matter come from)
4. `forecasting/grid.py` — per-cell overrides applied by the sweep

The single authoritative record of what a given run used is the frozen copy
written beside its outputs:

```
logs/cluster/<model>/<dataset>/L<L>_O<O>_cp/k<k>/<seed>/args.yaml
```

### Repo-side config

| File | What's there |
| --- | --- |
| `forecasting/multivariate/config/config.py:778-819` | `class Cluster` (`enabled`, `k_clusters`, `beta`) + design notes |
| `forecasting/grids.py:156-160` | `GRIDS["cluster"]` — models, datasets, windows, alpha, `k_values` |
| `forecasting/grids.py:100-105` | `CLUSTER_MODELS` |
| `forecasting/grids.py:38` / `:44` | `WINDOWS` / canonical `SEED` |
| `forecasting/grid.py:333-343` | in-process per-cell overrides (k, compile off) |
| `forecasting/grid.py:135-158` | identical overrides on the subprocess path |
| `README.md` → "Grid mode" → `cluster` | prose description of the whole design + caveats |

### Repo-side model / loss glue

| File | What's there |
| --- | --- |
| `forecasting/multivariate/models/ccm.py:47-60` | `_backbone_dims` — which config fields become CCM's `d_model` / `d_ff` |
| `forecasting/multivariate/models/ccm.py:62-114` | `build_ccm_args` — the full Namespace handed upstream |
| `forecasting/multivariate/models/ccm.py:238-306` | similarity kernel, similarity loss, concrete-Bernoulli temperature |
| `forecasting/multivariate/utils/models.py:44-52` | `get_model` → `CCMForecaster` |
| `forecasting/multivariate/utils/models.py:450-480` | `compute_aux_loss` (where `beta` is applied) |
| `forecasting/multivariate/utils/models.py:348-355` | `OneCycleLR` construction |
| `forecasting/multivariate/exp/exp.py:495` / `:510` | aux loss added to the backward loss only |

### Upstream (hardcoded, not configurable)

| File | What's there |
| --- | --- |
| `TimeSeriesCCM-main/models/layers.py:262-313` | `Cluster_assigner` + `sinkhorn` |
| `TimeSeriesCCM-main/models/patch_layer.py:244-267` | `Cluster_wise_linear` |
| `TimeSeriesCCM-main/models/Dlinear.py:40-73` | `DLinearC` |
| `TimeSeriesCCM-main/models/tsmixer.py:59-94` | `TSMixerC` |
| `TimeSeriesCCM-main/models/patchtst.py:9-58` | `PatchTSTC` |
| `TimeSeriesCCM-main/models/patch_layer.py:14-53` | `Patch_backbone` (PatchTST) |

### Job configs (backbone + training)

```
forecasting/multivariate/jobs/exp/rf_emf_mv/dlinear/args.yaml
forecasting/multivariate/jobs/exp/rf_emf_mv/patchtst/args.yaml
forecasting/multivariate/jobs/exp/rf_emf_mv/tsmixer/args.yaml
forecasting/multivariate/jobs/exp/rf_emf_mv/timesnet/args.yaml   # never run
```

---

## 2. How the runs were launched

```bash
python main.py --grid --setup cluster --window 336x96 --k <k> [--dataset ...] [--seeds ...]
```

Overrides the grid applies on top of the job YAML, per cell (`grid.py:325-343`):

| Field | Value |
| --- | --- |
| `data.dataset` | `ED2022` \| `UH2022` \| `EMF_2023_university_5_with_PTV` |
| `data.seq_len` / `data.pred_len` | 336 / 96 (from `--window`) |
| `cluster.enabled` | `True` |
| `cluster.k_clusters` | `<k>` (from `--k`; also becomes `exp.id = "k<k>"`) |
| `exp.compile` | `False` — forced off: CCM rebinds module attributes inside `forward`, which `torch.compile` cannot trace stably |
| `conf.conf` / `conf.alpha` | `True` / `0.1` |
| `exp.neptune` / `exp.ablation_id` | `False` / `1` |
| `exp.seed` / `exp.seed_list` | `<seed>` / `[<seed>]` |

No memory profile is registered for the cluster grid, so `batch_size` stays at the
job config's 512 with `accum_steps` 1 (`grid.py:52-56`).

---

## 3. Clustering hyperparameters proper

### Configurable — `class Cluster`, `config.py:778-819`

| Field | Value | Note |
| --- | --- | --- |
| `cluster.enabled` | `True` | swaps the whole network for CCM |
| `cluster.k_clusters` | 1…9 | swept; the value in the log path (`k1`…`k9`) |
| `cluster.beta` | `0.3` | weight on the auxiliary similarity loss; upstream's default; **0.3 in all 132 published runs.** A second group of 54 runs under `logs/cluster/` uses `beta = 0.05` and is excluded — see §8. Within the published set beta was never swept |

Objective (`models.py:450-480`, `exp.py:495`/`:510`):

```
backward loss = forecast_MSE + beta * similarity_loss
```

> **Deliberate divergence from upstream.** The similarity term is trained on but
> kept **out** of `sl_{train,val,test}/loss` and out of the best-model /
> early-stopping metric, which stay the pure forecast MSE. It is logged
> separately as `sl_train/similarity_loss`. Upstream's `Exp_CCM.train`
> early-stops on the combined value; at 27 channels the similarity term runs
> 15–250 against an MSE of ~0.5, so combining them hands model selection to the
> regularizer.

### Hardcoded upstream — identical for every run, no config field exists

| Hyperparameter | Value | Where |
| --- | --- | --- |
| sinkhorn epsilon | 0.05 | `layers.py:263`, `:309` |
| sinkhorn iterations | 1 | the `sinkhorn_iterations=3` argument is dead — the function does a single `exp(x/ε)` + row-normalisation pass |
| concrete-Bernoulli temperature | 0.07 | `layers.py:301` (assignment mask), `ccm.py:302` (similarity loss) |
| cluster-embedding init | `kaiming_uniform_(a=√5)` | `layers.py:273` |
| channel → cluster scoring | cosine similarity (L2-normalised, `dim=1`) | `layers.py:287` |
| channel projection | single `nn.Linear(seq_len, d_model)` (the MLP variant is commented out) | `layers.py:271` |
| embedding refresh | 1-head cross-attention, training batches only (`if_update=self.training`) | `layers.py:276`/`:297` |
| similarity kernel | `exp(-5 · d² / max(d²))`, squared Euclidean over channel windows, batch-averaged | `ccm.py:238` |
| entropy term | mean over channels of `-Σ p log p`, added to the similarity loss with weight 1 | `ccm.py:274` |
| cluster head | k parallel `nn.Linear`, mixed by `prob` | `patch_layer.py:244` |
| upstream mode | `individual = "c"` (clustered) always | `ccm.py:88` |

### Provenance of the two scalar knobs: `beta` and the temperature

Both are upstream's values, but only one of them is reachable from our config.

| | `beta` (similarity-loss weight) | `temp` (concrete-Bernoulli temperature) |
| --- | --- | --- |
| Value used | **0.3** | **0.07** |
| Upstream origin | `TimeSeriesCCM-main/main.py:27` — `--beta` argparse default `0.3` (their stock entry point uses `0.1`, and the M4/stock scripts force `0.0` on some branches) | a *default argument* of `concrete_bern(prob, temp=0.07)`, repeated in five upstream places (`models/layers.py:255`, `:301`, `models/patch_layer.py:371`, `exp/exp_ccm.py:300`, `:315`) — **no call site ever passes it** |
| In this repo | `cluster.beta`, a real config field (`config.py:811`); applied at `utils/models.py:480` as `beta * ccm.similarity_loss(...)` | ported verbatim as the default of `CCMForecaster._concrete_bern` (`models/ccm.py:302`); also never passed |
| Configurable? | **Yes** — settable per run (`--ablation '{"cluster.beta": 0.1}'`) | **No** — there is no config field; changing it means editing source |
| Actually varied? | No — 0.3 in all 105 runs on disk | No — 0.07 everywhere |

The temperature acts in **two separate places**, both at 0.07:

1. **Inside the assigner** (`layers.py:293`) — relaxes `prob_avg` into the mask fed to the cross-attention that refreshes the cluster embeddings each training batch.
2. **Inside the similarity loss** (`ccm.py:302`, ported from `exp_ccm.py:300`) — relaxes `prob` into the `membership` matrix the spectral term is computed on.

To sweep it you would have to change both (or decide to decouple them deliberately),
and the assigner one lives in vendored upstream code — editing it means the
`TimeSeriesCCM-main/` tree is no longer the released code, which is the property
the whole adapter design exists to preserve. A cleaner route would be to add a
`cluster.temp` field and thread it through `build_ccm_args` → `Cluster_assigner`,
which would require a small upstream patch either way.

`beta = 0` is meaningful and supported: it trains the clustered architecture on the
forecast loss alone, leaving the assignment unconstrained by channel similarity
(`compute_aux_loss` returns `None` and skips the term entirely). That ablation has
not been run.

The assignment is **soft**: `prob` is a `[num_channels, k]` distribution and every
channel passes through all k heads weighted by it. There is no hard partition, no
per-cluster training pass, no leftover group.

`k = 1` is the controlled no-clustering baseline: sinkhorn returns an all-ones
assignment, `Cluster_wise_linear` collapses to one shared channel-independent
head, and the similarity loss becomes an additive constant with no gradient.

---

## 4. Per-backbone hyperparameters, as actually consumed by CCM

Upstream reads one `args` object across all four wrappers, so unused keys still
have to exist. Below is what each model *really* uses.

### Shared by all three models that were run

| Key | Value |
| --- | --- |
| `in_len` / `out_len` | 336 / 96 |
| `data_dim` | 27 or 29 — inferred from the CSV at load time (`dataloading.py:1201`), see gotcha (c) |
| `batch_size` | 512 |
| `individual` | `"c"` |
| `n_cluster` | k |
| `dropout` | 0.3 |
| `cuda` | device ordinal only |

### DLinear — `DLinearC` (`Dlinear.py:40`)

| Hyperparameter | Value | Source |
| --- | --- | --- |
| cluster-assigner embedding dim | **256** | `args.sl.d_ff` — a **pydantic default**; the dlinear job YAML sets no `d_ff` |
| decomposition kernel | **25** | hardcoded upstream; the job config's `dlinear.moving_avg: 3` is **ignored** |
| head | `Cluster_wise_linear` ×2 (seasonal + trend), 336 → 96 | upstream |
| RevIN | **not applied** | see gotcha (b) |
| lr / weight_decay | 1e-5 / 1e-5 | job YAML |

### PatchTST — `PatchTSTC` (`patchtst.py:9` + `patch_layer.py:14`)

| Hyperparameter | Value | Source |
| --- | --- | --- |
| cluster-assigner embedding dim | 16 | `args.sl.d_model` |
| `d_model` / `d_ff` | 16 / 32 | job YAML |
| `n_heads` / `n_layers` | 2 / 3 | job YAML |
| `attn_dropout` / `dropout` | 0.0 / 0.3 | job YAML |
| `patch_len` / `stride` | 16 / 8 | `data.patch_dim` / `patch_stride` → patch_num = 42 at L=336 |
| `padding_patch` | `"end"` | repo-side literal in `build_ccm_args` |
| positional encoding | `pe="zeros"`, `learn_pe=True` | hardcoded upstream (ignores `sl.pos_enc`) |
| encoder norm | `BatchNorm`, `d_k=d_v=None` | hardcoded upstream (`TV_Encoder` defaults) |
| `max_seq_len` / `pre_norm` / `pretrain_head` | 1024 / `False` / `False` | repo-side literals |
| head | `Flatten_Head`, `individual="c"`, `head_dropout=0.0` | hardcoded upstream |
| RevIN | **yes** — `RevIN(affine=True, subtract_last=False)` | hardcoded in upstream's `Patch_backbone` |
| lr / weight_decay | 1e-4 / 1e-4 | job YAML |

### TSMixer — `TSMixerC` (`tsmixer.py:59`)

| Hyperparameter | Value | Source |
| --- | --- | --- |
| cluster-assigner embedding dim | **256** | `args.sl.d_ff` — a **pydantic default** |
| mixer hidden dim (`d_ff`) | 256 | the job YAML's `sl.d_model: 32` is stored by the wrapper but never used |
| `n_mixer` layers | 1 | `sl.num_enc_layers` |
| `dropout` | 0.3 | job YAML |
| normalisation inside a mixer layer | global `(x − mean)/std` over the whole tensor, *not* a `nn.BatchNorm` | hardcoded upstream (`tsmixer.py:140`) |
| head | `TemporalProj`, `individual="c"`, 336 → 96 | upstream |
| RevIN | **not applied** — `TSMixerC` builds a `RevIN` but both calls are commented out (`tsmixer.py:86`, `:91`) | upstream |
| lr / weight_decay | 1e-4 / 1e-4 | job YAML |

`TimesNet` is in `CLUSTER_MODELS` and has a job config, but **no TimesNet cluster
run exists on disk.**

---

## 5. Training / data / conformal settings (identical across all 132 published runs)

### Optimisation

| Setting | Value |
| --- | --- |
| criterion | MSE |
| optimizer | Adam (bias and norm params excluded from weight decay) |
| scheduler | `OneCycleLR`, `max_lr = sl.lr`, `pct_start = 0.3`, `epochs = 100`, `steps_per_epoch = len(train_loader)` (`scheduler.lradj: type3` is inert on the onecycle path) |
| epochs | 100 (max) |
| early stopping | on, patience 20, delta 0.0, monitored on val loss (`exp.best_model_metric = "loss"`) |
| batch size | 512, `accum_steps` 1 |
| EMA | not used on the supervised path |
| compile | `False` (forced) |
| dtype / precision | float32, full precision (no AMP) |

### Data

| Setting | Value |
| --- | --- |
| window | `seq_len` 336 → `pred_len` 96, `window_stride` 1 |
| splits | train 0.7 / val 0.1 / test 0.2, chronological slices |
| scaling | `data.scale = True` (standardised on train statistics) |
| channels | ED2022 27, UH2022 27, `EMF_2023_..._PTV` 29 — all bands forecast jointly (`univariate: False`) |
| loader | `num_workers` 4, `pin_memory` True |

### Conformal prediction

| Setting | Value |
| --- | --- |
| `conf.conf` | `True` |
| `conf.alpha` | 0.1 (nominal 90% coverage) |
| `conf.corrected` | `False` → reported `ic`/`jc` use the **uncorrected** critical scores, so joint coverage is not expected to hit 90%; matches the rest of the grids |
| `conf.intervals` | `True` |
| `conf.validation_eval` | `False` → calibrated once, on the validation split, using the selected (best-val-loss) model, then applied to test |
| scores | per horizon-step and per channel (`exp.py:1133+`) |

### Seeds

Canonical `69421`; ED2022 DLinear and TSMixer additionally at `69422` and `69423`.
The lookback ladder is `69421` only. Seeds `42` and `43` are **not** part of this
set and are not seeds of it — see §8.

---

## 6. Runs on disk

**132 runs** that enter the published numbers: the 105-run k-sweep at `L336_O96`
below, plus the 27-run lookback ladder restored on 2026-08-14 (see the end of this
section). A further 54 runs sit under `logs/cluster/` and are **excluded** — see
§8.

The k-sweep: 105 runs, all at `L336_O96`. Trained 2026-08-06 (59 runs, all nine
model × dataset pairs) and 2026-08-10 (46 runs, ED2022 only: the `69422`/`69423`
seeds plus k = 7, 8).

| Model | Dataset | k | Seeds | Runs |
| --- | --- | --- | --- | --- |
| DLinear | ED2022 | 1…9 | 69421, 69422, 69423 | 27 |
| DLinear | UH2022 | 1,2,3,4,5,6,9 | 69421 | 7 |
| DLinear | emf2023 | 1,2,3,4,5,6,9 | 69421 | 7 |
| PatchTST | ED2022 | 1…9 | 69421 | 9 |
| PatchTST | UH2022 | 1,2,3,4,5,6,9 | 69421 | 7 |
| PatchTST | emf2023 | 1,2,3,4,5,6,9 | 69421 | 7 |
| TSMixer | ED2022 | 1…9 | 69421, 69422, 69423 | 27 |
| TSMixer | UH2022 | 1,2,3,4,5,6,9 | 69421 | 7 |
| TSMixer | emf2023 | 1,2,3,4,5,6,9 | 69421 | 7 |

Note the sweep on disk (k = 1…9, one window) is **not** the `k_values = [2, 3, 5, 9]`
× 4 windows declared in `GRIDS["cluster"]`; it was run with explicit `--k` values on
the 336x96 window. k = 7 and k = 8 exist for ED2022 only.

### The lookback ladder

A second cluster sweep: the 2026-08-10 L-ladder — DLinear/PatchTST/TSMixer,
ED2022, k = 5, seed 69421, L = 48, 96, 144, 192, 240, 288, 384, 432, 480 with
O = 96, 27 runs. With the three `L336` k = 5 cells from the sweep above it makes a
complete ten-point ladder on the same axis as the multivariate and per-channel
ladders, which is what `plot_L_ladder.py` overlays.

Commit `9ff78806` ("removed all the other L and O in clsuetering") deleted these
from the working tree. On **2026-08-14** the 54 `log.json` / `args.yaml` files were
restored from `9ff78806~1` — text only, 0.7 MB; the 60 MB of PNGs and 42 MB of
checkpoints that shipped alongside them were left in history, since nothing
downstream reads them. Their hyperparameters are identical to the tables above
except for `data.seq_len`.

> Earlier revisions of this note said `git checkout -- logs/cluster` restores
> them. It does not: the deletion was **committed**, so that command is a no-op
> here. The working form names the commit —
> `git checkout 9ff78806~1 -- logs/cluster`. `logs/grid_report_cluster.json`, also
> cited by earlier revisions, was deleted too and exists only in history.

---

## 7. Gotchas when quoting these numbers

**(a) Several fields in a cluster run's `args.yaml` are inert.** They are config
defaults, dumped for every run, but CCM never reads them: `sl.num_heads` (16) and
`sl.attn_dropout`/`ff_dropout`/`pred_dropout` for DLinear and TSMixer;
`sl.d_model` (32) for TSMixer; `dlinear.moving_avg` (3) for DLinear;
`sl.norm_mode`, `sl.head_type`, `sl.channel_independent`, `sl.revin*`/`revout` for
all of them; and every block for a model that is not the one being run
(`patchtst:`, `tsmixer:`, `timesnet:`, `arima:`, …). Read §4, not the raw YAML.

**(b) RevIN is applied for PatchTST only**, and it comes from upstream's own
`Patch_backbone`, not from `sl.revin`. DLinear and TSMixer cluster runs have no
reversible instance normalisation at all, whereas their non-cluster multivariate
counterparts do.

**(c) `data.num_channels` is recorded as 27 in every `args.yaml`** because the YAML
is dumped from the job config; the real value is inferred from the CSV at load time
and is 29 for `EMF_2023_university_5_with_PTV` (confirmed by the length of
`sl_test/mse_per_channel` in those runs' `log.json`).

**(d) The plain (non-cluster) DLinear column is not a clean ablation** for CCM
DLinear: upstream's `DLinearC` hardcodes a decomposition kernel of 25 where the
repo's DLinear uses `moving_avg` 3, and it uses default `nn.Linear` init where the
repo applies the DLinear paper's `1/seq_len` constant init. Use `k = 1` as the
no-clustering baseline instead.

**(e) This grid replaced a static KMeans-partition head on 2026-08-06.** Any k1/k9
cells from commits `2ebc7f08` / `3d2de887` (deleted in `f24cf7a2`) came from that
older head and are not resumable against the current config — their
`cluster_method` / `cluster_seed` / `cluster_assignment` fields no longer exist.

---

## 8. The excluded `beta = 0.05` group — read this before quoting any cluster number

`logs/cluster/` holds **two different configurations**, distinguishable only by the
seed number in the run path. 132 runs are published; 54 are excluded by
`results/specs/runs.yaml`.

| parameter | published (`69421`/`69422`/`69423`) | excluded (`42`/`43`) |
| --- | --- | --- |
| `cluster.beta` | **0.3** | **0.05** |
| `sl.batch_size` | 512 | 32 |
| `sl.lr` | 1e-5 / 1e-4 | 1e-3 |
| `sl.d_ff` | 256 / 32 | 128 |
| `sl.epochs` | 100 | 30 |
| `early_stopping.patience` | 20 | 5 |

Six parameters differ across the whole tree, `beta` and `d_ff` among them — `d_ff`
sets the cluster-assigner embedding dim for DLinear and TSMixer (§4), so the two
groups are not even the same width: **237,248 vs 280,384 parameters** for DLinear
at k = 3. Averaging them is a mean over two models, not a mean over seeds.

Doing so inflated the seed stdev roughly threefold (DLinear 0.0239 vs 0.0081) and
moved every best-k: DLinear 5 → 3, PatchTST 9 → 8, TSMixer 7 → 1.

**The structural defect.** A cluster run's path is
`<model>/<dataset>/L<L>_O<O>_cp/k<k>/<seed>/` — there is no component for the
configuration. Two experiments therefore interleave in one tree with nothing but
`42` vs `69421` to tell them apart, and every tool that globbed the tree silently
averaged across them. If the `beta = 0.05` arm is ever revived, give it its own
root rather than another seed number.

To publish that arm instead, exclude the canonical seeds in `specs/runs.yaml`
rather than deleting the rule. The one thing that is never right is averaging the
two together.
