# EMForecaster (Reproduction Repository)

This repository is a personal reproduction and extension of the [EMForecaster paper](https://ieeexplore.ieee.org/document/11095417) by Xavier Mootoo, Hina Tabassum, and Luca Chiaraviglio (IEEE Transactions on Network Science and Engineering, 2025).
It is maintained by Daniel Chahine and hosted at https://github.com/DanielChahine0/emforecaster-repo.
It is **not** the official paper repository.
The code here is a working setup used to reproduce the paper's electromagnetic-field (EMF) exposure forecasting results and to experiment with a multivariate extension.
The original paper PDF is included at the repository root as [`EMForecaster Paper.pdf`](./EMForecaster%20Paper.pdf).

## Description

The repository contains a single deep-learning forecasting package with two parallel setups plus a standalone measurement-analysis tool:

- **`forecasting.univariate`** - the univariate forecasting setup, which reproduces the paper's single-series EMF exposure forecasting.
- **`forecasting.multivariate`** - a multivariate extension of the same setup (channel-independent forecasting across many frequency bands), a near-identical twin of `forecasting.univariate` with the genuine differences confined to dataloading, the experiment loop, model heads, and the config schema.
- **`multivariate/`** - a self-contained matplotlib/pandas CLI for exploratory analysis of the processed EMF measurement CSVs in `data/multivariate/processed/` (EMF-level time trends and correlation heatmaps), unrelated to model training.

Both training setups are driven by the root `main.py`, which selects the setup automatically from the job name.
Both share the same conformal-prediction uncertainty quantification (independent/joint coverage and the paper's Trade-off Score), the same hyperparameter-tuning entrypoint, and the same optional Neptune logging and HPC submission paths.

Model families available through `get_model`: PatchTST, recurrent models (LSTM/RNN/GRU/Mamba via `backbone_id`, reachable as `RecurrentModel` or under the grid's `LSTM` alias), Linear, DLinear, EMForecaster, ModernTCN, TimesNet, KNN, TSMixer, ARIMA and Persistence (both classical and non-gradient; see `exp.no_train`), and four RF baselines (RF_EMF_MLP, RF_EMF_CNN, RF_EMF_LSTM, RF_EMF_Transformer).

## Repository layout

```
.
├── main.py                 # Root entrypoint; picks the setup by job name
├── forecasting/            # Forecasting package
│   ├── univariate/         # Univariate forecasting setup
│   └── multivariate/       # Multivariate forecasting setup
├── multivariate/           # Standalone EMF measurement-analysis CLI (+ its own README)
├── data/                   # Measurement data (see Data)
│   ├── ML_DATASETS/        # Raw VoIP traces + build_voip.py
├── logs/                   # Offline training-run logs (per run, per seed)
├── results/                # Tables, figures and the tidy CSVs behind them (see results/README.md)
├── pyproject.toml          # Poetry metadata, MIT license, Python >=3.11,<3.14
├── requirements.txt        # Fully pinned dependencies (Python 3.12 / macOS arm64)
└── EMForecaster Paper.pdf  # Original paper (reference)
```

Each training setup mirrors the same subtree: `config`, `exp`, `models`, `layers`, `conformal`, `tuning`, `jobs`, `utils`, `analysis`.
Note that "analysis" exists in two unrelated places: `forecasting/<univariate|multivariate>/analysis/` analyzes training runs (loss curves, bar graphs, ADF/ARIMA, Neptune fetching), while `multivariate/` analyzes measurement CSVs.

## Installation

Requires Python `>=3.11,<3.14`.
`requirements.txt` is fully pinned (all versions, including transitives) and targets Python 3.12 on macOS arm64.

Using a virtual environment:

```bash
python3.12 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

Using conda:

```bash
conda create -n emforecaster python=3.12
conda activate emforecaster
pip install -r requirements.txt
pip install -e .
```

The final `pip install -e .` installs the project (distribution name `emforecaster`) in editable mode, exposing the importable package `forecasting`.
Note that `import emforecaster` will not work; the importable package is `forecasting` (with `forecasting.univariate` and `forecasting.multivariate`).

## Data

The EMF measurement data is proprietary and is not distributed with the repository.
To obtain the data, contact the original paper's authors (Luca Chiaraviglio for the Italian measurement datasets).

Univariate series live in `data/univariate/` (single-column exposure series in V/m):

| File | Dataset id |
|---|---|
| `DET_Monitoring.csv` | `rf_emf_det` |
| `PTV_Monitoring.csv` | `rf_emf_ptv` |
| `PTV_Monitoring2.csv` | `rf_emf_ptv2` |

These three ids are the effectively usable univariate datasets.
The dataloader references additional dataset ids (for example `rf_emf`, `rf_emf_tur`) whose files are not present on disk, so those will fail; do not trust the config docstring as the runnable set.

Multivariate data lives in `data/multivariate/`:

- `data/multivariate/processed/` (read by both the training setup and the analysis CLI): `ED2022.csv`, `ED2024.csv`, `UH2022.csv`, `UH2024.csv`, `EMF_2023_university_5_with_PTV.csv`.
- `data/multivariate/unprocessed/` - the pre-normalization raw sources, kept as an archive and not read by any tool.

Each processed multivariate CSV has a `Timestamp` column, per-frequency-band columns in V/m, and `EMF_*` aggregate columns.
The multivariate dataloader drops `Timestamp`, the `EMF_*` aggregates, and `PTV` (to avoid target leakage), interpolates missing bands, and then sets `data.num_channels` automatically from the data (27 or 29), so the value in `args.yaml` is advisory only.

### VoIP call-quality data

A second, unrelated corpus lives in `data/ML_DATASETS/`: 16 VoIP call traces, one per (access type, codec) over `{Mobile, Fixed}` x `{g711, g722, g729, gsm, mpeg16, opus, spx8000, spx16000}`, each a `Time,MOS,BW,RTT,JIT,DJB,SNR` table of 161-820 rows.

`python data/ML_DATASETS/build_voip.py` pools them into `data/multivariate/processed/VOIP_MOBILE.csv` and `VOIP_FIXED.csv` (eight traces each, six channels, a `Trace` column naming the codec). Its only cleaning is to null and interpolate the three physically impossible `JIT` values in `mob_g729` (up to 9.9e11); RTT outliers and the leading `RTT = 0` rows are kept deliberately.

The loader splits **each trace on its own** and then concatenates the parts, so every split carries the same codec mix, and flags one row per join so no window can span two calls (`pooled_trace_splits` in `utils/dataloading.py`). The traces are far shorter than the EMF files, so these datasets run at their own windows (`L48_O12`, `L24_O6`) and their own split (0.6/0.2/0.2) - see the `voip*` grids below.

## Usage: running experiments

Run from the repository root. `main.py` has two modes:

```bash
# Single-job mode: run one job by name
python main.py <job_name> [--quick] [--ablation '<json>'] [--ablation_id <int>]

# Grid mode: run a full model x dataset x window sweep
python main.py --grid [--setup <setup>] [--window <spec>] [--seeds <spec>] [--k <spec>] [--dry_run] [--jobs <int>]
```

The setup is chosen automatically from the job name: any `job_name` starting with `rf_emf_mv` dispatches to `forecasting.multivariate`, everything else to `forecasting.univariate`.
`main.py` then dynamically imports `{package}.config.config.load_config` and `{package}.exp.exp.Experiment`, and reads the config from `forecasting/<univariate|multivariate>/jobs/exp/<job_name>/args.yaml`.

Runnable job names (each is a folder under `forecasting/<setup>/jobs/exp/` holding an `args.yaml`):

Univariate (`forecasting.univariate`):

- `rf_emf/emforecaster`
- `rf_emf/dlinear`
- `rf_emf/patchtst`
- `rf_emf/tsmixer`
- `rf_emf/baselines/cnn`
- `rf_emf/baselines/lstm`
- `rf_emf/baselines/mlp`
- `rf_emf/baselines/transformer`
- `test`

Multivariate (`forecasting.multivariate`, `rf_emf_mv` prefix):

- `rf_emf_mv/arima`
- `rf_emf_mv/dlinear`
- `rf_emf_mv/emforecaster`
- `rf_emf_mv/lstm`
- `rf_emf_mv/patchtst`
- `rf_emf_mv/timesnet`
- `rf_emf_mv/transformer`
- `rf_emf_mv/tsmixer`

The `forecasting.multivariate` setup also mirrors the univariate `rf_emf/*` and `test` jobs, but because dispatch keys off the `rf_emf_mv` prefix, those still run through the `forecasting.univariate` setup.

Single-job flags:

- `--quick` runs a single seed (`seed_list[:1]`).
- `--ablation '<json>'` takes a JSON string of dotted overrides (for example `{"sl.lr": 0.001}`) mapped onto the config; invalid JSON exits with an error.
- `--ablation_id <int>` tags the run (default 1) and is used in output paths.

### Grid mode

Passing `--grid` (or omitting the job name entirely) runs a full sweep over every model x dataset x window cell of a grid defined in `forecasting/grids.py` instead of a single job.
The exit code is non-zero if any cell failed.

- `--setup {univariate,multivariate,univariate_series,cluster,voip,voip_perchannel,voip_cluster}` selects which grid to run:
  The first three sweep the same seven models — EMForecaster, DLinear, PatchTST, TSMixer, Transformer, LSTM, ARIMA — over 3 datasets x 4 windows, so every grid is 84 cells and a model's column exists in every table.

  - `multivariate` - one model per cell, forecasting all 27/29 frequency bands jointly, over the `ED2022`, `UH2022`, and `EMF_2023_university_5_with_PTV` datasets. (PatchTST left this sweep on 2026-07-24 and rejoined it on 2026-08-03. Its pre-drop cells are not on disk: commit `e84faec4` deleted `logs/multivariate/PatchTST/` along with the other three windows of every model, leaving only the `L336_O96` slice. `git checkout e84faec4^ -- logs/multivariate` restores the old tree.)
  - `univariate` - the per-band counterpart of the multivariate grid: same models, datasets, and windows, but each cell trains one independent model per frequency band and averages the bands back into one number per cell, so its cells are directly comparable to the multivariate ones. Driven by `forecasting/perchannel.py`, which fans the band trainings out to a process pool. It shares the multivariate grid's model list outright (`UNIVARIATE_MODELS is MV_MODELS`) and reuses the same `rf_emf_mv/*` job configs one band at a time.
  - `univariate_series` - the original single-series sweep over the DET/PTV monitoring datasets (`rf_emf_det`, `rf_emf_ptv`, `rf_emf_ptv2`). Same seven models, but its own job configs under `forecasting/univariate/jobs/exp/rf_emf/`, because it is the only grid that runs the `forecasting.univariate` package (see `grid.SETUP_PACKAGE`). Its LSTM is `rf_emf/lstm` (model_id `LSTM` -> `RecurrentModel(backbone_id="LSTM")`, the same model the other two grids run), *not* the frozen `rf_emf/baselines/lstm` (`RF_EMF_LSTM`) that produced the published "LSTM (1997)" column — do not merge those two columns.

  - `cluster` - the CCM channel-clustering sweep: same datasets and windows as `multivariate`, with CCM's *learned, soft* channel clustering ([Channel Clustering Module, NeurIPS 2024](https://arxiv.org/abs/2404.01340)) threaded through **this repo's own model implementations**. The clustering modules — `Cluster_assigner` and `Cluster_wise_linear` — are the paper's own code, imported verbatim from the vendored `TimeSeriesCCM-main/` tree (`forecasting/multivariate/utils/ccm_upstream.py`) and orchestrated by `forecasting/multivariate/models/ccm.py`. Each channel's window is projected to `d_model`, scored by cosine similarity against `k` trainable cluster embeddings, and sinkhorn-normalized into a `[num_channels, k]` distribution; the model's per-channel temporal linears are swapped for `k` parallel cluster-wise ones mixed by that assignment. The cluster embeddings are refreshed from each training batch by cross-attention, and the objective is `forecast_loss + beta * similarity_loss` (`cluster.beta`, default `0.3` from upstream), where the similarity term pulls the assignment toward the data's own channel-similarity structure. It adds a `k` dimension to the grid (`k_values = [2, 3, 5, 9]`, so 3 models x 3 datasets x 4 windows x 4 k = 144 cells) and writes to `logs/cluster/<model>/<dataset>/L<L>_O<O>_cp/k<k>/<seed>/`.

    The three models are `DLinear`, `PatchTST` and `TSMixer` — exactly the ones whose implementations have a per-channel temporal linear for CCM to route (DLinear's seasonal/trend linears, PatchTST's flatten head, TSMixer's temporal MLPs and head — the same places upstream's own wrappers put `Cluster_wise_linear`). All three are in `MV_MODELS`, so every cluster cell has same-window multivariate and per-channel baselines beside it: the cluster cell is the multivariate cell's exact network — same config, same init, same RevIN — differing only in the clustering module, which is what makes a cluster-vs-multivariate (or vs per-channel) comparison attributable to the *paradigm* rather than to implementation drift. `EMForecaster`, `Transformer`, `LSTM` and `ARIMA` lack that per-channel structure and are rejected at model build instead of being silently adapted.

    **Divergences from the paper's released code** — only the clustering *layers* run from `TimeSeriesCCM-main/`; its model wrappers and Exp pipeline do not. (1) The networks are ours, as above; until 2026-08-25 the grid ran upstream's own `DLinearC`/`PatchTSTC`/`TSMixerC`/`TimesNetC`, whose implementations differ from the multivariate cells' (e.g. `DLinearC` hardcodes a decomposition kernel of 25 where the job config sets `moving_avg: 3`, and skips the repo's `1/seq_len` constant init) — runs from that era are not comparable to current ones and must not share a logs tree with them: the L-ladder rungs sit under `logs/_archive/`, and any upstream-era cell still under `logs/cluster/` needs the same retirement before its rerun lands. `TimesNet` left the grid with the wrappers (our TimesNet embeds all channels jointly from the first layer, so there is nothing per-channel to route, and it never had an MV baseline). (2) PatchTST: upstream additionally injects the cluster embeddings into its attention layers; ours keeps the encoder byte-identical to the multivariate cell and clusters only the head. (3) The similarity term is trained on, but kept out of the reported `sl_{train,val,test}/loss` and out of the best-model / early-stopping metric, which stay the pure forecast loss; it is logged separately as `sl_train/similarity_loss`. Upstream (`Exp_CCM.train`) early-stops on the combined value, but at 27 channels the similarity term runs 15–250 against an MSE around 0.5, so combining them hands model selection to the regularizer — on an ED2022 `L336_O96` sweep it early-stopped `k=2` at epoch 27 having selected epoch 7, while the plain DLinear baseline kept improving for all 100. Keeping them separate also means a cluster cell's loss is comparable to every other cell in every grid. See `compute_aux_loss` in `forecasting/multivariate/utils/models.py`. (4) Upstream's cluster-embedding refresh returns a per-example `[batch, k, d_model]` tensor that the released wrappers write back into the `[k, d_model]` slot — fatal on training batch two as released; `CCMForecaster` pools it over the batch (the assigner's own batch-mean convention) and holds it in a registered buffer so it checkpoints and moves with the module. (5) Upstream's batch-wise similarity matrix only has the right shape in its M4/stock setups; the same kernel is applied along the channel axis instead (`CCMForecaster.channel_similarity`).

    **The `--k 1` cell is the controlled no-clustering baseline**: at `k=1` sinkhorn returns an all-ones assignment, so every cluster-wise linear collapses to exactly one shared channel-independent map, and the similarity loss becomes an additive constant with no gradient. Every difference between `k=1` and `k>1` is then attributable to the clustering itself — and because the backbone is now the multivariate implementation, the plain multivariate column is a fair comparison too (the residual difference at `k=1` is the assigner's parameters, since all three models' job configs already use shared per-channel maps — PatchTST runs `channel_independent: True`).

    Finally, this grid replaced a KMeans-partition head on 2026-08-06; the k1/k9 cells in commits `2ebc7f08` / `3d2de887` (deleted in `f24cf7a2`) came from that head and are not resumable against the current config.

  - `voip`, `voip_perchannel`, `voip_cluster` - the multivariate, per-channel and CCM-clustering paradigms over the pooled VoIP datasets (`VOIP_MOBILE`, `VOIP_FIXED`; 6 channels; windows `L48_O12` and `L24_O6`; `k_values = [2, 3]`, with `k = 1` and `k = 6` reachable through `--k`). They run the same drivers, job configs and log trees as their EMF counterparts - a VoIP cell lands in `logs/multivariate/<model>/voip_mobile/...` beside the EMF ones, one dataset slug apart - and add a **Persistence** column (`rf_emf_mv/persistence`, `exp.no_train`), which the EMF grids deliberately do not carry: four of the six VoIP channels repeat their previous value in 80-99% of steps, so without a last-value baseline a low error cannot be told apart from having learned nothing.

    Every cell of these three grids also takes `grids.VOIP_OVERRIDES` on top of its job config (`sl.batch_size` 64, `data.train_split` 0.6, `data.val_split` 0.2). The split is the wider one because a 10% validation split of a 400-row trace leaves too few windows to calibrate a conformal interval on, and the batch is smaller because a pooled VoIP train split is ~1,900 windows, where a 512 batch would be four optimizer steps per epoch.

  If `--setup` is omitted in grid mode, you are prompted interactively to pick one of the four EMF grids (`u`/`m`/`s`/`c`); the VoIP grids must be named explicitly.
- `--window <spec>` restricts the sweep to specific lookback/horizon window(s) instead of all four. Every grid uses the same four `(seq_len, pred_len)` windows: `192x96`, `192x336`, `336x96`, `336x336`. Each window is written as `LxO` (e.g. `336x96`), `L_O` (`336_96`), or the `logs/` directory form `L336_O96`; comma-separate to select several (e.g. `--window 336x96,336x336`). The spec is validated against the grid up front, so a typo or a window not in the grid is a clean error rather than a mid-run failure. Omit to run every window.
- `--seeds <spec>` trains every cell on more than one seed instead of just the canonical `69421` (default: 1 seed). A count counts up from the canonical seed, so `--seeds 3` runs `69421`, `69422`, `69423`; a comma-separated list (`--seeds 69422,69423`) selects those seeds exactly, for topping up a grid that already has `69421`. Because the canonical seed comes first and resume is per `(cell, seed)`, adding seeds to a finished sweep re-runs only the new ones — the existing runs are skipped, not retrained. Seeds are run seed-major (the whole grid at one seed, then the next), so an interrupted multi-seed sweep leaves a complete grid at one seed rather than a handful of cells with error bars. Note that on the per-channel grid a seed multiplies *band* runs: a second seed on the full `--setup univariate` grid is another ~1,700 trainings.
- `--k <spec>` applies to the `cluster` grid only: restrict the sweep to specific cluster count(s) instead of every `k` in the grid. A single value (`--k 5`) or a comma-separated list (`--k 1,2`); duplicates collapse and the values run in ascending order. Unlike `--window`, a requested `k` need *not* be one of the grid's own `k_values`: the degenerate endpoints `k=1` (every channel in one cluster) and `k=num_channels` (one cluster per channel) are deliberately outside the sweep so they can be run as one-off sanity checks, and `--k 1` reaches them without editing `forecasting/grids.py`. An off-grid value is logged as `[NOTE ] ... running as a one-off` so it is never mistaken for part of the sweep, and the `k_values` actually covered are recorded in `logs/grid_report_cluster.json`. Passing `--k` to a grid without a `k` dimension is a clean error.
- `--dry_run` lists the cells and their target log directories without training anything. Useful to preview what a sweep will do.
- `--jobs <int>` applies to the per-channel (`--setup univariate`) grid only: how many band trainings run concurrently (default 8). Lower it if the GPU runs out of memory.

Examples:

```bash
# Full multivariate sweep, all four windows
python main.py --grid --setup multivariate

# Per-band grid, one window only, 4 concurrent band trainings
python main.py --grid --setup univariate --window 336x96 --jobs 4

# Preview the single-series sweep for two windows without training
python main.py --grid --setup univariate_series --window 192x96,336x96 --dry_run

# Single-series sweep on one window, three seeds (69421-69423) for error bars
python main.py --grid --setup univariate_series --window 336x96 --seeds 3

# Cluster sweep, one window, every k in the grid (48 cells)
python main.py --grid --setup cluster --window 336x96

# Cluster at k=1 only (one cluster for every channel), not part of the sweep
python main.py --grid --setup cluster --window 336x96 --k 1

# Trial run: DLinear+CCM on one dataset/window, sweeping k (4 cells).
# --k 1 is the controlled no-clustering baseline; see the `cluster` notes above
# for why the plain DLinear column is not one.
python main.py --grid --setup cluster \
    --model DLinear --dataset ed2022 --window 336x96 --k 1,2,4,6

# The plain DLinear cell it sits beside, if not already on disk
python main.py --grid --setup multivariate \
    --model DLinear --dataset ed2022 --window 336x96

# Build the comparison table (adds a k column on the cluster tree)
python results/make.py extract && python results/make.py tables
```

### Seeds and outputs

In single-job mode `main.py` iterates over every seed in `args.exp.seed_list`, building a fresh `Experiment` per seed.
Grid mode ignores the config's `seed_list` and pins the seeds itself — the canonical `69421` unless `--seeds` asks for more — so a cell's log path is stable and the resume check can find it.
With offline logging (`exp.neptune=False`), outputs land under `logs/<ablation_id>_<model_id>_<exp.id>/<seed>/`, containing the resolved `args.yaml`, `log.json` metrics, best weights `supervised.pth`, conformal calibration `scores` (when enabled), and 10 randomly-sampled forecast plots named `plot_<index>.png` (for example `plot_5320.png`), where `<index>` is a random position in the test set, not a sequential `plot_0.png … plot_9.png`.

## Usage: hyperparameter tuning

```bash
python forecasting/univariate/tuning/tune.py <job_name> [--cc] [--acc <account>]
```

Use `forecasting/multivariate/tuning/tune.py` for the multivariate jobs.
The tuner reads `forecasting/<setup>/jobs/exp/<job_name>/ablation.yaml`, takes the `itertools` product of the ablation grid, and either runs each configuration locally or, with `--cc`, submits SLURM / Compute Canada jobs via `jobs/hpc/submit.py`.

Each setup ships its own tuner: `forecasting/univariate/tuning/tune.py` hardcodes the `forecasting/univariate` base path, and `forecasting/multivariate/tuning/tune.py` hardcodes the `forecasting/multivariate` base path.
Caveat: only the `rf_emf/*` jobs ship an `ablation.yaml`.
The five `rf_emf_mv/*` job folders contain only `args.yaml`, so the tuner cannot grid-search the multivariate-specific models out of the box; it would not find an ablation file.

## Usage: conformal prediction and uncertainty quantification

Conformal prediction is shared between the two setups (`forecasting/<setup>/conformal/`, byte-identical between them).

- `coverage.py` is used at train/test time when `conf.conf` is set.
  It calibrates per-step L1 nonconformity scores on the validation set, computes uncorrected and Bonferroni-corrected critical scores per horizon and channel (from `conf.alpha`, default 0.1), forms prediction intervals as `pred ± scores`, and reports independent (per-horizon) and joint (all-horizon) coverage plus interval widths.
- `tradeoff.py` is an offline post-hoc script that computes the paper's Trade-off Score (TOS), combining weighted-average coverage with a sigmoid-transformed mean interval width.
  It reads a results CSV and is invoked separately:

```bash
python forecasting/univariate/conformal/tradeoff.py <dataset> <pred_len> <alpha>
```

## Usage: multivariate EMF analysis CLI

The `multivariate/` tool visualizes the processed measurement CSVs and has its own detailed documentation in [`multivariate/README.md`](./multivariate/README.md).

```bash
cd multivariate/src
../../venv/bin/python emf_multivariate.py <command> [options]
```

Commands are `all`, `emf-levels`, `heatmaps`, and `normalize`.
Paths are resolved relative to the script, so the working directory does not matter: it reads `data/multivariate/processed/` and writes `results/figures/` (override with `--outdir`).
See `multivariate/README.md` for the full option tables and dataset labels.

## Experiment tracking and HPC

Neptune.ai logging is optional and free for researchers and students.
Set `neptune: True` in `args.yaml` (under the `exp` category) and provide `NEPTUNE_API_TOKEN` in the environment (a `.env` file is supported).
With Neptune enabled the run logs to the configured project and uploads the config and model checkpoint; otherwise all logs stay offline under `logs/`.

To aggregate a hyperparameter run, `forecasting/<setup>/analysis/neptune/get_results.py` reads a `<model>.yaml` from `forecasting/<setup>/analysis/neptune/ablations/`, iterates every run, reports the best result by a `deciding_metric` (for example MSE), and saves `results.csv`.

Cluster submission uses `submitit` through `forecasting/<setup>/jobs/hpc/submit.py`, driven by the SLURM / Compute Canada configuration and the `--cc` path of the tuner.

## License

MIT License (declared in `pyproject.toml`).

TODO: a `LICENSE` file is not yet present at the repository root and should be added.

## Original paper and citation

The methods reproduced here are from Mootoo, Tabassum, and Chiaraviglio (IEEE Transactions on Network Science and Engineering, 2025).
If you use this work, please cite the original paper:

```bibtex
@article{mootoo2025emforecaster,
  title={EMForecaster: A deep learning framework for time series forecasting in wireless networks with distribution-free uncertainty quantification},
  author={Mootoo, Xavier and Tabassum, Hina and Chiaraviglio, Luca},
  journal={IEEE Transactions on Network Science and Engineering},
  year={2025},
  publisher={IEEE}
}
```

## Acknowledgments

Full credit for the EMForecaster method, the paper, and the underlying measurement datasets goes to the original authors, Xavier Mootoo, Hina Tabassum, and Luca Chiaraviglio.
This repository only reproduces and extends their work.

## Contact

Daniel Chahine (repository owner) - chahinedaniel0@gmail.com - https://github.com/DanielChahine0/emforecaster-repo.
For access to the proprietary measurement data, contact the original paper's authors (Luca Chiaraviglio).
