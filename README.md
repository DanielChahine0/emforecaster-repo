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

Model families available through `get_model`: PatchTST, recurrent models (LSTM/RNN/GRU/Mamba via `backbone_id`), Linear, DLinear, EMForecaster, ModernTCN, TimesNet, KNN, TSMixer, and four RF baselines (RF_EMF_MLP, RF_EMF_CNN, RF_EMF_LSTM, RF_EMF_Transformer).

## Repository layout

```
.
├── main.py                 # Root entrypoint; picks the setup by job name
├── forecasting/            # Forecasting package
│   ├── univariate/         # Univariate forecasting setup
│   └── multivariate/       # Multivariate forecasting setup
├── multivariate/           # Standalone EMF measurement-analysis CLI (+ its own README)
├── data/                   # Proprietary EMF measurement CSVs (see Data)
├── logs/                   # Offline training-run logs (per run, per seed)
├── results/                # Generated outputs (multivariate/ from the CLI, univariate/ from training analysis)
├── pyproject.toml          # Poetry metadata, MIT license, Python >=3.10,<3.14
├── requirements.txt        # Fully pinned dependencies (Python 3.12 / macOS arm64)
└── EMForecaster Paper.pdf  # Original paper (reference)
```

Each training setup mirrors the same subtree: `config`, `exp`, `models`, `layers`, `conformal`, `tuning`, `jobs`, `utils`, `analysis`.
Note that "analysis" exists in two unrelated places: `forecasting/<univariate|multivariate>/analysis/` analyzes training runs (loss curves, bar graphs, ADF/ARIMA, Neptune fetching), while `multivariate/` analyzes measurement CSVs.

## Installation

Requires Python `>=3.10,<3.14`.
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

## Usage: running experiments

Run from the repository root:

```bash
python main.py <job_name> [--quick] [--ablation '<json>'] [--ablation_id <int>]
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

- `rf_emf_mv/dlinear`
- `rf_emf_mv/emforecaster`
- `rf_emf_mv/patchtst`
- `rf_emf_mv/timesnet`
- `rf_emf_mv/tsmixer`

The `forecasting.multivariate` setup also mirrors the univariate `rf_emf/*` and `test` jobs, but because dispatch keys off the `rf_emf_mv` prefix, those still run through the `forecasting.univariate` setup.

Flags:

- `--quick` runs a single seed (`seed_list[:1]`).
- `--ablation '<json>'` takes a JSON string of dotted overrides (for example `{"sl.lr": 0.001}`) mapped onto the config; invalid JSON exits with an error.
- `--ablation_id <int>` tags the run (default 1) and is used in output paths.

By default `main.py` iterates over every seed in `args.exp.seed_list`, building a fresh `Experiment` per seed.
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
Paths are resolved relative to the script, so the working directory does not matter: it reads `data/multivariate/processed/` and writes `results/multivariate/`.
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
