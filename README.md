# EMForecaster (Reproduction Repository)

A personal reproduction and extension of the [EMForecaster paper](https://ieeexplore.ieee.org/document/11095417) by Xavier Mootoo, Hina Tabassum and Luca Chiaraviglio (IEEE Transactions on Network Science and Engineering, 2025).
It forecasts electromagnetic-field (EMF) exposure from wireless networks with deep-learning models, and attaches calibrated uncertainty intervals to every forecast using conformal prediction.

This is **not** the official paper repository.
It is maintained by Daniel Chahine.
The paper is included as [`pdfs/EMForecaster.pdf`](./pdfs/EMForecaster.pdf).

## What this repository does

1. **Reproduces the paper.**
   It trains EMForecaster and its baselines on single EMF exposure series and reports point-forecast error (MSE, MAE) and conformal coverage.
2. **Extends it to many frequency bands.**
   The measurement data has 27-29 frequency bands per site.
   The repo compares three ways to forecast them:
   - **Multivariate:** one model sees all bands at once.
   - **Per-channel:** one independent model per band.
   - **Cluster:** one model whose per-band layers are shared within learned groups of bands, using the [Channel Clustering Module (CCM)](https://arxiv.org/abs/2404.01340).
3. **Repeats the comparison on a second domain.**
   The same three approaches run on VoIP call-quality traces, to check whether the findings hold outside EMF data.

## Quickstart

Requires Python 3.11-3.13.
The pinned `requirements.txt` targets Python 3.12 on macOS (Apple silicon).

```bash
git clone https://github.com/DanielChahine0/emforecaster-repo.git
cd emforecaster-repo

python3.12 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

Install IBM Plex Sans, the typeface every figure uses (once per machine, no root needed).
Without it, figures fall back to another font and one figure-layout test fails.

```bash
(cd results && python -m emfres.fonts)
```

Run the smoke test, a 5-epoch EMForecaster run on ED2022 that checks the whole pipeline works.
It takes about 10 minutes on an Apple-silicon laptop and writes to `logs/multivariate/EMForecaster/ed2022/L20_O30_pf/`.

```bash
python main.py rf_emf_mv/test --quick
```

Preview a full sweep without training anything:

```bash
python main.py --grid --setup multivariate --dry_run
```

Rebuild the results tables from the logs:

```bash
python results/make.py extract
python results/make.py tables
```

Run the tests (`pytest` is a dev dependency, not in `requirements.txt`):

```bash
pip install pytest
python -m pytest forecasting results/tests
```

## Repository layout

```
.
├── main.py                # Entry point: run one job, or a whole grid of jobs
├── forecasting/           # The forecasting package
│   ├── univariate/        #   Single-series setup (the paper's original experiments)
│   ├── multivariate/      #   Many-band setup (multivariate, per-channel and cluster runs)
│   ├── grids.py           #   Definitions of every experiment grid
│   └── perchannel.py      #   Driver that trains one model per band
├── data/                  # Measurement datasets (see Data)
├── logs/                  # One folder per finished run: config, metrics, plots
├── results/               # Tables, figures and the CSVs behind them
├── multivariate/          # Standalone tool for plotting the raw EMF measurements
├── docs/                  # Detailed reference docs (start with docs/grids.md)
├── TimeSeriesCCM-main/    # Vendored code from the CCM paper (clustering layers)
├── pdfs/                  # Reference papers
├── pyproject.toml         # Package metadata
└── requirements.txt       # Pinned dependencies
```

`pip install -e .` installs the distribution `emforecaster`, but the importable package is `forecasting`.
`import emforecaster` will not work.

## Data

All datasets live in `data/` as CSV files.

| Dataset id | File | Kind | Used by |
|---|---|---|---|
| `rf_emf_det` | `data/univariate/DET_Monitoring.csv` | Single exposure series (V/m) | `univariate_series` grid |
| `rf_emf_ptv` | `data/univariate/PTV_Monitoring.csv` | Single exposure series (V/m) | `univariate_series` grid |
| `rf_emf_ptv2` | `data/univariate/PTV_Monitoring2.csv` | Single exposure series (V/m) | `univariate_series` grid |
| `ED2022` | `data/multivariate/processed/ED2022.csv` | 27-29 frequency bands (V/m) | EMF multivariate grids |
| `UH2022` | `data/multivariate/processed/UH2022.csv` | 27-29 frequency bands (V/m) | EMF multivariate grids |
| `EMF_2023_university_5_with_PTV` | `data/multivariate/processed/EMF_2023_university_5_with_PTV.csv` | 27-29 frequency bands (V/m) | EMF multivariate grids |
| `VOIP_MOBILE` | `data/multivariate/processed/VOIP_MOBILE.csv` | 6 call-quality channels | VoIP grids |
| `VOIP_FIXED` | `data/multivariate/processed/VOIP_FIXED.csv` | 6 call-quality channels | VoIP grids |

Notes:

- `ED2024.csv` and `UH2024.csv` are also in `data/multivariate/processed/`; the loader accepts them, but no grid uses them yet.
- `data/multivariate/unprocessed/` holds the raw sources before normalization; no code reads it.
- The dataloader also knows dataset ids such as `rf_emf` and `rf_emf_tur`, but their files are not in the repo, so those ids fail.
- For the multivariate files, the loader drops the `Timestamp` column, the `EMF_*` aggregate columns and `PTV` (to avoid leaking the target), fills missing readings (gaps up to 240 steps by linear interpolation, longer outages by repeating the nearest reading), and sets the channel count from the data.
- The EMF measurements belong to the original paper's authors.
  Contact Luca Chiaraviglio before reusing them.

### VoIP data

`data/ML_DATASETS/` holds 16 VoIP call traces: `{Mobile, Fixed}` access x 8 codecs (`g711`, `g722`, `g729`, `gsm`, `mpeg16`, `opus`, `spx8000`, `spx16000`).
Each trace is a `Time,MOS,BW,RTT,JIT,DJB,SNR` table of 161-820 rows.

```bash
python data/ML_DATASETS/build_voip.py
```

This pools the traces into `VOIP_MOBILE.csv` and `VOIP_FIXED.csv` (8 traces each, with a `Trace` column naming the codec).
Its only cleaning is to fix three physically impossible `JIT` values in `mob_g729`; RTT outliers are kept on purpose.
The loader splits each trace on its own before joining them, so every split has the same codec mix and no window spans two calls.

## Running experiments

`main.py` has two modes.

### Single job

```bash
python main.py <job_name> [--quick] [--ablation '<json>'] [--ablation_id <int>]
```

A job is a folder holding an `args.yaml` config under `forecasting/<setup>/jobs/exp/`.
Job names starting with `rf_emf_mv` run in the multivariate setup; every other job runs in the univariate setup.

| Setup | Jobs |
|---|---|
| Univariate (single series) | `rf_emf/emforecaster`, `rf_emf/dlinear`, `rf_emf/patchtst`, `rf_emf/tsmixer`, `rf_emf/lstm`, `rf_emf/arima`, `rf_emf/baselines/{cnn,lstm,mlp,transformer}`, `test` |
| Multivariate (many bands) | `rf_emf_mv/emforecaster`, `rf_emf_mv/dlinear`, `rf_emf_mv/patchtst`, `rf_emf_mv/tsmixer`, `rf_emf_mv/transformer`, `rf_emf_mv/lstm`, `rf_emf_mv/arima`, `rf_emf_mv/persistence`, `rf_emf_mv/timesnet`, `rf_emf_mv/test` |

| Flag | What it does |
|---|---|
| `--quick` | Run only the first seed in the config's `seed_list`. |
| `--ablation '<json>'` | Override config values, for example `'{"sl.lr": 0.001}'`. |
| `--ablation_id <int>` | Tag the run (default 1); non-default ids get their own log folder. |

### Grid

```bash
python main.py --grid --setup <grid> [--model ...] [--dataset ...] [--window ...] [--k ...] [--seeds ...] [--jobs N] [--dry_run]
```

A grid runs every model x dataset x window combination ("cell") and skips cells that already finished, so an interrupted sweep resumes where it stopped.

| `--setup` | What it compares | Data | Cells |
|---|---|---|---|
| `univariate_series` | The paper's original single-series experiments | DET/PTV series | 84 |
| `multivariate` | One model over all bands | ED2022, UH2022, EMF2023 | 84 |
| `univariate` | One model per band, averaged | ED2022, UH2022, EMF2023 | 84 |
| `cluster` | One model with CCM band clustering, at several cluster counts `k` | ED2022, UH2022, EMF2023 | 360 |
| `voip`, `voip_perchannel`, `voip_cluster` | The same three approaches on VoIP traces | VOIP_MOBILE, VOIP_FIXED | 32, 32, 96 |

The EMF grids sweep seven models (EMForecaster, DLinear, PatchTST, TSMixer, Transformer, LSTM, ARIMA) over four windows (lookback x horizon: `192x96`, `192x336`, `336x96`, `336x336`).
The cluster grids drop ARIMA, the VoIP grids add a Persistence baseline, and the VoIP grids use shorter windows (`48x12`, `24x6`).

| Flag | What it does |
|---|---|
| `--model`, `--dataset`, `--window` | Run only these models, datasets or windows (comma-separated). |
| `--k` | Cluster grids only: run only these cluster counts. |
| `--seeds` | Train each cell on more seeds, for example `--seeds 3` for 69421-69423. |
| `--jobs` | How many trainings run at once. |
| `--dry_run` | List the cells without training. |

Examples:

```bash
# Everything in the multivariate grid at one window
python main.py --grid --setup multivariate --window 336x96

# One model on one dataset, as a quick trial
python main.py --grid --setup multivariate --model DLinear --dataset ed2022 --window 336x96

# The cluster grid at k=1 (no clustering), the controlled baseline
python main.py --grid --setup cluster --window 336x96 --k 1
```

**[docs/grids.md](./docs/grids.md) is the full grid reference.**
It covers every flag in detail, how the cluster grid works and where it differs from the CCM paper's code, its two evaluation protocols, seeds and reproducibility, and running the per-channel grid faster with CUDA MPS.

## Outputs and results

Every run writes to its own folder:

```
logs/<setup>/<model_id>/<dataset>/L<lookback>_O<horizon>_<cp|pf>/<seed>/
```

`cp` means conformal prediction was on; `pf` means point forecasts only.
Each folder holds the resolved `args.yaml`, the metrics in `log.json`, the best weights `supervised.pth`, and 10 forecast plots from random test positions (`plot_<index>.png`).

To turn the logs into tables:

```bash
python results/make.py extract   # logs/   -> results/frames/*.csv (one row per run)
python results/make.py tables    # frames/ -> results/tables/*.md
```

Start reading at [`results/tables/index.md`](./results/tables/index.md).
[`results/README.md`](./results/README.md) documents every table, figure and check.

## Uncertainty (conformal prediction)

When `conf.conf` is on in a job's config, each run calibrates prediction intervals on the validation set and reports how often the test values fall inside them.
The target error rate is `conf.alpha` (default 0.1, meaning 90% coverage).
It reports coverage per horizon step and over the whole horizon, with and without a Bonferroni correction, plus the interval widths.
The code is in `forecasting/<setup>/conformal/coverage.py`.

The paper's Trade-off Score (TOS), which balances coverage against interval width, is computed offline by `tradeoff.py`.
It uses paths relative to its own folder, so run it from there:

```bash
cd forecasting/univariate/conformal
python tradeoff.py <dataset> <pred_len> <alpha>
```

It reads `forecasting/univariate/analysis/neptune/results/conformal/ctsf.csv`, which is produced from Neptune results (see below).

## Other tools

### Hyperparameter tuning

```bash
python forecasting/univariate/tuning/tune.py <job_name> [--cc] [--acc <account>]
```

The tuner reads the job's `ablation.yaml`, runs every combination in it, and with `--cc` submits them as SLURM jobs on Compute Canada instead of running locally.
Use `forecasting/multivariate/tuning/tune.py` for multivariate jobs.
Only the `rf_emf/*` jobs ship an `ablation.yaml`, so the `rf_emf_mv/*` jobs cannot be tuned out of the box.

### Neptune logging

Logging to [Neptune](https://neptune.ai) is optional.
Set `neptune: True` under `exp` in a job's `args.yaml` and provide `NEPTUNE_API_TOKEN` in the environment (a `.env` file works).
Without it, everything stays local under `logs/`.
`forecasting/<setup>/analysis/neptune/get_results.py` collects a tuning run's results from Neptune into a `results.csv`.

### EMF measurement plots

`multivariate/` is a separate plotting tool for the raw measurement CSVs (EMF levels over time and band correlation heatmaps).
It has nothing to do with model training.

```bash
python multivariate/src/emf_multivariate.py all
```

The commands are `all`, `emf-levels`, `heatmaps` and `normalize`.
See [`multivariate/README.md`](./multivariate/README.md) for the options.

## Further reading

| Doc | Covers |
|---|---|
| [`docs/grids.md`](./docs/grids.md) | Every grid and flag in detail, the cluster method, seeds, CUDA MPS |
| [`forecasting/README.md`](./forecasting/README.md) | The forecasting package and how the two setups relate |
| [`forecasting/univariate/README.md`](./forecasting/univariate/README.md) | The single-series setup: config, models, training |
| [`forecasting/multivariate/README.md`](./forecasting/multivariate/README.md) | The many-band setup and how it differs |
| [`results/README.md`](./results/README.md) | The results pipeline, tables and figures |
| [`multivariate/README.md`](./multivariate/README.md) | The measurement plotting tool |

## License

MIT, as declared in `pyproject.toml`.
A `LICENSE` file has not been added yet.

## Citation

The methods reproduced here are from Mootoo, Tabassum and Chiaraviglio (2025).
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

Full credit for the EMForecaster method, the paper and the measurement datasets goes to the original authors, Xavier Mootoo, Hina Tabassum and Luca Chiaraviglio.
This repository only reproduces and extends their work.

## Contact

Daniel Chahine (repository owner) - chahinedaniel0@gmail.com - https://github.com/DanielChahine0/emforecaster-repo.
For access to or reuse of the measurement data, contact the original paper's authors (Luca Chiaraviglio).
