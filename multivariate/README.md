# Multivariate EMF Analysis

Analysis and visualization of frequency-selective RF-EMF time-series data.
`src/emf_multivariate.py` is the command line tool for generating the EMF level plots, correlation heatmaps, and window analyses from the measurement datasets.

## Requirements

Use the repository virtual environment (Python 3.12); it already has the dependencies (pandas, numpy, matplotlib).
It lives at the repository root, so from `multivariate/src/` it is `../../venv/bin/python`.

The `window-analysis` command additionally needs an Excel engine (`openpyxl`) to read the antenna-factor `.xlsx`.
If it is not installed, the command still runs and falls back to a default antenna factor.

## Datasets

The datasets live at the repository root under `data/multivariate/`:

```
data/multivariate/processed/     E-field CSVs the tool reads (V/m per frequency)
data/multivariate/unprocessed/   raw-import copies of the same CSVs
```

The processed folder holds five datasets:

| File | Label | Output folder |
|------|-------|---------------|
| EMF_2023_university_5_with_PTV.csv | EMF_2023_university | EMF2023 |
| UH2022.csv | UH_2022 | UH2022 |
| ED2022.csv | ED_2022 | ED2022 |
| UH2024.csv | UH_2024 | UH2024 |
| ED2024.csv | ED_2024 | ED2024 |

UH = University Hospital, ED = Electronic Department.
Each row is a timestamped reading; each numeric column header is a frequency band in MHz, with values in V/m.
Each file also carries the `Timestamp` column and the `EMF_2G/3G/4G/5G`, `EMF_Iliad/TIM/VF/W3`, and `EMF_Total` aggregation columns.

## Running

All paths are anchored to the script location, so the working directory does not matter.

```bash
cd multivariate/src
../../venv/bin/python emf_multivariate.py <command> [options]
```

List the commands, or show the options for a single command:

```bash
../../venv/bin/python emf_multivariate.py --help
../../venv/bin/python emf_multivariate.py <command> --help
```

### Examples

```bash
# Generate every figure
../../venv/bin/python emf_multivariate.py all

# Time-trend plots for two datasets only
../../venv/bin/python emf_multivariate.py emf-levels --datasets UH_2022 ED_2022

# Heatmaps without spike clipping
../../venv/bin/python emf_multivariate.py heatmaps --no-freq-clip

# Rebuild the processed CSVs in place
../../venv/bin/python emf_multivariate.py normalize

# Window analysis from a raw measurement file
../../venv/bin/python emf_multivariate.py window-analysis \
    --input /path/to/fileCP_2024_UNIVERSITY_HOSPITAL.txt \
    --antenna-factors /path/to/9018-18835.xlsx
```

## Commands

### `all`

Runs `emf-levels` then `heatmaps` in one go.
It accepts every `emf-levels` option, and the clipping settings also apply to the heatmaps step.

### `emf-levels`

EMF level (V/m) time-trend plots.
For each dataset it plots the operators (W3, Iliad, TIM, VF), the technologies (2G, 3G, 4G, 5G), and every individual frequency, each over the full measurement period and over a roughly 2-week zoom.
It writes one combined stacked figure, one operators-and-technologies figure, and six per-panel figures into `src/result/EMF levels/<folder>/`.

| Option | Default | Meaning |
|--------|---------|---------|
| `--datasets LABEL [LABEL ...]` | all | Restrict to a subset of datasets (use the labels above). |
| `--zoom-days N` | 14 | Length of the auto-derived zoom window in days (the 2023 dataset keeps its curated window). |
| `--remove-outliers-std N` | off | Drop points outside +/- N standard deviations per series before plotting. |
| `--freq-legend-mode {colorbar,legend}` | colorbar | How the all-frequencies panels are keyed: a frequency colorbar, or a multi-column legend. |
| `--freq-cmap NAME` | turbo | Matplotlib colormap for the all-frequencies panels. |
| `--freq-clip-quantile Q` | 0.999 | Quantile used for per-frequency spike clipping. |
| `--freq-clip-factor F` | 3.0 | A sample above `F x` the quantile is treated as an artifact and dropped. |
| `--no-freq-clip` | off | Disable per-frequency spike clipping entirely. |

### `heatmaps`

Correlation heatmaps per dataset: frequency bands, technology aggregations (2G/3G/4G/5G), and operator aggregations (Iliad/TIM/VF/W3).
For each dataset it writes a combined trio plus the three individual heatmaps into `src/result/Heatmaps/<folder>/`, and an all-datasets comparison grid at the `Heatmaps` root when two or more datasets are processed.

| Option | Default | Meaning |
|--------|---------|---------|
| `--datasets LABEL [LABEL ...]` | all | Restrict to a subset of datasets (use the labels above). |
| `--freq-clip-quantile Q` | 0.999 | Quantile used for per-frequency spike clipping. |
| `--freq-clip-factor F` | 3.0 | A sample above `F x` the quantile is treated as an artifact and masked. |
| `--no-freq-clip` | off | Disable per-frequency spike clipping entirely. |

### `normalize`

Rebuilds the UH/ED processed CSVs in place into the same schema as the 2023 reference file.
It strips a UTF-8 BOM, renames the timestamp column to `Timestamp`, drops empty or unnamed trailing columns, snaps off-canonical 2022 frequency labels to the project's band centres, and appends the `EMF_*` aggregation columns.
It is idempotent: re-running recomputes from the frequency columns rather than stacking new columns.
The 2023 reference file is left untouched.

| Option | Default | Meaning |
|--------|---------|---------|
| `--files NAME [NAME ...]` | UH2022.csv ED2022.csv UH2024.csv ED2024.csv | CSV file names under `data/multivariate/processed/` to normalize. |

### `window-analysis`

Operates on a raw fileCP `.txt` (dBm per frequency) rather than the processed CSVs.
It filters to the longest continuous measurement period, converts dBm to E-field (V/m) using antenna-factor calibration, adds the technology/operator/total aggregations, and produces zoom-in plots for the 1, 3, 7, and 14-day windows centred on the middle of that period.
Output goes to `src/result/window_analysis_<timestamp>/`.

| Option | Default | Meaning |
|--------|---------|---------|
| `--input PATH` | fileCP_2024_UNIVERSITY_HOSPITAL.txt | Raw fileCP `.txt` in dBm (relative to your working directory). |
| `--antenna-factors PATH` | 9018-18835.xlsx | Antenna-factor `.xlsx` used to convert dBm to V/m. |

## Outputs

Everything is written under `src/result/` (created automatically):

```
result/EMF levels/<folder>/                       from emf-levels
    Combined (operators, technologies, frequencies).png/.pdf
    Combined (operators, technologies).png/.pdf
    Operators (full period).png/.pdf
    Operators (2 week zoom).png/.pdf
    Technologies (full period).png/.pdf
    Technologies (2 week zoom).png/.pdf
    All frequencies (full period).png/.pdf
    All frequencies (2 week zoom).png/.pdf

result/Heatmaps/<folder>/                          from heatmaps
    Combined (frequency, technology, operators).png/.pdf
    Frequency bands.png/.pdf
    Technology bands.png/.pdf
    Operators.png/.pdf
result/Heatmaps/All datasets comparison.png/.pdf   (when 2+ datasets are processed)

result/window_analysis_<timestamp>/plots/          from window-analysis
    emf_technology_<label>_<window>_window.png/.pdf
    emf_operator_<label>_<window>_window.png/.pdf
    emf_<label>_all_windows_summary.png/.pdf
result/window_analysis_<timestamp>/window_analysis_summary.txt
```

Every figure is saved as both `.png` (300 dpi) and `.pdf`, with a serif font (Times New Roman).

## Notes

Per-frequency spike clipping.
Some per-frequency columns contain isolated corrupted samples (for example a single ~119 V/m reading at 1835 MHz) that would otherwise dominate a correlation or flatten a time-trend panel.
By default any value above `--freq-clip-factor` times the `--freq-clip-quantile` of a frequency's data is treated as an artifact and dropped.
The `EMF_*` aggregation columns are never clipped, so the technology and operator outputs are unaffected.
`emf-levels` clips against a single global threshold over all plotted frequencies; `heatmaps` clips per frequency column.
Pass `--no-freq-clip` to see the raw data.

Aggregation columns.
`EMF_2023_university` ships the `EMF_*` aggregations precomputed; for the other datasets they are reused when present and otherwise computed on the fly by root-sum-square (RSS) combining of the relevant frequencies.
