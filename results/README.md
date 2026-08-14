# results/

Tables, figures, and the tidy CSVs behind both.

One rule governs the layout: **tracked text, ignored pixels.** Everything with a
number in it is committed and diffs meaningfully; everything made of pixels is
regenerated on demand and git-ignored. `results/` tracks about 1 MB.

## Rebuild everything

```bash
python results/make.py extract      # logs/        -> frames/*.csv
python results/make.py tables       # frames/      -> tables/*.md + csv twins
python results/make.py reconcile    # check tables.md against frames/

python results/plot_metrics.py          # the benchmark families
python results/plot_complexity.py
python results/plot_trustworthiness.py
python results/plot_L_ladder.py
python results/plot_paradigms.py
python results/emfres/fig_curves.py --logs-dir logs/univariate --combined
python multivariate/src/emf_multivariate.py all   # emf-levels + heatmaps
```

`extract` reads every `log.json` under `logs/` (1,685 today) and writes one row
per run. Nothing downstream reads a log file, and nothing downstream parses
markdown.

## Layout

| Path | What lives here | Tracked |
| :--- | :--- | :--- |
| `emfres/` | All the code. One flat package. | yes |
| `frames/` | `runs.csv`, `cells.csv`, `MANIFEST.json` — the numeric API | yes |
| `tables/` | Generated markdown, one file per section, plus CSV twins | yes |
| `notes/` | Hand-written prose. Never generated. | yes |
| `archive/` | Frozen predecessors, read-only | yes |
| `figures/` | Every figure. 100% derived. | **no** |
| `scratch/` | Exploratory renders | **no** |
| `tables.md` | Legacy single-file report, still the plotters' fallback | yes |

## Naming

Figure paths are **generated, never typed** — `emfres/naming.py` builds every
one from the slice actually plotted. There is no `--outdir` that decides what a
file means.

```
figures/<family>/[<setup>/]<axes...>_<metric>[_<view>].<ext>

figures/by-dataset/multivariate/l48-480x48_o96_mse.png
figures/cluster-sweep/ed2022_l336_o96_kall_mae.png
figures/trustworthiness/multivariate/l336_o96_coverage.png
figures/l-ladder/ed2022_l48-480x48_o96_mse_facet-model.png
figures/emf-levels/ed2022/operators_zoom14d.png
figures/loss-curves/det/dlinear_l192_o336_loss.png
```

The rules:

- **Charset.** Lowercase, `[a-z0-9._-]`. No spaces, parentheses or capitals.
- **Two separators.** `_` between axis tokens, `-` inside one. So `mae-vs-tt`
  is one token and `multivariate_l336_o96_mse` is four.
- **Numbers label themselves.** `l336`, `o96`, `k5` — a bare number never
  appears, so a name parses without positional guessing.
- **The family is the directory,** never a filename token.
- **Pooling is named, never implied.** A figure averaged over the lookback
  ladder is `l48-480x48_o96`, not a bare `mse.png`. This is the important one:
  omission would let today's filename quietly mean something else the day
  another window is added.

## Vocabulary

"univariate" means three different things in this repo — the per-band grid's
CLI name, the single-series log tree, and a `plot_metrics` section — so inside
`results/` it is retired. The four setups are:

| Setup | Is | `logs/` directory |
| :--- | :--- | :--- |
| `multivariate` | all channels, one model | `logs/multivariate` |
| `perchannel` | one model per frequency band | `logs/univariate (per channel)` |
| `series` | a single aggregate series | `logs/univariate` |
| `cluster` | CCM soft channel clustering, swept over k | `logs/cluster` |

Datasets use their log spelling: `ed2022`, `uh2022`, `emf2023`, `ed2024`,
`uh2024`, `det`, `ptv`, `ptv2`. Display strings live in `emfres/vocab.py` and
appear only inside rendered content.

## Provenance

`frames/MANIFEST.json` records the run count, a digest over the input paths and
their metrics, and the last commit that touched `logs/`. It changes when the
inputs change and not otherwise, so a rebuild leaves `git status` clean.

`make.py reconcile` recomputes every published MSE/MAE cell from the CSVs and
compares it with `tables.md` — 498 of 498 match. `make.py tables` refuses to
write if any section has drifted, so the CSV path and the legacy markdown
cannot silently diverge while both exist.

## Known gaps

- `clustering/make_tables.py` is a fork of the extractor's loader. It holds the
  only k-major standard-deviation renderer in the repo, which should be lifted
  into `tables.py` before it is deleted — more relevant now that cluster is a
  5-seed setup and `cells.csv` carries `mse_std` / `n_seeds`.
- `tables.md` and `plot_metrics.parse_tables` survive as the `--source tables`
  fallback and as the reconciliation gate. Both go when the legacy file does.
- `frames/epochs.csv` (per-epoch losses) is opt-in behind `--epochs` and
  git-ignored: 11.7 MB of diagnostic curves is not a published number.
