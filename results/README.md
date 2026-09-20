# results/

Tables, figures, and the tidy CSVs behind both.

One rule governs the layout: **tracked text, ignored pixels.** Everything with a
number in it is committed and diffs meaningfully; everything made of pixels is
regenerated on demand and git-ignored. `results/` tracks about 1 MB.

## Rebuild everything

```bash
python results/make.py extract      # logs/        -> frames/*.csv
python results/make.py tables       # frames/      -> tables/*.md + csv twins
python results/make.py reconcile    # check the frozen report against frames/

python results/plot_metrics.py          # the benchmark families
python results/plot_complexity.py
python results/plot_trustworthiness.py
python results/plot_L_ladder.py
python results/plot_paradigms.py
python results/emfres/fig_curves.py --logs-dir logs/univariate --combined
python multivariate/src/emf_multivariate.py all   # emf-levels + heatmaps
```

`extract` reads every `log.json` under `logs/` (1,712 today, of which 1,658 are
published — see *Which runs count*) and writes one row per run. Nothing
downstream reads a log file, and nothing downstream parses markdown.

## Layout

| Path | What lives here | Tracked |
| :--- | :--- | :--- |
| `emfres/` | All the code. One flat package. | yes |
| `specs/` | Declared run exclusions, with the reason for each | yes |
| `frames/` | `runs.csv`, `cells.csv`, `MANIFEST.json` — the numeric API | yes |
| `tables/` | Generated markdown, one file per section, plus CSV twins | yes |
| `notes/` | Hand-written prose. Never generated. | yes |
| `figures/` | Every figure. 100% derived. | **no** |
| `scratch/` | Exploratory renders | **no** |
| `archive/` | Frozen predecessors, incl. the legacy single-file report | yes |

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
their metrics, the last commit that touched `logs/`, and — since 2026-09-19 —
the set of *code* commits the runs themselves were stamped with
(`code_commits`, from `parameters/exp/git_commit`) plus how many runs predate
that stamping (`unstamped_runs`). `runs.csv` carries `git_commit` and
`concurrency` (the grid's `--jobs` while the run trained) per run, and
`cells.csv` carries `commits`, the distinct commits behind a cell. It changes
when the inputs change and not otherwise, so a rebuild leaves `git status` clean.

`make.py reconcile` recomputes every published MSE/MAE cell from the CSVs and
compares it with the frozen snapshot named by `PUBLISHED` in `emfres/cli.py`,
currently `archive/2026-09-19_tables.md` — 390 of 390 match. `make.py tables`
runs that same check before writing and refuses if any published value fails to
reproduce, so the CSV path and the frozen report cannot silently diverge.
The previous snapshot, `archive/2026-08-11_tables.md`, is kept as provenance:
its cluster section describes the upstream-wrapper runs that were retired to
`logs/_archive/cluster/` on 2026-08-25, which is why it was superseded.

The gate is deliberately **value-level, not text-level**. It used to diff the
rendered markdown against the archive and refuse on any difference, which fails
the moment a section legitimately grows: restoring the cluster lookback ladder
added nine windows and reshaped the parameter table from one column to ten,
changing 131 lines while leaving every published number identical. A tripwire
that fires on correct work only teaches you to pass `--force`. Growth is now
reported as a `NOTE`; only a changed value is fatal.

## Which runs count

`specs/runs.yaml` declares what `extract` drops and why. It still excludes the
54 cluster runs stamped seed 42 / seed 43 — a different configuration, not
replicate seeds (`notes/ccm-hyperparameters.md` §8) — though since 2026-08-25
those runs live under `logs/_archive/cluster/` and are skipped as an archive
anyway; the rule stays so they cannot come back silently. `--no-spec` loads
everything.

Cluster publishes **105** runs: the native-CCM k-sweep at L336/O96 rerun on
2026-08-25 (`forecasting/multivariate/models/ccm.py` threading upstream's
clustering layers through the repo's own DLinear / PatchTST / TSMixer). The
earlier upstream-wrapper sweep and the 27-run lookback ladder are in
`logs/_archive/`; the ladder survives as `figures/l-ladder/*.png`.

## Known gaps

- The frozen snapshot (`PUBLISHED` in `emfres/cli.py`, currently
  `archive/2026-09-19_tables.md`) is what `make.py tables --check` and
  `make.py reconcile` compare against. It covers only the cells published when
  it was frozen; anything added since is unguarded, which is the cost of
  checking against a snapshot rather than a second implementation. Re-freeze
  it (copy `tables/all.md` into `archive/` and repoint `PUBLISHED`) whenever a
  new set of numbers is accepted as published.
- Every run in the live tree predates provenance stamping. `specs/provenance.yaml`
  assigns 707 of them a commit by setup and `exp/time` window (`commit_source`
  = `inferred` in `runs.csv`; the audit behind each rule is written next to it),
  the 168 single-series runs get none, and runs made from 2026-09-19 on stamp
  the commit themselves (`commit_source` = `log`). `MANIFEST.json` counts the
  three under `provenance` and lists the resulting `code_commits` — three today
  (`401ba0a2`, `cbaab8d0`, `e30d4f08`), which is the reason the Phase 2 rerun in
  `.personal/PLAN.md` exists.
- `frames/epochs.csv` (per-epoch losses) is opt-in behind `--epochs` and
  git-ignored: 11.7 MB of diagnostic curves is not a published number.
- `emfres/fig_curves.py` still walks `logs/` directly rather than reading
  `epochs.csv`, so loss curves are the one family outside the frames pipeline.
