# results/

Tables, figures, and the tidy CSVs behind both.

One rule governs the layout: **tracked text, ignored pixels.** Everything with a
number in it is committed and diffs meaningfully; everything made of pixels is
regenerated on demand and git-ignored. `results/` tracks about 2.3 MB.

## Rebuild everything

```bash
python results/make.py extract      # logs/        -> frames/*.csv
python results/make.py tables       # frames/      -> tables/*.md + csv twins
python results/make.py reconcile    # hold the frozen report's values against frames/
python results/make.py figures      # frames/      -> the paradigm-comparison figures
python -m pytest results/tests      # provenance, paradigm comparison, baselines, figures

python results/plot_metrics.py          # the benchmark families
python results/plot_complexity.py
python results/plot_trustworthiness.py
python results/plot_L_ladder.py
python results/plot_paradigms.py
python results/emfres/fig_curves.py --logs-dir logs/univariate --combined
python multivariate/src/emf_multivariate.py all   # emf-levels + heatmaps
```

`extract` reads every `log.json` under `logs/` outside `logs/_archive/` (3,469
today: 2,109 EMF, 1,360 VoIP — see *Which runs count*) and writes one row per run. Nothing
downstream reads a log file, and nothing downstream parses markdown. A log
without `sl_test/interval_width_mean`, the last key a finished run writes, is a
run still training (or one that crashed): it is skipped and counted in a `NOTE`,
so extracting mid-sweep never publishes half a run.

## Layout

| Path | What lives here | Tracked |
| :--- | :--- | :--- |
| `emfres/` | All the code. One flat package. | yes |
| `specs/` | Declared run exclusions, provenance rules and the gate's expected changes, with the reason for each; the solo timing pass | yes |
| `frames/` | `runs.csv`, `cells.csv`, `bands.csv`, `timing.csv`, `MANIFEST.json` — the numeric API | yes |
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

figures/paradigm-compare/emf/gap-forest_l336_o96_mae.png
figures/per-band/uh2022/pc-vs-mv_l336_o96_mse.png
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

## Figures

`make.py figures` draws the paradigm comparison from `frames/` alone
(`emfres/figures.py`); `--family emf|voip|all`, `--only P1,P5`, `--frames`
and `--outdir` pick what is drawn, from which frames, under which root. Every
number goes through `emfres/figdata.py`, which takes it from
`emfres/paradigms.py`, the module behind `tables/paradigms.md`: the same seed
means, matched seeds, tie band, Wilcoxon test and validation-chosen k (fixed
membership, the primary cluster protocol), so a figure cannot name a winner
or a best k the table does not.

| Id | Path under `figures/` | Question |
| :--- | :--- | :--- |
| P1 | `paradigm-compare/emf/dots_l336_o96_{mae,mse}` | each model under each paradigm, seeds as ticks |
| P2 | `paradigm-compare/<family>/gap-forest_<window>_{mae,mse}` | every cell's gap against the seed-noise tie band; W/T/L and p per panel |
| P3 | `per-band/<dataset>/{pc-vs-mv,best-k-vs-k1}_<window>_mse` | is a gap broad, or carried by a few bands (needs `bands.csv`) |
| P4 | `paradigm-compare/voip/dots_l24-48_o6-12_{mae,mse}` | VoIP, datasets x windows, Persistence as a rule |
| P5 | `paradigm-compare/<family>/k-gain_<window>_{mae,mse}` | clustering gain against k per model; val k ringed, test-best k marked, batch protocol dashed |
| P6 | `paradigm-compare/scoreboard_<windows>` | the four comparisons' W/T/L, EMF and both VoIP windows, with and without a cell that does not train |
| P7 | `paradigm-compare/emf/cost_l336_o96_{mae,mse}` | error and uncontended cost relative to the multivariate run (needs `timing.csv`) |
| P8 | `paradigm-compare/<family>/calibration_<windows>` | coverage and interval width per paradigm |
| P9 | `paradigm-compare/<family>/seed-skill_<windows>` | every seed's skill over the lookback mean, with the Transformer's trains threshold |
| P10 | `series/dots_l336_o96_{mae,mse}` | the single-series grid, titled as the separate dataset it is |

The paradigm owns the colour: multivariate Charcoal Blue filled, per-channel
the same hue hollow, cluster Verdigris (a diamond for the chosen k, joined to
a small dot for k = 1 and labelled with its k beside it, on its outer side
away from the k = 1 dot unless that side is more than two marker widths away
and the other nearer; either side when k = 1 is chosen). A label in a dot
plot's lane steps clear of every mark, seed tick and other label
(`figkit.settle_labels`), then the axis widens to hold it; only reference
rules and gridlines run under its opaque box. Models are rows or facets,
named in text; outcomes are Verdigris (challenger better), Burnt Peach
(baseline better) and a hollow muted ring (tie). Where a figure colours by
dataset (P5, P7) it takes `style.DATASET_COLORS`, three hues that are none of
those. A cell where no seed trains in any paradigm (the VoIP Transformer) is
marked † and muted wherever it is named, and P3 leaves it out of the share
strip, whose bars are each band's share of the dataset's summed band gaps
(the gap summed over models, so models that disagree on a band cancel there)
and add up to 100%. ARIMA's cluster lane reads n/a, as it is per-band by
construction.
Type is IBM Plex Sans (`style.rc_fonts`), sizes the IEEEtran `paper` preset
unless `--style slide`.

The figures render from whatever the frames hold. A model whose sweep is still
training is drawn faded (and hollow in P5), labelled partial, and left out of
every count, exactly as the tables count it pending; one whose planned sweep
has no run yet on a dataset reads pending in the dot plots' cluster lane and
on the gap forests' zero rule (so the drawn pending cells match the count in
each panel title), and is named in a note under P3 and P5. A figure the
frames give nothing to draw is named in a `NOTE`. Re-extracting after the
sweep draws it all in full with no code change. The model list, datasets and k
values all come from the frames. ED2024 and UH2024 never appear
(`figdata.HIDDEN_DATASETS`; the EDA tool leaves them out unless named). Each
run spot-checks the seed means the figures are drawn from against
`cells.csv` and exits non-zero on a mismatch (`figdata.spot_check`,
`spot_check_series`): the accuracy seed means behind P1, P2, P4, P5, P6 and
P9, P8's coverage and width, P7's multivariate and cluster training time and
P10's series means, each wherever `cells.csv`'s plain mean is that seed mean
(every seed present and, per-channel, complete). It checks the numbers, not
the drawn marks. Outside it, for want of a `cells.csv` counterpart: P7's
per-channel cost (the solo estimate in `timing.csv`) and P3's band ratios
(`bands.csv`). P2's gaps and P6's counts are computed from the checked seed
values by the functions behind `tables/paradigms.md`.

`figures/_archive/` holds the 56 figures of 2026-08-14 (`l192_o96_*`,
`l48-480x48_o96_*`: the ED2022 lookback ladder and the L192 window of the
multivariate, per-channel and cluster sweeps, whose runs are now in
`logs/_archive`) that no current frame regenerates; it is git-ignored like the
rest of `figures/`.

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
their metrics, the last commit that touched `logs/`, and the **training-code
versions** behind the runs (`code_versions`, each with the commit ids that
carry it), how many runs have no establishable code (`unknown_code_runs`) and
how each run's commit was established (`provenance`). It changes when the
inputs change and not otherwise, so a rebuild leaves `git status` clean.

Provenance is judged on the code a run executed, not on its commit id. A code
version is a fingerprint of `forecasting/`, `main.py`, `pyproject.toml` and the
vendored CCM tree at that commit, ignoring markdown, tests (`test_*.py`) and
the data fixtures beside them such as `forecasting/test_golden.json`
(`frames.code_fingerprint`). Commits that differ only in `results/`, `logs/`,
`data/`, tests or docs therefore share a version, which is what lets results
work be committed while a sweep is running without splitting the sweep.

Per run, `runs.csv` carries `git_commit` (stamped, or dated by a rule),
`git_dirty`, `commit_source` — `log` (stamped, clean tree), `attested` (stamped
on a dirty tree and resolved by an evidenced rule), `dirty` (unresolved),
`inferred` (logged before stamping, dated by a rule) — plus `code_version` and
`concurrency` (the grid's `--jobs` while it trained), plus `lookback_mse` and
`skill`: the error of forecasting each window's lookback mean on the same
windows, and the run's skill over it (1 - mse / lookback_mse; zero means no
better than the trivial forecast, see `emfres/baselines.py` for why that is a
measure and not a collapse flag), and `train_mean_mse`, the error of
forecasting the training split's mean (0 in the scaled space), where a model
without RevIN lands when it collapses. The single-series tree is scored with
its own package's loader. `best_epoch` is the 0-based epoch of `val_loss`, the
checkpoint early stopping kept; 0-2 marks a run that never trained (the
failed per-channel PatchTST bands). `cells.csv` carries
`code_versions`, the distinct versions behind a cell (`?` for a run whose code
is unknown); more than one means the cell is not a single-code seed mean.

A training-code edit that provably leaves every run's numbers alone still makes
a new fingerprint. An `equivalent_code` rule in `specs/provenance.yaml` counts
such a version as another — `from`, `to`, and the `evidence` that shows it
(golden hashes, a bit-exact GPU rerun); a rule without evidence is ignored,
and so, with a `WARNING`, are rules that map one version to two others or form
a cycle (`a -> b` beside `b -> a`), which would otherwise give a swapped,
non-canonical mapping. `runs.csv` keeps the raw fingerprint in `code_version`
and adds the version it counts as in `code_equivalent` (`cells.csv`:
`code_equivalents`);
`MANIFEST.json` lists every raw version under `code_versions`, with
`equivalent_to` on the mapped ones, and the grouping under `code_equivalents`.
`extract` warns about a code split only across versions that remain distinct
after the rules. Today `bb6be0a379d9`, the code the 2026-09-24 CCM sweep trains
on, counts as `21e1374ddc2c`; the August per-channel runs (`21edc003e04e`) are
not declared equivalent and still show as a second version.

`make.py reconcile` holds every value the frozen snapshot publishes (`PUBLISHED`
in `emfres/cli.py`, currently `archive/2026-09-22_tables.md`) against the value
today's frames give, and `make.py tables` runs the same gate before writing.
Three kinds of value are held (`emfres/reconcile.py`). The last column is how
many values `2026-09-22_tables.md` publishes of each; how many of them
reproduce today, and which changes are expected, move with every run the
sweep finishes, so `make.py reconcile --all` is where to read them:

| Kind | What | Held against | Published in the snapshot |
| :--- | :--- | :--- | ---: |
| `cells` | the MSE/MAE tables of Multivariate, Univariate Series, Per Channel and Cluster | `frames/cells.csv`, the column each section prints (Cluster: the fixed means, and the batch ones too; see below) | 342 |
| `scoreboard` | the Paradigm Comparison's scoreboard and sensitivity rows: the three counts, cells, median gap, Wilcoxon p | today's render (the cluster rows: see below) | 48 |
| `sweep` | the CCM sweep: each model's plain cell, its cell at every k, its val k and oracle k | today's render | 21 |

Two values are compared at the coarser of the steps they are known to. A
number is known to its printed places, or to its third significant digit
where that is finer: a Wilcoxon p is printed with `:.3g`, which drops
trailing zeros, so a p of `1` is 1.00 and does not match `0.5`, while
`0.3765` matches `0.37649` and `+1.04%` matches `+1.035%` (printed to more
places, a value agrees at the fewer). The text around a value counts: a tie
mark (≈) that appears, or an `(n=…)` marking a cell still training, is a
change. A scoreboard row is keyed by its comparison
("Per-channel vs Multivariate"), not by its question's wording. The sweep is
read in both layouts it has had: the snapshot's one section per dataset and
metric, which underlined the k with the best *test* value, and today's single
section, whose oracle k is that same choice. A snapshot section is found by
its heading, or by its heading plus a ` · ` suffix (the old sweep's
`… vs. plain · ED2022 · MAE`).

A changed value is **expected** when a rule in `specs/gate.yaml` matches it.
Each rule names the kind and fields it covers, why they changed, and in `now`
the value each of them prints today: a matched value that prints anything
else is unexpected, so a rule forgives the transition its reason quotes
(Tie 10 → 7) and no other value in its columns. The file names the snapshot
its rules were written for: against any other they do not apply. The
declared changes are the per-channel EMForecaster cells (16 bands retrained
at 128x4); ARIMA's cells moving from Tie to the Identical
column, and the per-channel vs multivariate median gap; the k = 1 vs
multivariate median gap, now taken over every counted cell; the sweep's
k ≥ 2 values and oracle k under the fixed protocol; and the Cluster
section's k ≥ 2 cells, which it now prints under the fixed protocol too (63
of the 72 move in the fourth decimal). The sweep's plain and k = 1 values,
the Cluster section's k = 1 cells and the per-channel vs multivariate wins,
losses and p are deliberately not declared. No rule
forgives a value that vanished or turned into `-`: a rule declares a value
that moved, not one that went missing. A change no rule declares fails the
gate; a value the snapshot never had is counted as new and never fails it.

The snapshot's three cluster comparisons (k = 1 vs multivariate, best k vs
k = 1, best k vs multivariate) are **held elsewhere**. They were the logged
batch protocol over the three models then clustered; today's scoreboard rows
of the same name are the fixed protocol over up to six models, which the
snapshot never published. So `held_elsewhere` in `specs/gate.yaml` holds each
of their values against today's batch-protocol sensitivity row, re-counted
over DLinear, PatchTST and TSMixer alone (`render_paradigms(models=…)` in
`emfres/cli.py`), where every count and p has to reproduce. Today's
fixed-protocol values under the old names are a quantity the snapshot never
published, so they are not compared and count as new, but they still have to
be printed: a value missing from its own row, or printed `-` where the
snapshot printed one, fails the gate however well the batch-protocol row
reproduces, and no rule forgives it.

The snapshot's Cluster cells are the logged batch protocol, and today's
section leads with the fixed one. So `hold_cells` holds each of them twice:
against today's fixed mean (`mse_fixed`/`mae_fixed`), where the k ≥ 2 values
moved and the last rule in `specs/gate.yaml` pins each one; and against the
batch mean it was published from (`mse`/`mae`), whose runs nothing touched.
There it has to reproduce, and no rule forgives a change
(`reconcile.BATCH_VIA`). Which protocol a snapshot's Cluster section printed
is read from its first MSE heading (`sl_test/loss` or `sl_test_fixed/loss`),
so after a re-freeze its cells are held against the fixed means alone.

`make.py reconcile --all` lists every change and every new value. A changed
section's `NOTE` says what the gate held of it and, for a partly gated
section, which part is text only: the per-setup sections' tables other than
MSE/MAE, and the paradigm comparison's per-cell tables. Multivariate vs
Univariate, CCM vs Base and the Ultimate Comparison are text only. The same
multivariate and per-channel cells are held in their own sections, and the
cluster cells in the sweep, which in this snapshot covers ED2022 MAE only.

On the tracked frames the gate fails until they are re-extracted: they
predate the fixed-protocol columns, so the sweep cannot render and all 21 of
its published values read as no longer published, the three cluster
scoreboard rows count no cell and print `-` for their median gap and p (12
values their own rows no longer print), and the Cluster section's 90 cells
have no fixed mean to be held against. That section then renders under the
batch protocol, labelled as such, with a warning.

Earlier snapshots are kept as provenance: `2026-09-19_tables.md` holds the
single-seed EMF benchmark that the three-seed Phase 2 rerun replaced, and
`2026-08-11_tables.md` the upstream-wrapper cluster runs retired on 2026-08-25.

The gate is deliberately **value-level, not text-level**. It used to diff the
rendered markdown against the archive and refuse on any difference, which fails
the moment a section legitimately grows: restoring the cluster lookback ladder
added nine windows and reshaped the parameter table from one column to ten,
changing 131 lines while leaving every published number identical. A tripwire
that fires on correct work only teaches you to pass `--force`. Growth is now
reported as a `NOTE`; only a changed value that `specs/gate.yaml` does not
declare is fatal.

## What the frames hold

- **Cluster runs: the fixed protocol beside the batch one.** A cluster run's
  `mse`, `mae`, `val_loss`, `ic`, `jc` and `interval_width` are the logged batch
  protocol, as before. `mse_fixed`, `mae_fixed`, `val_loss_fixed`, `ic_fixed`,
  `jc_fixed` and `interval_width_fixed` are the same metrics under the fixed
  train-mean membership, the primary cluster protocol, and `fixed_source` says
  where they came from: `log` (the run logged `sl_test_fixed/*` itself, and
  that wins), `sidecar` (the `eval_fixed.json` re-score
  `forecasting/ccm_rescore.py` writes beside an older run, used only if its
  self-check passed and the checkpoint it hashed is still the run's
  `supervised.pth`), `k1` (neither, at k = 1, where one cluster leaves nothing
  to fix and the batch values are the fixed ones by construction) or `none`.
  `fixed_code_version` is the fingerprint of the code that computed them — the
  run's own for `log` and `k1`, the re-score's commit for a `sidecar` — and
  `MANIFEST.json` totals them under `fixed_code_versions`, so a sidecar written
  later by other re-score code shows up as a second version.
  Other setups leave these blank. A sidecar is a CPU re-evaluation, `log` and
  `k1` the training run's own on its GPU, and the same checkpoint scored both
  ways differs by up to 2.6e-5 relative in `val_loss_fixed` on EMF (1.7e-4 on
  VoIP PatchTST) — as much as the gap between the best and the next k of an
  EMF DLinear sweep. So a k chosen on `val_loss_fixed` may only rank runs of
  one origin: `runlogs.comparable_fixed` says whether a sweep's
  `fixed_source` values allow it (all 22 live sweeps do today). A cell's fixed means are blank unless every
  run of it has one, and its fixed-protocol table cells print `-` then too;
  `fixed_sources` lists what went in.
- **`mse_std` / `mae_std` are the seed standard deviation in every setup.** For
  a per-channel cell that is the sd of its per-seed band means, over the seeds
  that have every band. Until 2026-09-24 it was the spread over every band x
  seed run, which measures how different the bands are (ED2022 DLinear 0.494
  against a seed sd of 1.3e-4); that spread is kept as `mse_band_seed_sd` /
  `mae_band_seed_sd`. The per-channel CSV twins in `tables/csv/` publish the
  column, so they change meaning with it; no rendered table used it. A seed is
  complete when it has exactly the bands every other seed of the cell has, one
  run each (`frames.complete_seeds`, the one rule the paradigm comparison and
  `timing.csv` use too); `n_complete_seeds` is the sd's sample size. `mse`,
  `mae` and `n_seeds` still count every run, a seed still training included,
  as the published tables always have -- but no table passes such a cell off
  as complete: one with fewer complete seeds than the other cells of its
  dataset and window have (`extract_results.shortfall`) is printed with
  `(n=…)` and left out of every bold, underline, win count and IMP mean in
  the per-setup sections, CCM vs Base, the Ultimate Comparison and MV vs
  univariate.
- **`bands.csv`**: one row per run and band, over the joint-channel corpora —
  a multivariate or cluster run's `sl_test/mse_per_channel` (and its fixed
  twin), a per-channel run's own band. `channel` is the loader's column index,
  the same in every paradigm; `band` is that column's header (the frequency in
  MHz, or the VoIP signal name). Joint runs log no per-band MAE, so it is blank
  there. `lookback_mse` is the band's lookback-mean baseline. The per-band
  values are the logged float32 vectors, so they average to their run's
  number only approximately: a batch vector to `mse` within 2.4e-7 relative,
  a sidecar's fixed vector to `mse_fixed` within 1.6e-5 (EMF2023; 2.3e-6 on
  UH2022, under 7e-7 elsewhere).
- **`timing.csv`**: the uncontended cost of each complete per-channel seed at
  L336/O96: every band's epochs (`runs.csv` now carries `epochs`) times the
  seconds per epoch the solo timing pass of 2026-09-22 measured on three bands
  trained alone (`specs/timing.yaml`, copied from the git-ignored
  `.personal/timing/`), with the fastest and slowest measured band as the
  bracket, beside the logged contended sum and the multivariate run of the same
  seed. EMF only: VoIP had no solo pass and shows no measurable contention.

## The paradigm comparison

`tables/paradigms.md` (and, for VoIP, `paradigms-l24.md` at the second window)
is the seed-aware answer to which paradigm forecasts best, and
`tables/ccm-sweep.md` shows every cluster sweep of the family behind it. Both
come from `emfres/paradigms.py`, so they cannot name different best k's; its
docstring has the full rules.

- **Fixed membership is the primary cluster protocol.** Cluster cells read
  `mse_fixed` / `mae_fixed`, and the best k is the lowest seed-mean
  `val_loss_fixed` of the cell's sweep, ranked only over k values with every
  seed of the sweep and only when the ranked runs' fixed metrics share one
  origin (`runlogs.comparable_fixed`). The logged batch protocol, k chosen on
  the logged `val_loss`, is a sensitivity row and a per-cell table.
- **Counted cells are complete.** A comparison counts a cell only when both
  sides have every seed of the family and the best k comes from a finished
  sweep; anything else is shown with its n and counted as *pending*. ARIMA
  and Persistence are the same computation in the multivariate and
  per-channel grids: shown once (n = 1), *identical*, left out of every count
  and test, and n/a for clustering.
- **Median gap and test.** The median gap is over every counted cell. A cell
  whose two sides differ by less than 5e-5 in the metric itself (an absolute
  floor, the tie band's, not a relative gap) is a zero difference: it stays
  in the count and the median but leaves the Wilcoxon test, as the test's
  default drops zeros, so `n (test)` can be smaller than `Cells`.
- **Whether the Transformer trains.** A Transformer seed trains when its skill
  is at least half the best multivariate skill of the other models on the
  dataset; the tables give, per pipeline, how many of its seeds train (n/N), a
  descriptive row over the seeds whose multivariate run trains (EMF: 69421),
  and flag a cell where no seed of any paradigm trains (the VoIP Transformer),
  with the scoreboard repeated without it.
- **Sensitivity rows** re-count the comparisons on the ratio-of-sums skill,
  without the two near-unforecastable bands (UH2022 ch21, EMF2023 ch24),
  without each cell's top band (its largest multivariate band MSE; this is the
  row that moves the EMF per-channel-vs-multivariate MSE Wilcoxon p, 0.018 to
  0.090 on the 2026-09-24 logs, where the named two move it to 0.024), and
  without the failed per-channel bands (`best_epoch` <= 2), both sides
  recomputed from `bands.csv`.
- The cluster sweep marks the validation k and prints the test-best k as an
  *oracle* when they differ; per-channel times everywhere are labelled per
  band and contended.
- **The per-setup and cross tables lead with the fixed protocol too.** The
  Cluster section prints `sl_test_fixed/*` in its accuracy and conformal
  tables. The logged batch protocol is its last group, `Batch protocol
  (sensitivity)`. CCM vs Base's + CCM column is the fixed cell at the k
  chosen on `sl_val_fixed/loss`, the paradigm comparison's own choice
  (`tables.validation_choices`), so the three tables cannot name different
  k's for a cell. Table 1.2 repeats it under the batch protocol with that
  protocol's own validation k, and the test-best k appears only in a
  configuration row labelled as an oracle. The Ultimate Comparison's cluster
  rows are the fixed protocol, and the batch ones follow the time table
  (1.4, 1.5; 1.3 stays the time table). `extract_results.render_setup` and
  `protocol_cells` do the split for the frames path and the logs walk alike.
  The legacy plotters' records (`emfres/records.py`) read a cluster cell's
  fixed twins as well, so a regenerated figure plots what the table prints.
  A fixed twin is all or nothing in the tables as in `cells.csv`
  (`extract_results.fixed_cells`): a cell with a run that has no fixed
  metrics prints `-` there, not a mean over its other seeds, while its time,
  epochs and size keep every run (no run lacks them today). The csv twin
  `tables/csv/cluster.csv` keeps `cells.csv`'s columns, so its primary values
  are the `*_fixed` ones and `mse`/`mae`/`ic`/`jc`/`interval_width` the batch
  sensitivity; `tables/index.md` says so beside its link. CCM vs Base bolds
  neither side of a pair that prints the same at 4 decimals.

## Which runs count

`specs/runs.yaml` declares what `extract` drops and why. It still excludes the
54 cluster runs stamped seed 42 / seed 43 — a different configuration, not
replicate seeds (`notes/ccm-hyperparameters.md` §8) — though since 2026-08-25
those runs live under `logs/_archive/cluster/` and are skipped as an archive
anyway; the rule stays so they cannot come back silently. `--no-spec` loads
everything.

Cluster publishes **135** EMF runs: the native-CCM k-sweep at L336/O96
(`forecasting/multivariate/models/ccm.py` threading upstream's clustering
layers through the repo's own DLinear / PatchTST / TSMixer), five k × three
datasets × three seeds, rerun under the comparability contract on 2026-09-21.
The single-seed 2026-08-25 sweep it replaced (105 runs) is in
`logs/_archive/2026-09-21_pre-contract/`; the earlier upstream-wrapper sweep and
the 27-run lookback ladder are in `logs/_archive/`; the ladder survives as
`figures/l-ladder/*.png`.

## Known gaps

- The frozen snapshot (`PUBLISHED` in `emfres/cli.py`, currently
  `archive/2026-09-22_tables.md`) is what `make.py tables --check` and
  `make.py reconcile` compare against. It covers only the values published
  when it was frozen; anything added since is counted as new but unguarded,
  which is the cost of checking against a snapshot rather than a second
  implementation. Re-freeze it once the CCM sweep has ended and the
  expected changes (`make.py reconcile --all`) are reviewed: copy
  `tables/all.md` into `archive/`, repoint `PUBLISHED`, and set
  `specs/gate.yaml`'s `snapshot` to the new file with empty `held_elsewhere`
  and `expected` lists. The VoIP tables have no snapshot yet (`SNAPSHOTS` in
  `emfres/cli.py`) and are rendered ungated.
- `specs/provenance.yaml` holds three kinds of evidenced rule. `inferred` dates
  runs logged before stamping (2026-09-19) by setup and `exp/time` window;
  `attested` resolves runs stamped on a dirty tree to the commit whose code they
  ran; `equivalent_code` counts one training-code version as another. The 168
  single-series runs have none and stay unknown. The per-channel runs kept from
  August are a different code version from everything run since
  (the Phase 0/1 changes), so their cells show two versions once newer seeds
  join them; `.personal/PLAN.md` records the bit-exact reproduction that shows
  those runs behave identically under the current code.
- `frames/bands.csv` is 1.19 MB at full precision (10,608 rows on
  2026-09-24), more than half again what `results/` tracks (2.28 MB), and the
  CCM sweep's cluster runs add roughly 5,000 rows more. Whether to track it
  as is, round it, or regenerate it on demand is still open; decide before
  the regenerated frames are next committed. It is neither tracked nor
  ignored today, and `tables/paradigms.md`'s band-level sensitivity rows and
  per-cell band table come from it: `make.py tables` on the tracked frames
  alone renders them out (the section's note says so, and `make.py tables`
  prints a WARNING). Committing it with `runs.csv` is what makes those rows
  reproducible from the tracked frames.
- `frames/epochs.csv` (per-epoch losses) is opt-in behind `--epochs` and
  git-ignored: 11.7 MB of diagnostic curves is not a published number.
- `emfres/fig_curves.py` still walks `logs/` directly rather than reading
  `epochs.csv`, so loss curves are the one family outside the frames pipeline.
  It draws every seed's curve and a mean only over the epochs every seed
  reached, with each seed's own best validation epoch; the NaN-padded mean it
  drew before let a collapsed, early-stopped seed drop out of the average.
