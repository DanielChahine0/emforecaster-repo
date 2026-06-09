# EMForecaster Codebase Guide

*A plain walkthrough of the EMForecaster research framework: what each part does, how a run flows from start to finish, and what each of the ten models is and how it works.*

> Written 2026-06-08 by reading the source code directly. Line references are written as `path:line` and point into the repository as it stands today. Where the code is fragile or surprising, this guide says so instead of hiding it.

## Table of Contents

**Part I. The System**

- [1. What This Project Is](#1-what-this-project-is)
- [2. Repository Layout](#2-repository-layout)
- [3. End to End: How One Run Flows](#3-end-to-end-how-one-run-flows)
- [Entry Point and Experiment Orchestration](#entry-point-and-experiment-orchestration)
- [Configuration System](#configuration-system)
- [Data Pipeline](#data-pipeline)
- [Training Internals (loss, schedulers, init, logging)](#training-internals-loss-schedulers-init-logging)
- [Conformal Prediction and Uncertainty Quantification](#conformal-prediction-and-uncertainty-quantification)
- [Auxiliary Classification Task, Analysis, and HPC Jobs](#auxiliary-classification-task-analysis-and-hpc-jobs)

**Part II. The Model Zoo**

- [Model Comparison Table](#model-comparison-table)
- [EMForecaster](#emforecaster)
- [PatchTST](#patchtst)
- [TimesNet](#timesnet)
- [ModernTCN](#moderntcn)
- [DLinear](#dlinear)
- [TSMixer](#tsmixer)
- [Mamba](#mamba)
- [Recurrent (RNN / LSTM / GRU)](#recurrent-rnn--lstm--gru)
- [Linear](#linear)
- [RF-EMF Baselines (CNN / MLP / Transformer / LSTM)](#rf-emf-baselines-cnn--mlp--transformer--lstm)

**Part III. Operating the Framework**

- [How to Run: Quick Reference](#how-to-run-quick-reference)
- [Caveats, Sharp Edges, and Dead Code](#caveats-sharp-edges-and-dead-code)
- [Glossary](#glossary)

## 1. What This Project Is

**EMForecaster** is the official code release for this paper:

> Mootoo, Xavier, Hina Tabassum, and Luca Chiaraviglio. *"EMForecaster: A deep learning framework for time series forecasting in wireless networks with distribution-free uncertainty quantification."* **IEEE Transactions on Network Science and Engineering (2025).**

It is a PyTorch research framework for **single-variable time-series forecasting**. The main benchmark is **electromagnetic-field (EMF) exposure forecasting** for wireless networks. The repository has two headline contributions, plus a set of supporting models and tooling:

1. **The EMForecaster model.** A lightweight model that cuts the input series into overlapping *patches*, embeds each patch with a small linear-style "patch model" (DLinear or TSMixer), and then forecasts with a single linear layer. The whole thing is wrapped in reversible instance normalization (RevIN). It is deliberately cheap (about 2.4M parameters in the shipped run) and still competes with much heavier Transformers.
2. **A distribution-free conformal-prediction pipeline.** This turns any point forecaster into one that produces calibrated *prediction intervals* with a coverage guarantee. It also proposes a single **trade-off score** that balances narrow intervals (good) against high empirical coverage (good).

Everything else in the repo exists to support, compare against, or analyze those two things. That includes nine more models (Transformers, CNNs, RNNs, a state-space model, MLP-mixers, and linear baselines), a Pydantic-validated YAML config system, a single-device training and evaluation engine, Neptune plus offline logging, an HPC job-submission layer (SLURM and Compute Canada through `submitit`), and analysis scripts (ARIMA and ADF stationarity baselines, plotting, results compilation).

### The data and the task

The forecasting data is **single-variable EMF exposure in V/m**, sampled every **6 minutes** at fixed monitoring stations (`DET_Monitoring`, `PTV_Monitoring`, `PTV_Monitoring2` under `data/forecasting/rf_emf2/`, with 26k to 98k timesteps each). The CSV has two columns: a `TimeStamp DD/MM/YYYY hh:mm:ss` and `Exposure V/m`.

The default supervised task is **long-horizon forecasting**. Given a lookback window of `seq_len = 336` steps (about 33.6 hours) of normalized exposure, predict the next `pred_len = 96` steps (about 9.6 hours). The same framework *also* carries a second **classification** task from a completely different domain: **intracranial-EEG (iEEG) seizure-onset-zone (SOZ) detection** on the OpenNeuro dataset (see the `OpenNeuro` config class at `config.py:536` and the wavelet/shapelet feature layers). Most models have a classification head for it, but it is secondary to the EMF forecasting the paper is about.

## 2. Repository Layout

```text
emforecaster/                         # repo root (also the importable package name)
├── main.py                           # CLI entry point: python main.py <job_name>
├── pyproject.toml / requirements.txt # pinned deps (torch 2.11, numpy 2.2, pydantic 2.x, ...)
├── README.md / PLOT_NOTES.md         # project docs plus notes on the saved test plots
├── data/forecasting/rf_emf2/*.csv    # the EMF exposure datasets (single-variable, 6-min cadence)
├── dataset/*.csv                     # raw monitoring CSVs (DET / PTV, 2022 to 2024)
├── logs/<run>/<seed>/                # per-run outputs: args.yaml, log.json, *.pth, plot_*.png
│
└── emforecaster/                     # the actual Python package
    ├── config/config.py              # Pydantic config tree (Global plus nested sub-models)
    ├── exp/exp.py                    # Experiment class: the train/validate/test engine (~1k lines)
    ├── models/                       # the 10 forecasting/classification models
    │   ├── emforecaster.py           #   flagship: patch + linear-style mixer + RevIN
    │   ├── patchtst.py               #   PatchTST channel-independent Transformer
    │   ├── timesnet.py               #   TimesNet FFT-period 2D-conv model
    │   ├── modern_tcn.py             #   ModernTCN large-kernel depthwise conv
    │   ├── dlinear.py                #   DLinear trend/seasonal decomposition linear
    │   ├── tsmixer.py                #   TSMixer all-MLP time/channel mixing
    │   ├── mamba.py                  #   standalone Mamba SSM (DEAD CODE, see caveats)
    │   ├── recurrent.py              #   RecurrentModel: RNN/LSTM/GRU/Mamba behind one wrapper
    │   ├── linear.py                 #   single linear layer baseline (LTSF-Linear style)
    │   └── rf_emf_baselines.py       #   four tiny baselines: MLP/CNN/LSTM/Transformer
    ├── layers/                       # reusable building blocks
    │   ├── patcher.py                #   cut a series into overlapping patches
    │   ├── patchtst/                 #   backbone, encoder block, positional enc, RevIN
    │   ├── timesnet/                 #   TimesBlock, Inception block, data embedding
    │   ├── modern_tcn/               #   reparam large-kernel conv backbone
    │   ├── dlinear/series_decomp.py  #   moving-average trend/seasonal split
    │   └── wavelets/                 #   DWT/CWT transforms plus perturbation (classification)
    ├── conformal/                    # the uncertainty-quantification pipeline
    │   ├── coverage.py               #   split conformal: intervals plus empirical coverage
    │   └── tradeoff.py               #   the proposed width-vs-coverage trade-off score
    ├── utils/                        # the plumbing
    │   ├── models.py                 #   get_model factory plus optimizer/loss/scheduler factories
    │   ├── dataloading.py            #   CSV to windowed tensors to DataLoaders (~1k lines)
    │   ├── datasets.py               #   torch Dataset classes
    │   ├── classification.py         #   auxiliary classification task helpers
    │   ├── logger.py                 #   Neptune plus offline JSON logging, plots, checkpoints
    │   ├── schedulers.py             #   LR schedulers (cosine, cosine_warmup, ...)
    │   ├── train.py / weight_init.py / sincos_pos_emb.py / utils.py
    ├── tuning/tune.py                # grid-search wrapper over main.py (ablation.yaml)
    ├── analysis/                     # post-hoc analysis
    │   ├── general/                  #   ARIMA, ADF, plotting, amplitude/lambda plots
    │   └── neptune/                  #   pull and compile run results into results.csv
    └── jobs/                         # experiment plus cluster configs
        ├── exp/<model>/args.yaml     #   one config per model (plus ablation.yaml for sweeps)
        └── hpc/                      #   submitit SLURM / Compute Canada submission
```

## 3. End to End: How One Run Flows

Here is the whole pipeline in one pass. Each step links to its detailed section below.

```text
python main.py rf_emf/emforecaster
        │
        ▼
main.py -> load_config(args.yaml) -> Global pydantic tree (args.exp / args.data / args.sl / args.conf / ...)
        │        (optional --ablation JSON overrides keys in place)
        ▼
for seed in args.exp.seed_list:                       # multi-seed loop
        Experiment(args).run()
                │
                ├─ seed RNGs (torch / numpy / random / cuda)
                ├─ init_logger()      -> logs/<ablation>_<model_id>_<exp.id>/<seed>/   (Neptune or JSON)
                ├─ init_device()      -> MPS | cuda:<gpu> | CPU
                └─ supervised_train()
                        ├─ init_dataloaders()  -> dataloading.get_loaders()  -> train/val/test  (B, C, L) to (B, C, H)
                        ├─ init_model()        -> utils.models.get_model(args)  -- dispatch on exp.model_id
                        ├─ init_optimizer()    -> Adam/AdamW (plus a no-weight-decay group for norms/bias)
                        ├─ criterion           -> MSE / SmoothL1 / BCE / CE
                        ├─ train()             -> epoch loop, early stopping, best ckpt -> supervised.pth
                        └─ test()              -> metrics (MSE/MAE), sample plots, optional conformal
                                                   └─ conformal: calibrate -> intervals -> coverage plus trade-off score
```

The four pieces you touch most often:

- **What runs** is chosen by `exp.model_id` in the YAML (for example `"EMForecaster"`, `"PatchTST"`, `"RecurrentModel"`). `get_model` in `emforecaster/utils/models.py` is a flat `if/elif` that maps that string to a concrete `nn.Module`.
- **What data** is controlled by `args.data` (`seq_len`, `pred_len`, `num_channels`, dataset class, patch geometry).
- **How it trains** is controlled by `args.sl` (optimizer, criterion, epochs, lr, batch size, RevIN flags, scheduler).
- **Whether you get uncertainty** is controlled by `args.conf`. Setting `conf.conf=True` turns on the conformal pipeline, and `conf.alpha=0.1` targets 90% coverage.

The detailed mechanics of each stage follow.

## Entry Point and Experiment Orchestration

This section follows the full path from `python main.py <job_name>` through config loading, the `Experiment` lifecycle, model dispatch, the train/validate/test loop, and the grid-search wrapper. All references are `path:line` into the actual source.

### Launching a run: `main.py`

- You run it as `python main.py <job_name> [--ablation <json>] [--ablation_id <int>] [--venv_test]`. The positional `job_name` (default `"test"`) names a directory under the jobs tree, not a model. The model is chosen inside the YAML through `exp.model_id` (`main.py:94-100`).
- `__main__` parses args. If `--ablation` is set, it `json.loads` the string and forwards it to `main(...)`. An invalid JSON ablation calls `sys.exit(1)` (`main.py:106-114`).
- `main()` loads `.env` (`load_dotenv()`), then builds the config path and loads it (`main.py:60-72`).
- The config path is built in a surprising and fragile way:
  ```python
  base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
  args_path = os.path.join(base_dir, "emforecaster", "emforecaster", "jobs", "exp", job_name, "args.yaml")
  ```
  (`main.py:67-68`). `base_dir` climbs **two** levels above `main.py` (to `.../USRA`), then re-appends a **doubled** `emforecaster/emforecaster/...`. This only works because the repo root folder is itself named `emforecaster`. The real files live at `emforecaster/jobs/exp/<job>/args.yaml`. **This is fragile. Renaming or moving the repo root, or relocating `main.py`, silently breaks config discovery.**
- Ablation injection: when `--ablation` is passed, `update_global_config(ablation, args, ablation_id)` changes the loaded Pydantic config in place. Keys are dotted paths like `global.sl.lr` (3 parts) or `sl.lr` (2 parts), resolved with `getattr`/`setattr`. Unknown keys only print a warning, they do not raise (`main.py:24-58`). It also stamps `args.exp.ablation_id = ablation_id`.

### Config object: `load_config` to `Global`

- `load_config(file_path)` is a thin `yaml.safe_load` plus `Global(**config_data)`, so the entire config is a validated Pydantic tree (`emforecaster/config/config.py:738-743`).
- `Global` collects sub-models accessed throughout as `args.exp`, `args.data`, `args.sl`, `args.conf`, `args.early_stopping`, plus per-model blocks `args.patchtst`, `args.dlinear`, `args.tsmixer`, `args.timesnet`, `args.moderntcn` (`config.py:719-735`).
- Key defaults the orchestration reads: `exp.model_id="PatchTST"`, `exp.best_model_metric="loss"`, `exp.tuning_metric="loss"`, `exp.seed_list=[2024]`, `sl.optimizer="adam"`, `sl.criterion="MSE"`, `sl.epochs=100`, `sl.lr=1e-4`, `sl.batch_size=64`, `conf.conf=False`, `conf.alpha=0.1`, `conf.validation_eval=False`.

### Multi-seed loop

- `main()` loops over `args.exp.seed_list`, setting `args.exp.seed` and a fresh `args.exp.time` timestamp per seed. It pretty-prints the config with `rich.pprint`, builds `Experiment(args)`, and calls `exp.run()` once per seed (`main.py:74-89`). Each seed is a fully independent run with its own logger and log directory.

### The `Experiment` lifecycle: `run()`

`Experiment` lives in `emforecaster/exp/exp.py`. `run()` (`exp.py:75-103`) is the orchestrator:

1. **Reproducibility.** It seeds a `torch.Generator` plus `torch`, `numpy`, `random`, and (if available) CUDA RNGs (`exp.py:80-86`). The generator is passed into the dataloaders and the model factory.
2. **`init_logger()`.** It builds the log directory `logs/{ablation_id}_{model_id}_{exp.id}/{seed}/`, dumps the resolved config back out with `global_to_yaml(...)` to `args.yaml`, and either starts a Neptune run (when `exp.neptune`) or an in-memory `dict` logger that is JSON-dumped later (`exp.py:220-259`). When `conf.conf` is on, it sets `self.score_path = log_dir/scores` for conformal calibration tensors.
3. **`init_device()`.** It picks MPS (if `exp.mps`), else `cuda:{exp.gpu_id}`, else CPU (`exp.py:105-120`).
4. **`supervised_train()`.** The actual driver (see below).
5. **Logger teardown.** `self.logger.stop()` for Neptune, else `json.dump` to `log.json` (`exp.py:97-103`).

Note: despite distributed-sounding helpers (`print_master` "prints to the rank 0 node", comments about "all_reduce" in `evaluate`), **this is single-device only**. `main.py:88` logs "Using single device". There is no `DistributedDataParallel`, process group, or real `dist.all_reduce`. The `all_reduce` mention at `exp.py:843` is a leftover log string over a plain `total_loss.item()`.

### Driver: `supervised_train()`

`supervised_train()` (`exp.py:626-696`) wires everything together and runs train then test:

- `init_dataloaders(loader_type="all", learning_type="sl")` to `seq_load(...)` to `get_loaders(args, learning_type, generator, args.sl.dataset_class, loader_type)`, which fills `self.train_loader/val_loader/test_loader` (`exp.py:122-180`). If `data.median_seq_len`, the seq length is overwritten with the loader-reported median (`exp.py:177-180`).
- `init_model()` to `get_model(...)`, moves to device, logs the param count with `count_parameters()` (`exp.py:195-211`).
- `init_optimizer()` to `get_optim(args, model, args.sl.optimizer)` (`exp.py:213-218`).
- `self.criterion = get_criterion(args, args.sl.criterion)` (`exp.py:642`).
- The scheduler is built only if `args.sl.scheduler != "None"` through `get_scheduler(...)`, else `None` (`exp.py:646-655`).
- The best-model checkpoint path is `log_dir/supervised.pth` (`exp.py:659`). Training runs through `self.train(...)`, then (optionally) the checkpoint is uploaded to Neptune, then `self.test(...)` runs (`exp.py:660-696`).

### Model factory: `get_model` dispatch

`get_model(args, generator)` in `emforecaster/utils/models.py:41-227` is a flat `if/elif` chain keyed on `args.exp.model_id`, mapping each string to a concrete class:

```text
"PatchTST"          -> PatchTST           (models.py:42)
"RecurrentModel"    -> RecurrentModel     (models.py:64)   # backbone via args.exp.backbone_id (LSTM/RNN/GRU/Mamba)
"Linear"            -> Linear             (models.py:86)
"DLinear"           -> DLinear            (models.py:92)
"EMForecaster"      -> EMForecaster       (models.py:103)  # flagship, takes patch_model_id plus backbone_id
"ModernTCN"         -> ModernTCN          (models.py:125)
"TimesNet"          -> TimesNet           (models.py:147)
"KNN"               -> KNeighborsTimeSeriesClassifier or GridSearchCV  (models.py:167)
"TSMixer"           -> TSMixer            (models.py:185)
"RF_EMF_{MLP,CNN,LSTM,Transformer}" -> baselines  (models.py:199-224)
```

- Anything else raises `ValueError("Please select a valid model_id.")` (`models.py:225-226`).
- The flagship `EMForecaster` is built from `args.exp.patch_model_id`, `args.exp.backbone_id`, patch geometry (`args.data.patch_dim/patch_stride/...`), and RevIN flags (`models.py:103-123`).
- `"KNN"` is the only `model_id` that branches on `args.exp.grid_search` to wrap the estimator in sklearn `GridSearchCV` (`models.py:168-175`). This is sklearn's own cross-validation grid search, which is different from the repo's own grid-search in `tune.py`.

### Optimizer / loss / scheduler factories

- `get_optim` (`models.py:230-245`): maps `"adam"` to `optim.Adam` and `"adamw"` to `optim.AdamW`, and returns `None` for sklearn. It calls `exclude_weight_decay(...)` so that biases and `BatchNorm*/LayerNorm/RevIN` params go into a `weight_decay=0.0` group while everything else uses `args.sl.weight_decay` (`models.py:248-268`).
- `get_criterion` (`models.py:316-328`): `"MSE"` to `nn.MSELoss`, `"SmoothL1"` to `nn.SmoothL1Loss`, `"BCE"` to `nn.BCEWithLogitsLoss`, `"BCE_normal"` to `nn.BCELoss`, `"CE"` to `nn.CrossEntropyLoss`.
- `get_scheduler` (`models.py:271-313`): supports `cosine_warmup` (pretrain), `cosine` (`CosineAnnealingLR`, `T_max=args.sl.epochs`, `eta_min=lr*1e-2`), `patchtst` (`PatchTSTSchedule`), and `onecycle` (`OneCycleLR`).

### Train / validate / test loop

**`train()`** (`exp.py:349-535`):
- For sklearn models it short-circuits to `model.fit` and returns (`exp.py:383-385`).
- Otherwise it loops `range(eval(f"self.args.{flag}.epochs"))`. Note `flag="sl"`, so this reads `args.sl.epochs`. The `eval(...)` on an f-string is an indirection used to share the loop across learning types (`exp.py:400`).
- Per batch: `optimizer.zero_grad()`, then `forward_pass(args, model, batch, model_id, device)` to `compute_loss(...)` to `model_update(model, loss, optimizer, model_id, alpha)`. `forward_pass` pulls `batch[0]` as input `x` and optionally first-differences it when `args.data.difference_input` (padding the last step) (`models.py:331-364`). `compute_loss` squeezes output and target through `prepare_output_and_target` and applies `criterion` (`models.py:395-420`). `model_update` is a plain `loss.backward()` then `optimizer.step()` (`models.py:423-443`). The `alpha`/EMA path only matters for `flag=="ssl"` and is unused here.
- Loss is accumulated as `loss * num_batch_examples` (where `num_batch_examples = batch[0].shape[0]`) and divided by `len(train_loader.dataset)` for the epoch mean (`exp.py:415-451`). Progress prints every 100 batches.
- The scheduler's `.step()` is called **per batch** inside the inner loop (`exp.py:448-449`). This matches the OneCycle and PatchTST schedulers, but it means a `cosine` (`CosineAnnealingLR`, `T_max=epochs`) scheduler gets stepped far more often than `epochs`. **The cosine path has surprising step granularity.**

**`validate()`** (`exp.py:538-624`), called each epoch when a `val_loader` exists:
- Delegates to `evaluate(...)`, reads `stats["loss"]`, and logs validation metrics.
- Chooses `val_metric = val_loss` for `best_model_metric=="loss"`, or `-stats[metric]` for `acc`/`ch_acc`/`ch_f1` (negated so "lower is better" holds the same way for all) (`exp.py:582-589`).
- Checkpointing: if `early_stopping` is on, it defers entirely to `self.early_stopping(val_metric, model)`. Otherwise it manually compares against `self.best_val_metric` and calls `torch.save(model.state_dict(), best_model_path)` on improvement (`exp.py:591-620`). When `conf.conf and conf.validation_eval`, conformal `calibration_scores` are also saved to `score_path` alongside the best model.

**Conformal calibration** is interleaved in `train()`. When `conformal=True`, `evaluate(..., conformal_calibrate=True)` runs on the validation set either every epoch (`conf.validation_eval`) or only on the last epoch or at early-stop (`exp.py:458-524`).

**`evaluate()`** (`exp.py:764-903`): a single `model.eval()` plus `torch.no_grad()` pass that collects `all_logits/all_labels`, computes mean loss, optional MAE, window and channel metrics, optional plots, and (when `coverage=True`) conformal independent and joint coverage (`stats["ic"]`, `stats["jc"]`) plus interval widths through `self.coverage(...)`. Coverage tensors are `(n_samples, pred_len, num_channels)`, and 2-D preds/targets are unsqueezed to add the channel axis (`exp.py:925-997`).

**`test()`** (`exp.py:698-762`): reloads the best checkpoint with `torch.load(best_model_path)` plus `load_state_dict`, runs `evaluate(..., coverage=conf.conf, plot_results=True, num_plots=10)`, logs test metrics, and stores `self.tuning_score = stats[args.exp.tuning_metric]` for the grid-search wrapper (`exp.py:735-762`). Test data loading depends on flags: `calibrate` reloads `"all"` loaders, otherwise `"test"` only (`exp.py:711-716`).

### Early stopping and checkpoints

- `EarlyStopping` (`emforecaster/utils/train.py`) is built in `init_earlystopping(path)` with `patience/verbose/delta` from `args.early_stopping` and `path=best_model_path` (`exp.py:261-267`). Its `__call__` increments a counter when `val_loss > best_val_loss` and flips `early_stop` once `counter >= patience`. On improvement it `save_checkpoint`s the `state_dict` and resets the counter. **Surprising: it ignores the `delta` constructor arg entirely (no `delta` in the comparison) and has a typo'd attribute `self.verbos`.**
- Two checkpoint code paths exist: the manual save in `validate()` (when early stopping is off) and `EarlyStopping.save_checkpoint` (when on). Both write the same `log_dir/supervised.pth`, which `test()` reloads.

### Grid-search / tuning wrapper: `tune.py`

`emforecaster/tuning/tune.py` wraps the whole `main()` flow for hyperparameter sweeps:

- It inserts the project root on `sys.path` and imports `from main import main` (`tune.py:7-13`).
- You run it as `python tune.py <ablation_job_name> [--cc] [--acc <account>]` (`tune.py:83-95`).
- `run_grid_search(job_name, cc, acc)` reads two files per job: `jobs/exp/<job>/args.yaml` (base config) and `jobs/exp/<job>/ablation.yaml` (the sweep) (`tune.py:36-45`). **Note that these use the single, correct `emforecaster/jobs/exp/...` layout through `project_root`, unlike `main.py`'s doubled path.**
- `load_ablation_config` returns the `ablations` dict and an optional `cc` (Compute Canada) base block (`tune.py:20-25`).
- `generate_ablation_combinations` takes the **full Cartesian product** of all ablation lists through `itertools.product`, yielding one dict per cell (`tune.py:28-32`). This is a true grid search, so the number of combinations grows multiplicatively.
- For each combination, indexed by `i` (the `ablation_id`):
  - **Local mode** (default): calls `main(job_name=job_name, ablation=ablation, ablation_id=i)` directly in-process, re-entering `main.py`, which re-loads `args.yaml` and applies the ablation overrides (`tune.py:78-80`).
  - **Compute Canada mode** (`--cc`): JSON-encodes the ablation, `shlex.quote`s it, builds a CLI string `"<job> --ablation <json> --ablation_id <i>"`, and submits a SLURM job through `submit_job(...)` using `load_configs(cc_base)` (with optional `--acc` account override) (`tune.py:52-77`). Each grid cell becomes its own cluster job.

There is no built-in best-config selection or aggregation in `tune.py`. It fans out runs. The per-run `self.tuning_score` (set in `test()`) and the per-cell `ablation_id`-tagged `logs/` directories are the artifacts a downstream comparison would read.

## Configuration System

The configuration system lives in one file, `emforecaster/config/config.py`, and is built on **pydantic v1-style `BaseModel`** classes. Every experiment is fully described by one `Global` object: a flat container of typed sub-models, one per concern (experiment, data, supervised-learning, conformal, and per-architecture hyperparameter blocks). YAML files under `emforecaster/jobs/exp/.../args.yaml` are parsed and validated into a `Global` instance, then optionally overridden by ablation grids.

### Top-level object: `Global`

`Global` (`emforecaster/config/config.py:719`) is the root. Each field is a nested sub-model that is **default-instantiated** (for example `exp: Experiment = Experiment()`), so a bare `Global()` is always valid with all defaults. The full set of categories:

```python
class Global(BaseModel):
    exp: Experiment
    data: Data
    sl: SL
    ddp: DDP
    early_stopping: EarlyStopping
    ema: EMA
    scheduler: Scheduler
    open_neuro: OpenNeuro
    mlp_mixer: MLPMixer
    patchtst: PatchTST
    dlinear: DLinear
    tsmixer: TSMixer
    timesnet: TimesNet
    moderntcn: ModernTCN
    conf: Conformal
```

The YAML top-level keys map 1:1 to these field names (`exp:`, `data:`, `sl:`, `tsmixer:`, `dlinear:`, `ddp:`, `scheduler:`, `early_stopping:`, `conf:`). Throughout the codebase the loaded object is referenced as `args`, and fields are read by dotted path, for example `args.exp.model_id`, `args.data.seq_len`, `args.sl.lr` (see `emforecaster/exp/exp.py:80`, `emforecaster/utils/models.py:42`).

### How YAML becomes a validated `Global`

`load_config(file_path)` (`config.py:738`) is the entry point. It `yaml.safe_load`s the file into a dict and splats it into the constructor, `return Global(**config_data)`. Pydantic then validates and coerces types per field, filling any omitted keys or sub-models with defaults.

`main()` (`/Users/danielchahine/Desktop/Programs/USRA/emforecaster/main.py:60`) drives the real flow:

```python
args_path = os.path.join(base_dir, "emforecaster", "emforecaster",
                         "jobs", "exp", job_name, "args.yaml")
args = load_config(args_path)
if ablation is not None:
    args = update_global_config(ablation, args, ablation_id)
```

- The path is hardcoded as `.../jobs/exp/<job_name>/args.yaml`. `job_name` is a CLI positional arg (for example `rf_emf/emforecaster`). **This is fragile.** The doubled `"emforecaster", "emforecaster"` path segment is literal, so the file must sit at that exact nesting.
- `args.exp.seed_list` drives an outer loop in `main()` that re-runs the experiment once per seed, overwriting `args.exp.seed` and `args.exp.time` each time (`main.py:75-89`).

### Ablation / hyperparameter override mechanism

Two YAML files live together in each job folder: `args.yaml` (the base config) and `ablation.yaml` (a grid). `ablation.yaml` has a single `ablations:` block mapping **dotted keys to lists of values**:

```yaml
ablations:
  data.pred_len: [192, 336, 512]
  tsmixer.d_model: [128, 256, 512]
  conf.alpha: [0.05, 0.01]
```

`tuning/tune.py` expands this Cartesian product with `itertools.product` in `generate_ablation_combinations` (`tune.py:28-32`), producing one flat dict per run (for example `{"data.pred_len": 192, ...}`), JSON-encodes it, and passes it to `main.py` through `--ablation`. `update_global_config` (`main.py:24`) then applies each `sub_model.param` key with `getattr`/`setattr`:

```python
sub_model, param = key.split('.')      # e.g. "sl", "lr"
sub_config = getattr(global_config, sub_model)
setattr(sub_config, param, value)
```

It supports both 2-part keys (`sl.lr`) and 3-part `global.sub.param` keys, and finally stamps `global_config.exp.ablation_id = ablation_id`. **Surprising:** unknown keys are silently warned and skipped, not errored, so a typo'd ablation key is a no-op. Also note that `setattr` **bypasses pydantic validation** (these models do not enable `validate_assignment`), so overrides are not type-checked.

### How the config selects model, dataset, and task

Three independent string fields act as switches:

- **Model.** `args.exp.model_id`. `get_model(args, ...)` (`emforecaster/utils/models.py:41`) is a long `if/elif` chain on it: `"PatchTST"`, `"RecurrentModel"`, `"Linear"`, `"DLinear"`, `"EMForecaster"`, `"ModernTCN"`, `"TimesNet"`, `"KNN"`, `"TSMixer"`, plus baselines `"RF_EMF_MLP/CNN/LSTM/Transformer"`. Each branch pulls hyperparameters from the matching sub-model. For example the `EMForecaster` branch (`models.py:103`) reads `args.exp.patch_model_id`, `args.exp.backbone_id`, `args.sl.patch_norm/patch_act/patch_embed_dim/independent_patching/pos_enc`, and `args.data.patch_dim/patch_stride`.
- **Two extra switches for EMForecaster.** The flagship model is composable: `args.exp.patch_model_id` (default `"DLinear"`, configs use `"TSMixer"`) chooses the patch-embedding network, and `args.exp.backbone_id` (default `"LSTM"`, configs use `"Linear"`) chooses the temporal backbone. So EMForecaster's behavior is set by a three-way `(model_id="EMForecaster", patch_model_id, backbone_id)` combination, not a single field.
- **Dataset.** `args.data.dataset` (default `"electricity"`, with RF-EMF variants `"rf_emf"`, `"rf_emf_det"`, `"rf_emf_ptv"`, `"rf_emf_tur"`, `"rf_emf_nov"`, and so on). Used in `get_loaders` (`emforecaster/utils/dataloading.py:950`), which branches on the dataset name to call `load_forecasting(dataset_name=args.data.dataset, ...)`.
- **Task / dataset class.** Two separate fields. `args.exp.task` (`"forecasting"` vs `"classification"`) selects the metric and eval path. `args.sl.dataset_class` (`"forecasting"` vs `"univariate_forecasting"`) selects the windowing `Dataset` through `dataset_class = eval(f"args.{flag}.dataset_class")` (`dataloading.py:921-933`). **Fragile:** this uses `eval` on a config-derived string.

### Most important knobs by category

- **`Experiment` (`exp`, `config.py:24`).** `model_id` (default `"PatchTST"`), `backbone_id`, `patch_model_id`, `task` (default `"classification"`), `seed`/`seed_list` (default `[2024]`), `learning_type` (`"sl"`/`"ssl"`), `neptune` (default `False`, so offline JSON logging), `gpu_id`, `mps`, `mae`, `ablation_id`, and conformal-related calibration flags (`calibrate`, `calibration_type`). `id` is a random 10-char string from `generate_random_id` (`config.py:16`). `time` defaults to import-time `datetime.now()`.
- **`Data` (`data`, `config.py:151`).** `seq_len` (default `512`, RF-EMF configs use `336`), `pred_len` (default `96`), `num_channels` (default `321`, RF-EMF single-variable uses `1`), `patch_dim`/`patch_stride` (defaults `16`/`8`, configs use `8`/`-1` where `-1` means stride equals patch), `train_split`/`val_split` (defaults `0.6`/`0.2`), `scale`, `univariate`, `seq_load`, `dtype` (`"float32"`). Tensors are channel-windowed `(batch, num_channels, seq_len)` inputs forecasting `(batch, num_channels, pred_len)`.
- **`SL` (`sl`, `config.py:306`).** The largest block. It holds training settings plus most model dims. `optimizer` (`"adam"`/`"adamw"`), `criterion` (default `"MSE"`), `lr` (default `1e-4`), `epochs` (default `100`), `batch_size` (default `64`), `scheduler` (default `None`, configs use `"onecycle"`), `weight_decay`, `early_stopping`, `d_model`/`d_ff`/`num_heads`/`num_enc_layers`, RevIN flags (`revin`/`revout`/`revin_affine`/`eps_revin`, all default `True`/`1e-5`), and EMForecaster-specific `patch_norm`/`patch_act`/`patch_embed_dim`/`independent_patching`/`pos_enc`. **Note:** generic dims (`d_model`, `num_enc_layers`, `dropout`) live in `sl`, but the per-model blocks (`patchtst`, `tsmixer`, `timesnet`, `moderntcn`, `dlinear`) also define their own `d_model`/`num_enc_layers`. Which one wins depends on what each `get_model` branch reads. For example TSMixer reads `args.tsmixer.d_model`, PatchTST reads `args.patchtst.*`, but the EMForecaster and Linear branches read `args.sl.*`.
- **`Conformal` (`conf`, `config.py:283`).** The uncertainty pipeline switch. `conf` (default `False`, enables conformal prediction), `alpha` (default `0.1`, the miscoverage rate), `corrected` (Bonferroni correction for multi-horizon intervals), `intervals` (report width mean/std), `validation_eval`.
- **Per-architecture blocks.** `PatchTST`, `DLinear` (`moving_avg`, `individual`, plus EMForecaster's `final_moving_avg`), `TSMixer` (`num_enc_layers`, `d_model`, `dropout`), `TimesNet` (`top_k`, `num_kernels`, `c_out`), `ModernTCN` (list-valued `num_enc_layers`/`d_model`/`large_size` for multi-stage), `MLPMixer`. These are read only when the matching `model_id` or `patch_model_id` is active.
- **Infra blocks.** `DDP` (`ddp`, `master_port`, `slurm`, `find_unused_parameters`), `Scheduler` (`lradj` default `"type3"`, `pct_start` for OneCycle, warmup/cosine params), `EarlyStopping` (`patience` default `10`), `EMA`, and `OpenNeuro` (an unrelated SOZ-localization/classification dataset block carried over from a sibling project, mostly irrelevant to RF-EMF forecasting).

### Practical notes for a new engineer

- A real RF-EMF forecasting run (`jobs/exp/rf_emf/emforecaster/args.yaml`) sets `exp.model_id="EMForecaster"`, `exp.patch_model_id="TSMixer"`, `exp.backbone_id="Linear"`, `data.dataset="rf_emf_det"`, `data.univariate=True`, `sl.dataset_class="univariate_forecasting"`, `seq_len=336`, `pred_len=96`. All three switches plus the univariate dataset class must agree.
- Defaults in `config.py` are tuned for the legacy `electricity`/`classification` setup (for example `num_channels=321`, `task="classification"`), so always start from a job `args.yaml` rather than a bare `Global()`.
- To sweep, edit `ablation.yaml`'s `ablations:` lists. The run count is the full Cartesian product of all lists, so adding a key multiplies the job count.

## Data Pipeline

The forecasting data path runs entirely through `emforecaster/utils/dataloading.py` and `emforecaster/utils/datasets.py`. The flagship EMForecaster model consumes the resulting `DataLoader` batches. **Patching** happens *inside* the model (`emforecaster/layers/patcher.py`), not in the data layer. The **wavelet** files (`emforecaster/layers/wavelets/*.py`) are dead scratch scripts that nothing imports.

### Entry point: `get_loaders` to `load_forecasting` to `get_loader`

The orchestration lives in three functions in `emforecaster/utils/dataloading.py`:

- `get_loaders(args, flag, ...)` (`dataloading.py:950`): checks the dataset name against a hardcoded whitelist (`dataloading.py:958`), calls `load_forecasting`, then wraps each split through `get_loader`. It returns `(train, val)` for `loader_type="train"`, `(test,)` for `"test"`, or all three for `"all"`.
- `load_forecasting(...)` (`dataloading.py:40`): reads the raw file, splits, scales, and (optionally) windows. Returns channel-major arrays `train_data.T, val_data.T, test_data.T`, each shaped `(num_channels, split_len)`.
- `get_loader(args, data, ...)` (`dataloading.py:905`): wraps an array in a `Dataset` and a `DataLoader`.

All knobs come from the `Data` pydantic block in `emforecaster/config/config.py:151` (passed as `args.data`). Key defaults: `seq_len=512`, `pred_len=96`, `window_stride=1`, `scale=True`, `train_split=0.6`, `val_split=0.2`, `dtype="float32"`, `clip=False`, `clip_thresh=0.8`, `univariate=False`, `datetime=False`.

> Fragile: `get_loaders` whitelists `rf_emf_det`, `rf_emf_ptv`, `rf_emf_ptv2`, `rf_emf_tur` but **leaves out** `rf_emf_det2` and `rf_emf_nov`, even though `load_forecasting` handles them. Those names raise `Invalid dataset name` before they ever reach the reader.

### Reading the EMF CSVs (DET_Monitoring, PTV_Monitoring)

Both live under `data/forecasting/rf_emf2/` and have the header `TimeStamp DD/MM/YYYY hh:mm:ss, Exposure V/m`, that is, a datetime column and a single EMF column in scientific notation (for example `3.719413e-01`). Row counts: DET about 40,924, PTV about 26,091, PTV2 about 98,209 samples (6-minute sampling).

- DET branch (`dataloading.py:95`): `rf_emf_det` to `DET_Monitoring.csv` (2022), `rf_emf_det2` to `DET_Monitoring2.csv` (2023).
- PTV branch (`dataloading.py:107`): `rf_emf_ptv` to `PTV_Monitoring.csv` (2022), `rf_emf_ptv2` to `PTV_Monitoring2.csv` (2023). PTV2 hardcodes `cutoff = 85251` because everything after it is NaN (`dataloading.py:117`).

The exposure column (column index `1`) is the signal:

```python
data[:, 1] = clip_and_interpolate(data[:, 1], thresh) if clip else data[:, 1]
data = data[:, 1].reshape(-1, 1) if not date else data   # dataloading.py:104-105
```

- When `date=False` (default), the result is `(num_samples, 1)`, a single channel, and the datetime is discarded.
- `clip_and_interpolate` (`dataloading.py:527`) replaces any value `> thresh` by linearly interpolating from the nearest in-range neighbours (searching outward in both directions). It is used to kill noisy peaks.
- `average_italy` (`dataloading.py:154`) applies `time_window_average(data, m=5)` (30-min averages) and only matters for the Turin and Novara datasets.

> Surprising: PTV2 forces `data[:, 1].astype(np.float64)` (`dataloading.py:121`) before clip, while DET does not. The DET/PTV CSVs are read with the default comma separator, while the Italian datasets use `sep=";"` (`dataloading.py:135`).

### Splitting and scaling

For non-ETT datasets (all `rf_emf*`), the split is proportional (`dataloading.py:199`):

```python
num_train = int(len(data) * train_split)   # 0.6
num_test  = int(len(data) * test_split)     # 1 - train - val = 0.2
train_idx = [0, num_train]
val_idx   = [num_train - seq_len, num_train + num_val]
test_idx  = [len(data) - num_test - seq_len, len(data)]
```

The validation and test windows are extended back by `seq_len` so the first prediction in each split has a full lookback. Scaling uses a single `StandardScaler` (`dataloading.py:215`) **fit on train only**, then applied to val and test. This is correct and has no leakage. With `date=False` the whole `(split_len, 1)` array is transformed. With `date=True` only column 0 is scaled.

The function returns `.T`, so downstream arrays are channel-major `(num_channels, split_len)`. For DET that is `(1, ~24554)` train.

### Dataset classes and windowing

Two `Dataset` classes in `emforecaster/utils/datasets.py` are selected by `args.<flag>.dataset_class`:

- `ForecastingDataset` (`datasets.py:13`): the default (`dataset_class="forecasting"`). It stores the channel-major tensor `(num_channels, num_time_steps)` and windows **lazily** in `__getitem__`:

```python
input_data  = self.data[:, idx : idx + self.seq_len]                      # (C, seq_len)
target_data = self.data[:, idx+self.seq_len : idx+self.seq_len+self.pred_len]  # (C, pred_len)
```

  `__len__ = num_time_steps - seq_len - pred_len` (`datasets.py:39`), so windows slide with **stride 1** regardless of the `window_stride` config. For DET (`seq_len=512, pred_len=96`) each sample is input `(1, 512)`, target `(1, 96)`.

- `UnivariateForecastingDataset` (`datasets.py:49`): used when `univariate=True`. Here windowing happens **eagerly** in `load_forecasting` (`dataloading.py:238`) through `create_windows`, which uses `torch.as_strided` to cut `seq_len+pred_len` windows at `window_stride` (default 1), splitting each into `[:seq_len]` / `[seq_len:]`. Output `x:(num_windows, seq_len)`, `y:(num_windows, pred_len)`. `get_loader` passes `data[0], data[1]` separately.

`get_loader` (`dataloading.py:926`) `eval`s the class name from config, converts numpy to a `float` tensor, and (unless `dataset_only`) builds the `DataLoader` with `persistent_workers=True`, `prefetch_factor=2`, `pin_memory=True`, `num_workers=4`, `drop_last=False`. Train and val shuffle `True`. Test shuffles per `shuffle_test` (default `False`).

> Surprising: `batch_size`/`dataset_class` are looked up under a per-task `flag` namespace through `eval(f"args.{flag}.batch_size")` (`dataloading.py:921,940`), for example `flag="sl"`.

### Where patching sits (inside the model, not the data layer)

`Patcher` (`emforecaster/layers/patcher.py:5`) is created by the model (`EMForecaster` at `emforecaster/models/emforecaster.py:111`), so a batch leaves the `DataLoader` un-patched as `(B, M, L)` and is patched in `forward`:

- Input `x: (B, M, L)` to output `patches: (B, M, N, P)` where `P=patch_dim` and `N = int((L - patch_dim)/stride) + 2` (`patcher.py:28`).
- It right-pads by repeating the **last timestep** `stride` times (`patcher.py:31-33`), then `x.unfold(dimension=2, size=patch_dim, step=stride)`.

In `EMForecaster.forward` (`emforecaster.py:142`) the flow is: RevIN normalize, then `patcher(x).squeeze()`, then patch model, then activation/norm, then flatten to `(B, 1, num_patches*patch_embed_dim)`, then backbone, then RevIN denorm, giving `(B, 1, pred_len)`. Defaults in the model are `patch_dim=24, patch_stride=12`. A `patch_stride=-1` means `patch_dim//2`, and `-2` means `patch_dim`. EMForecaster is single-variable (`num_channels=1`). `VerticalPatcher` (`patcher.py:49`) exists but is unused by EMForecaster.

> Bug-adjacent: `Patcher.forward` computes `patches_combined = patches.view(B*M, N, P)` (`patcher.py:42`) but **returns only `patches`**, so the combined tensor is dead. `EMForecaster` re-derives the `B*M` reshape itself at `emforecaster.py:156`.

### Wavelet transforms: present but not wired in

The four files under `emforecaster/layers/wavelets/` are **standalone demo/exploration scripts, not modules used by the pipeline**. A `grep` across `emforecaster/` finds zero imports of them outside the directory:

- `transform.py`: a top-level script comparing `pywt.wavedec` vs `ptwt.wavedec` (DWT) on a synthetic sine, then a `pywt.cwt` (CWT) demo, with `plt.show()` calls. Runs on import.
- `dwt.py`: a top-level `pywt`/`ptwt` `wavedec`/`waverec` round-trip demo with a `db32` wavelet.
- `cwt.py`: defines `perform_cwt(signal, freq_range, wavelet="cmor1.5-1.0", sampling_rate=1000)` (`cwt.py:6`) computing scales then `pywt.cwt`, but the module body is a synthetic-EEG plotting demo.

> Incomplete: there is no DWT/CWT preprocessing or wavelet layer in the live EMF forecasting path. Treat these files as research scratch, not pipeline code.

## Training Internals (loss, schedulers, init, logging)

This covers the five `emforecaster/utils/` modules that drive a training run, plus how the `Experiment` orchestrator in `emforecaster/exp/exp.py` and the factories in `emforecaster/utils/models.py` wire them together. The actual training loop lives in `Experiment.train()` (`exp/exp.py:349`), reached through `Experiment.run()` to `supervised_train()` (`exp/exp.py:626`).

### Loss functions

Loss is selected by string in `get_criterion(args, criterion_type)` (`utils/models.py:316`). It is a thin map onto stock `torch.nn` losses. There is no custom loss class:

```python
"MSE"        -> nn.MSELoss()
"SmoothL1"   -> nn.SmoothL1Loss()
"BCE"        -> nn.BCEWithLogitsLoss()
"BCE_normal" -> nn.BCELoss()
"CE"         -> nn.CrossEntropyLoss()
```

- The default is `MSE` (`config/config.py:311`, `SL.criterion`). For the EMF forecasting flagship path this is the loss that is used.
- The criterion is built once in `supervised_train()` (`exp/exp.py:642`) and passed through both `train()` and `evaluate()`.
- Per-batch loss goes through `compute_loss(output, batch, criterion, model_id, args, device)` (`utils/models.py:395`), which first calls `prepare_output_and_target(...)` to align shapes, then `criterion(out, target)`. The `model_id` only gates a whitelist of supported models. **Surprising and fragile:** an unrecognized `model_id` raises `ValueError` rather than defaulting.
- MAE is a **separate** side metric, not the training loss. `evaluate()` (`exp/exp.py:785`) creates its own `mae_loss = nn.L1Loss()` and accumulates `total_mae` only when `args.exp.mae` is true (default `False`, `config/config.py:76`).
- `compute_loss` and the train loop never normalize per-batch loss internally. The loop weights each batch loss by `batch[0].shape[0]` and divides by `len(train_loader.dataset)` to get a true sample-mean `epoch_loss` (`exp/exp.py:416,451`).
- `model_update(model, loss, optimizer, model_id, alpha=0.6)` (`utils/models.py:423`) does a plain `loss.backward()` then `optimizer.step()`. There is no gradient clipping (the `check_gradients` helper at `utils/models.py:446` is commented out at the call site). The `alpha` EMA argument is accepted but unused for these supervised `model_id`s. The loop passes `0.966` as a hardcoded default when no EMA iterator exists (`exp/exp.py:421`), which is dead in supervised mode.

### LR schedulers

Schedulers are built in `get_scheduler(args, scheduler_type, training_mode, optimizer, num_batches)` (`utils/models.py:271`). The default `SL.scheduler` is `None` (`config/config.py:356`), so by default **no scheduler runs**. Supported `scheduler_type` values for `training_mode="supervised"`:

- `"cosine"` to stock `CosineAnnealingLR` with `T_max=args.sl.epochs`, `eta_min=args.sl.lr*1e-2` (`utils/models.py:293`). Stepped per batch.
- `"onecycle"` to stock `OneCycleLR(steps_per_epoch=num_batches, pct_start=args.scheduler.pct_start, epochs=args.sl.epochs, max_lr=args.sl.lr)` (`utils/models.py:302`).
- `"patchtst"` to the custom `PatchTSTSchedule` (`utils/schedulers.py:8`).
- `"cosine_warmup"` is only valid for `training_mode="pretrain"` to the custom `WarmupCosineSchedule`, and is not reachable from the supervised path.

Custom schedulers (`utils/schedulers.py`, adapted from facebookresearch/ijepa):

- **`PatchTSTSchedule`** (`schedulers.py:8`) is a dispatcher keyed on `args.scheduler.lradj` (default `"type3"`, `config/config.py:530`):
  - `"type3"`: constant `args.sl.lr` for the first 3 epochs, then decays by `0.9 ** ((epoch-3)//1)` each epoch, applied at every `.step()` but driven by an internal `self.epoch` counter that increments once per `.step()` call. Fragile: `.step()` is called per batch in the loop (`exp/exp.py:448`), so `self.epoch` counts **batches, not epochs**, and the "first 3 epochs" really means the first 3 batches.
  - `"TST"`: wraps an internal `OneCycleLR` (built in `__init__` only for `lradj=="TST"`, `schedulers.py:15`) and copies `get_last_lr()` into each param group.
  - Any other `lradj` raises `ValueError`.
- **`WarmupCosineSchedule`** (`schedulers.py:44`): linear warmup from `start_lr` to `ref_lr` over `warmup_steps`, then cosine decay `ref_lr` to `final_lr` over `T_max - warmup_steps`. Step-based through `self._step`. Defaults from the `Scheduler` config (`config/config.py:498`): `warmup_steps=15`, `start_lr=1e-4`, `ref_lr=1e-3`, `final_lr=1e-6`, `T_max=100`.
- **`CosineWDSchedule`** (`schedulers.py:86`): a cosine **weight-decay** annealer that respects a per-group `WD_exclude` flag. Defined but not referenced by `get_scheduler`, so it is currently unused in the supervised path.

Stepping cadence: `scheduler.step()` is called inside the inner batch loop (`exp/exp.py:448`), that is, **once per batch**. This matches `OneCycleLR`/`cosine` but mismatches the epoch-oriented `type3` logic noted above.

### Optimizer plus weight-decay grouping

This is not in the five files but is essential context. `get_optim` (`utils/models.py:230`) supports `adam`/`adamw` (default `adam`, `SL.optimizer`), and `exclude_weight_decay` (`utils/models.py:248`) splits params into two groups. Biases and `BatchNorm1d/2d`/`LayerNorm`/`RevIN` params get `weight_decay=0.0`, and everything else gets `args.sl.weight_decay` (default `1e-6`).

### Weight initialization

`utils/weight_init.py` provides the init strategy applied by every model through `self.apply(...)`:

- `xavier_init(m, seed=None)` (`weight_init.py:48`):
  - `nn.Linear`: `xavier_uniform_` weight, zero bias.
  - `nn.LayerNorm`: weight 1, bias 0.
  - `nn.LSTM`: `xavier_uniform_` for `weight_ih`, `orthogonal_` for `weight_hh`, zero biases.
  - Surprising: there is no branch for `Conv` layers, so conv-based models (for example `RF_EMF_CNN`, `ModernTCN`) keep PyTorch's default conv init.
- `set_seed(seed)` (`weight_init.py:6`) seeds `torch`, `random`, and (if available) CUDA. If `xavier_init` is called with a non-None `seed`, it re-seeds **on every module visited**. `EMForecaster` does exactly this: `self.apply(lambda m: xavier_init(m, seed=seed))` (`models/emforecaster.py:140`), so the global seed is reset once per submodule. Other models call it without a seed: `self.apply(xavier_init)` in `tsmixer.py:90`, `mamba.py:76`, `recurrent.py:163`, `patchtst.py:77`.
- `trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.)` (`weight_init.py:44`) and its `_no_grad_trunc_normal_` backend are a vendored PyTorch truncated-normal init, present for ViT-style code but **not referenced by `xavier_init`** in this path.
- Run-level reproducibility is handled separately in `Experiment.run()` (`exp/exp.py:80`): a `torch.Generator().manual_seed(args.exp.seed)` plus manual seeding of torch/np/random/CUDA.

### Early stopping

`EarlyStopping` (`utils/train.py:7`, adapted from Bjarten/early-stopping-pytorch) is built by `Experiment.init_earlystopping(path)` (`exp/exp.py:261`) with config from the `EarlyStopping` Pydantic model (`config/config.py:480`): `patience=10`, `verbose=True`, `delta=0.0`.

- Mechanics (`train.py:33`): on each call with `(val_loss, model)`, if `val_loss > self.best_val_loss` it increments `self.counter` and trips `self.early_stop=True` once `counter >= patience`. Otherwise it calls `save_checkpoint` (which writes `torch.save(model.state_dict(), self.path)` and resets `counter=0`).
- Surprising and fragile details:
  - The constructor accepts `delta` but **`delta` is never used** in `__call__`. There is no min-improvement threshold despite the config field.
  - The improvement test is strict `>` (a tie does not increment the counter and triggers a re-save).
  - The "metric" passed in is `val_metric`, not raw loss. `validate()` (`exp/exp.py:582`) sets `val_metric = val_loss` for `best_model_metric=="loss"` (default, `config/config.py:87`) or `-stats[metric]` for `acc`/`ch_acc`/`ch_f1`, so a single "lower is better" convention works for both.
- Two distinct checkpoint paths exist depending on `args.sl.early_stopping`:
  - Early-stopping **on**: `EarlyStopping.save_checkpoint` owns the save (`train.py:44`), writing to `best_model_path` (which is `<log_dir>/supervised.pth`, `exp/exp.py:659`).
  - Early-stopping **off**: `validate()` saves manually when `val_metric < self.best_val_metric` (`exp/exp.py:596-610`), and also dumps conformal calibration scores to `self.score_path` when configured.
- Note the two `early_stopping` flags differ in default. `exp.early_stopping` defaults to `True` (`config/config.py:63`), but the supervised loop is gated by `sl.early_stopping`, which defaults to `False` (`config/config.py:366`) and is what `train()` actually checks (`exp/exp.py:672`).

### The Logger: Neptune online vs offline JSON

The mode switch is `args.exp.neptune` (default `False`, `config/config.py:49`). `Experiment.init_logger()` (`exp/exp.py:220`) sets up both branches and always writes `args.yaml` first through `global_to_yaml` (`utils/logger.py:244`):

- Log directory layout: `logs/{ablation_id}_{model_id}_{exp.id}/{seed}/` (`exp/exp.py:226`). Everything for a run lands here: `args.yaml`, `log.json`, `supervised.pth`, `plot_*.png`, and (if conformal) a `scores` file.
- **Online (Neptune):** `self.logger = neptune.init_run(project=args.exp.project_name, api_token=args.exp.api_token, custom_run_id=args.exp.time)` (`exp/exp.py:243`). `args.yaml` is uploaded under the `config` key. The default `project_name` is `"xmootoo/soz-localization"` (`config/config.py:56`), a stale leftover from the classification project. The token comes from `NEPTUNE_API_TOKEN`.
- **Offline (default):** `self.logger = dict()` and `self.log_file = <log_dir>/log.json` (`exp/exp.py:251`). The dict is dumped to disk **only at the very end** of `run()` (`exp/exp.py:101`): `json.dump(self.logger, f, indent=2)`. Fragile: a crash mid-run loses all offline metrics, since nothing is flushed incrementally.

Both branches use the same write helpers from `utils/logger.py`, which transparently handle dict-vs-Neptune:

- `log_pydantic(logger, args, "parameters")` (`logger.py:181`) recursively walks the Pydantic `Global` config and writes flattened keys like `parameters/sl/lr`, `parameters/exp/model_id`. Lists and tuples are stringified through `convert_pydantic_types` (`logger.py:203`).
- `epoch_logger(args, logger, key, value)` (`logger.py:172`) is the per-epoch metric appender. For Neptune it calls `logger[key].append(value)`. Offline it lazily creates `logger[key] = []` then appends. Used for `sl_train/loss` (`exp/exp.py:453`), `sl_val/loss`, `sl_val/mae`, coverage (`sl_*/ic`, `sl_*/jc`), interval widths, and so on (in `log_stats`, `exp/exp.py:1009`).
- Scalar assignments (for example `parameters/sl/num_params`, `parameters/running_time`) are written by direct `logger[key] = value` (`exp/exp.py:208,533`), the latter formatted by `format_time_dynamic` (`logger.py:214`).

### Checkpoints (`.pth`)

- Saved as a `state_dict` to `<log_dir>/supervised.pth`, either by `EarlyStopping.save_checkpoint` (`train.py:56`) or by the manual `torch.save` in `validate()` (`exp/exp.py:610`).
- In Neptune mode the best checkpoint is uploaded after training: `logger["model_checkpoints/{model_id}_sl"].upload(best_model_path)` (`exp/exp.py:680`).
- At test time the best checkpoint is reloaded: `torch.load(best_model_path)` to `model.load_state_dict(...)` (`exp/exp.py:735`).

### Per-sample prediction plots (`plot_*.png`)

- Triggered only in `test()`, which calls `evaluate(..., plot_results=True, num_plots=10)` (`exp/exp.py:750`).
- Inside `evaluate` (`exp/exp.py:802`), `random.sample(range(n), min(num_plots, n))` picks indices, then `self.plot(preds, targets, plotting_indices)` (`exp/exp.py:905`) renders one matplotlib figure per index: overlaid Predictions vs Targets curves, x-label `"Time (6min)"`, y-label `"Normalized EMF Exposure (V/m)"`, saved as `<log_dir>/plot_{idx}.png`. Fragile: figures are never `plt.close()`d, so a large `num_plots` leaks matplotlib figure handles.

### Offline to Neptune backfill

`offline_to_neptune()` (`logger.py:14`, the `__main__` entry of `logger.py`) is a standalone migration tool, **not** part of a training run. It scans `logs/`, and per run calls `process_experiment` (`logger.py:53`), which reads the run's `log.json`, re-initializes a Neptune run from `logs["parameters/exp/project_name"]` (`logger.py:104`), uploads the `.pth` under `model_checkpoints/{model_id}_sl`, replays metrics through `log_metrics_and_parameters` (list values appended, scalars assigned, `logger.py:132`), then `shutil.move`s the processed run dir to `old_logs/` (`logger.py:144`).

### `sincos_pos_emb.py` (peripheral to training)

This file is mostly a positional-embedding/feature library, not a training-loop component:

- `CyclicalFeatureEncoder` (`sincos_pos_emb.py:8`) turns a `(seq_len, 6)` temporal array `[year, month, day, hour, minute, second]` into `(seq_len, 2*n_features)` sin/cos pairs using fixed periods (month 12, day 31, hour 24, minute 60, second 60, weekday 7). Surprising: `"year"` is accepted in validation but silently dropped in `encode_sequence` (no period entry), and `"weekday"` has a period but no `feature_indices` entry.
- `get_{1d,2d}_sincos_pos_embed[_from_grid]` (`sincos_pos_emb.py:124-185`) are the standard ijepa/MAE fixed sinusoidal embeddings (base 10000), used by model architectures rather than the training/logging machinery documented above.

## Conformal Prediction and Uncertainty Quantification

This is the paper's headline feature: a split (inductive) conformal prediction (CP) pipeline that wraps any point-forecasting model with distribution-free prediction intervals, plus a unified **Trade-off Score (TOS)** metric. The core math lives in `emforecaster/conformal/coverage.py`. The TOS aggregation lives in `emforecaster/conformal/tradeoff.py`. Both are driven from the experiment runner `emforecaster/exp/exp.py`. The implementation is adapted from Stankevičiūtė et al.'s `conformal-rnn` (CFRNN), credited in `coverage.py:1-3`.

### Pipeline at a glance

- Split CP: the model is trained on the train set, then **calibrated on the validation loader** and **evaluated for coverage on the test loader**. The calibration and test sets are disjoint, which gives the standard split-conformal guarantee.
- Per-horizon, per-channel scores: nonconformity, critical scores, and intervals are all computed **independently for each forecast step and each channel**, never pooled. Tensors flow as `(n_samples, pred_len, num_channels)`.
- Defaults (`emforecaster/config/config.py:283-303`, `class Conformal`): `conf=False` (master switch), `alpha=0.1` (so target coverage 90%), `corrected=False` (Bonferroni off), `intervals=True` (report widths), `validation_eval=False` (calibrate only on the last or early-stopped epoch, not every epoch).

### Nonconformity scores

`nonconformity(output, target)` (`coverage.py:54-66`) is the **per-element absolute residual** through `torch.nn.functional.l1_loss(..., reduction="none")`. Despite the "Average MAE" docstring it returns the un-reduced `|output - target|`, shape `(n_samples, pred_len, num_channels)`.

```python
# coverage.py:66
return l1_loss(output, target, reduction="none")
```

### Calibration to critical scores

`get_all_critical_scores(preds, targets, alpha)` (`coverage.py:69-96`) computes the calibration scores and the conformal quantile level:

```python
# coverage.py:84-85
q           = min((n_samples + 1.0) * (1 - alpha)            / n_samples, 1)
corrected_q = min((n_samples + 1.0) * (1 - alpha / pred_len) / n_samples, 1)
```

- `q` is the finite-sample-corrected `(1-alpha)` quantile rank `ceil((n+1)(1-alpha))/n`, clamped to 1.
- `corrected_q` applies a **Bonferroni correction** over the `pred_len` horizons (`alpha` to `alpha/pred_len`) so the *joint* interval across all horizons holds at level `1-alpha`.
- Note `calibration_scores = nonconformity(...).T` (`coverage.py:79-81`) transposes to `(num_channels, pred_len, n_samples)`. `get_critical_scores` (`coverage.py:30-51`) then takes `torch.quantile(..., q)` over the sample axis for every (channel, position) and returns a `.T`-ed `(pred_len, num_channels)` tensor. Fragile: this relies on a double transpose and Python list-comprehension quantiles. It is correct but slow and easy to break if the channel axis moves.
- Returns `(critical_calibration_scores, corrected_critical_calibration_scores)`, each `(pred_len, num_channels)`.

### Forming prediction intervals at level alpha

`get_intervals(pred, scores)` (`coverage.py:100-119`) builds symmetric intervals by broadcasting the per-(horizon, channel) critical score around the point forecast:

```python
# coverage.py:115-119
lower = pred - scores
upper = pred + scores
return torch.stack((lower, upper), dim=1)   # [n_samples, 2, pred_len, num_channels]
```

The interval is centered on the point forecast and the width is **constant across samples** for a given (horizon, channel). It does not adapt per-sample.

### Empirical coverage and width

`get_coverage(preds, targets, scores)` (`coverage.py:122-140`) calls `get_intervals` then `coverage(intervals, target)` (`coverage.py:10-27`):

```python
# coverage.py:25-27
horizon_coverages = torch.logical_and(target >= lower, target <= upper)
return horizon_coverages, torch.all(horizon_coverages, dim=1)
```

- **Independent coverage (IC):** `horizon_coverages`, a hit/miss boolean per (sample, horizon, channel), the fraction of individual horizon points inside their interval.
- **Joint coverage (JC):** `torch.all(..., dim=1)`, where a sample counts only if **every** horizon step is covered at the same time. This is much stricter and is what the Bonferroni-`corrected` scores target.

### Where CP is invoked in the experiment flow

All orchestration is in `Experiment` (`emforecaster/exp/exp.py`), gated on `self.args.conf.conf`:

- Import: `from emforecaster.conformal.coverage import get_all_critical_scores, get_coverage` (`exp.py:50`).
- During `train(...)` (`exp.py:349`), when `conformal=True` it calls `self.evaluate(..., conformal_calibrate=True, coverage=False)` on the **val loader**, every epoch if `conf.validation_eval`, otherwise only on the last epoch or when early stopping fires (`exp.py:459-526`).
- `evaluate(...)` short-circuits on calibration: `if conformal_calibrate: return self.coverage(preds, targets, mode="calibrate")` (`exp.py:836-837`).
- `Experiment.coverage(...)` (`exp.py:925-997`) is the bridge to the conformal module. `mode="calibrate"` calls `get_all_critical_scores` and stores `self.calibration_scores = {"uncorrected": ..., "corrected": ...}`. `mode="inference"` selects corrected vs uncorrected through `self.args.conf.corrected` and calls `get_coverage`, then computes percentages and widths:

```python
# exp.py:978-988
ic_percent = ic.float().mean() * 100
jc_percent = jc.float().mean() * 100
interval_widths = intervals[:, 1] - intervals[:, 0]   # [n_samples, pred_len, 1]
mean_width = interval_widths.mean()
std_width = interval_widths.std()
```

- Test path: `supervised_train` to `test(...)` reloads the best checkpoint and calls `evaluate(..., coverage=self.args.conf.conf, ...)` (`exp.py:739-752`). The `coverage` block (`exp.py:885-901`) writes `stats["ic"]`, `stats["jc"]`, and (if `conf.intervals`) `stats["interval_width_mean"]`/`["interval_width_std"]`, which `log_stats` (`exp.py:1043-1067`) pushes to Neptune under keys `sl_test/ic`, `sl_test/jc`, `sl_test/interval_width_mean`, and so on.
- Score persistence (only when `validation_eval`): scores are saved to `self.score_path = log_dir/scores` (`exp.py:259`) on best-model improvement (`exp.py:613-618`) and `torch.load`-ed back at inference (`exp.py:959-962`) so coverage uses the **best** model's calibration, not the final epoch's. Surprising coupling: if `validation_eval=False` the in-memory `self.calibration_scores` from the last train call is reused instead.
- Shape guard: `coverage()` unsqueezes 2-D `(n_samples, pred_len)` tensors to add the channel axis (`exp.py:936-940`), so single-channel EMF series work without config changes.

### Unified Trade-off Score (TOS)

`compute_and_save_tos(df, save_path, beta=2/3, lambda_param=0.5)` (`tradeoff.py:6-74`) is a **standalone post-hoc CLI/aggregator** (no other module imports it, run through `tradeoff.py`'s `__main__`, `tradeoff.py:78-113`). It reads a per-model results DataFrame and renames `parameters/exp/model_id` to `model`, `sl_test/jc` to `JC`, `sl_test/ic` to `IC`, `sl_test/interval_width_mean` to `MIW` (`tradeoff.py:25-31`). It requires exactly one row per model (`tradeoff.py:40-43`). Then:

```python
# tradeoff.py:46
WAC = (beta * JC + (1 - beta) * IC) / 2          # Weighted Average Coverage, beta=2/3
# tradeoff.py:49-54
MIW_zscore  = (MIW - MIW.mean()) / MIW.std()     # width standardized across models
MIW_sigmoid = 1 / (1 + np.exp(MIW_zscore))       # narrower width -> higher score
# tradeoff.py:57-60
TOS = lambda_param * (WAC / 100) + (1 - lambda_param) * MIW_sigmoid
```

- **WAC** blends the two coverages, weighting strict joint coverage more heavily (`beta=2/3`). The trailing `/2` is a quirk: with IC/JC in percent (0 to 100), WAC stays in percent, and the `WAC/100` in TOS rescales it to [0,1], but the `/2` halves it, so WAC tops out near 50, not 100.
- **MIW_sigmoid** maps width to a "narrowness" reward in (0,1). Because the z-score is computed across the models in the file, **TOS is relative to the cohort being compared**, not absolute. A single-model CSV makes `MIW.std()` ill-defined.
- **TOS** is the final scalar balancing coverage vs narrowness, tuned by `lambda_param` in [0,1] (0.5 is equal weight, closer to 1 rewards coverage, closer to 0 rewards tight intervals).
- Output: a CSV with columns `model, IC, JC, MIW, WAC, MIW_zscore, MIW_sigmoid, TOS` written to for example `results/conformal/tos_{dataset}_pred_len{pl}_alpha{a}.csv` (`tradeoff.py:94`).

### Downstream visualization

The TOS CSV feeds the analysis plots. `lambda_plot.py` (`lambda_analysis`, `lambda_plot.py:6-172`) re-derives the TOS-vs-lambda curve per model directly from the stored `WAC` and `MIW_zscore` (sweeping `lambda` in [0,1], `lambda_plot.py:97-100`) to show how model ranking shifts with the coverage/width preference. `bar_graph.py` (`create_tos_barchart`) and `multibar_graph.py` render per-dataset and cross-dataset TOS bars. EMForecaster is the reference model in the legend ordering (`lambda_plot.py:72-81`).

### Key files

- `/Users/danielchahine/Desktop/Programs/USRA/emforecaster/emforecaster/conformal/coverage.py`: nonconformity, calibration quantiles, intervals, coverage.
- `/Users/danielchahine/Desktop/Programs/USRA/emforecaster/emforecaster/conformal/tradeoff.py`: WAC plus TOS aggregation (CLI).
- `/Users/danielchahine/Desktop/Programs/USRA/emforecaster/emforecaster/exp/exp.py`: `Experiment.coverage` (`:925`) and calibration/inference wiring.
- `/Users/danielchahine/Desktop/Programs/USRA/emforecaster/emforecaster/config/config.py:283`: `class Conformal` config keys/defaults.
- `/Users/danielchahine/Desktop/Programs/USRA/emforecaster/emforecaster/analysis/general/lambda_plot.py`: lambda-sensitivity TOS plot.

## Auxiliary Classification Task, Analysis, and HPC Jobs

This is the supporting cast around EMForecaster's forecasting core: a repurposing of the backbone for an iEEG seizure-classification task, a set of standalone analysis/plotting scripts (ARIMA, ADF, FFT/correlation figures), the Neptune-to-`results.csv` compilation pipeline, and a `submitit`-based HPC launcher for SLURM (Compute Canada). All paths below are absolute under `/Users/danielchahine/Desktop/Programs/USRA/emforecaster`.

### Auxiliary Classification Task (OpenNeuro iEEG, shapelets/wavelet features)

The classification path reuses the forecasting backbone on an OpenNeuro intracranial-EEG dataset, not on EMF data. The metrics module is keyed entirely off `args.open_neuro.task`.

- `emforecaster/utils/classification.py` is the metrics engine. The two task modes come from config (`emforecaster/config/config.py`): `OpenNeuro.task` is `"binary"` (seizure-onset-zone, SOZ vs non-SOZ) or `"multi"` (4 classes encoding SOZ times surgical-outcome, documented at `classification.py:99-104`: `0`=no-SOZ/positive, `1`=SOZ/positive, `2`=no-SOZ/negative, `3`=SOZ/negative).
- `binary_classification_metrics(preds, targets, probs, device)` (`classification.py:28`) takes shape-`(N,)` tensors, builds a `MulticlassConfusionMatrix(num_classes=2)`, and returns a stacked 7-vector `[acc, tpr, tnr, fpr, fnr, f1, auroc]` (the order matters, since `update_stats` indexes it positionally).
- `multi_classification_metrics` (`classification.py:96`) breaks the 4-class problem into two binary sub-metrics by summing softmax columns: `probs_soz = probs[:,1] + probs[:,3]`, `probs_outcome = probs[:,0] + probs[:,1]`, then appends macro `MulticlassF1Score(num_classes=4)` and multiclass `AUROC`.
- `compute_accuracy` vs `compute_channel_accuracy` (`classification.py:150`, `:212`): the latter aggregates *window-level* predictions up to a *channel-level* decision by masking on `ch_ids` and **averaging probabilities per channel** before thresholding (`args.exp.thresh`, default `0.5`) for `BCE`/`ch_loss_type` or `argmax` for `CE`. `get_metrics(...)` (`classification.py:307`) is the dispatcher: `mode="window"|"channel"`, and `args.exp.batchwise_metrics` toggles between a size-weighted running average over a *list* of batches vs concatenating all batches (the latter only computes on `rank==0`, DDP-aware).
- Calibration hooks (`calibrator.calibrate`) fire under `args.exp.calibration_type == "window"` or `"channel"`. `sklearn` `IsotonicRegression`/`CalibratedClassifierCV`/`SVC` are imported, but the calibrator object itself is passed in from outside this file.

```python
metrics = torch.stack([acc, tpr, tnr, fpr, fnr, f1_score, auroc_score])  # classification.py:91
```

- **Feature side:** `emforecaster/utils/wavelets.py::wavelet_features(signal, fs, num_scales=35, f_min=0.5, f_max=300.0, wavelet="cmor1.5-1.0")` runs a CWT (through `pywt`/`ptwt`) producing scalograms of shape `(batch_size, num_scales, signal_length)` tuned for iEEG bands (delta to HFO). The shapelet layer lives at `emforecaster/layers/shapelets/shap.py`, which loads data through `load_open_neuro_interchannel(...)`. Fragile note: `shap.py` runs feature extraction at module import time (top-level `data = load_open_neuro_interchannel(...)`), so importing it triggers I/O. Overall this task is clearly secondary scaffolding. There is a stray `# TODO:` at `classification.py:120` and the metrics ordering is implicit and positional.

### Analysis Scripts (ARIMA, ADF, plotting)

These are run-as-`__main__` research scripts under `emforecaster/analysis/general/`, all sharing `load_forecasting(...)` from `emforecaster/utils/dataloading.py` and LaTeX/`usetex` matplotlib styling. They write `.eps`/`.png` figures, usually to `../../../figures/general/` resolved relative to the current directory (so they must be run from their own directory).

- **ARIMA baseline:** `arima.py::evaluate_arima_sliding(train, val, test, window_size=336, pred_lengths=[96,192,336,512], order=(1,1,1))`. It fits a `statsmodels` `ARIMA` **once** on the flattened training series, then for each `pred_len` slides over the test set, re-applying the fitted params to each input window through `fitted_model.apply(input_window).forecast(steps=pred_len)` (no refit), reporting `mse/mae/rmse`. Surprising: the module runs `load_forecasting(dataset_name="rf_emf_ptv2", ...)` and the full evaluation at import/`__main__` with hardcoded args and `order=(1,0,1)` (overriding the signature default), and silently `except`s failed windows.
- **ADF stationarity:** `adf.py` runs `statsmodels` `adfuller` per channel and renders a scatter of ADF statistic vs series, color-coded by p-value bins (`<0.01` Strongly, `<0.05` Moderately, `<0.1` Weakly, else Non-Stationary, map at `adf.py:103-113`). `plot_adf_results` is single-series. `plot_adf_comparison` overlays original vs `np.diff` (first-difference) series. CLI: `python adf.py <dataset_name> [--diff]`, with a hardcoded `map`/`dataset_map` of the Italy (`rf_emf_det/ptv/ptv2/tur/tur2`) and Turkey (`rf_emf`, 17 locations) datasets. Fragile: the `"both"`-mode branch (`adf.py:586-602`) has an `if/elif/else` that raises on `"both"` because the `turkey` branch's `else` rejects it.
- **General plotting:** `plotting.py` (`set_theme`, `plot_correlation_heatmap`, `plot_fft`, `analyze_emforecaster`) builds the dataset/correlation/FFT figures. `plot_fft` converts FFT bins to period axes with hour markers `[1,2,4,8,12,24,48]h`. Note `plot_correlation_heatmap` references a global `mode` that is only defined in the `__main__` block (`plotting.py:131`), so it is not importable on its own. `analysis.py` is an older/overlapping variant of the same figure generator (much of its 2x2 panel logic is commented out), effectively superseded by `plotting.py`.

### Neptune to results.csv Compilation

`emforecaster/analysis/neptune/get_results.py` is the main aggregator. It must be run from inside `analysis/neptune/` because paths like `ablations/...` and `results/...` are relative.

- Entry: `python get_results.py <dataset> <model_id>`, which loads `ablations/<dataset>/<model_id>.yaml` (for example `ablations/rf_emf/patched_forecaster.yaml`) and `neptune.init_project(...)` using `NEPTUNE_API_TOKEN` from the environment.
- `get_query(filter)` (`get_results.py:27`) builds a Neptune NQL string from a typed dict, type-tagging each value (`(key:type = value)`) and always appending `AND sys/failed:bool = False`. `get_runs_table` then `fetch_runs_table(...).to_pandas()` pulling `params + ["parameters/exp/id"] + metrics`.
- For every Cartesian combination of `params` (`get_unique_combinations` through `itertools.product`), it filters runs, calls `filter_duplicate_runs(df, seeds, ...)` (which **only supports `mode="recent"`**, asserting otherwise, and on a row-count mismatch keeps the rows whose `parameters/exp/id` has the latest `sys/creation_time`), then `compute_averages` over the `seeds` (`[69421..69425]` in the YAML), and finally sorts by `deciding_metric` (`ranking: "min"|"max"`) and writes `results/<dataset>/<model_id>.csv`.
- Two modes: `single_filter` (one query to one CSV) and the commented-capable `multi_filter` (product of filter lists to `results/<dataset>/<model_id>/<i>.csv`). The ablation YAMLs define `metrics` like `sl_test/loss`, `sl_test/mae`, `sl_test/jc`, `sl_test/ic`, `sl_test/interval_width_mean`. Note `jc`/`ic`/`interval_width` are the **conformal-prediction** coverage/efficiency metrics, tying this back to the flagship uncertainty pipeline.
- `compile_results.py::transfer_rows(dataset, source_file, n_rows, target_file="ctsf.csv")` is a thin post-step that copies the top `n` rows of one results CSV into a combined `results/<dataset>/ctsf.csv`, intersecting only on columns present in the target.

### HPC Job Submission (`submitit`, SLURM / Compute Canada)

`emforecaster/jobs/hpc/submit.py` submits training/grid-search jobs to SLURM through `submitit.AutoExecutor`. It is a two-file-per-experiment config system. Run it as `python submit.py <exp>` where `<exp>` is a folder under `emforecaster/jobs/hpc/` (for example `rf_emf/tsmixer`, `test`).

- `load_configs(exp_name)` reads two YAMLs from that folder: `compute_canada.yaml` (env/cluster: `virtual_env`, `cluster`, `modules`, `logdir`, `script_path`, `job_name`) and `slurm.yaml` (resources: `gpus_per_node`, `tasks_per_node`, `cpus_per_task`, `slurm_mem` in **GB**, `slurm_time`, `slurm_account`, `slurm_mail_*`).
- `submit_job(...)` (`submit.py:79`) sets executor params. Note `slurm_mem` is multiplied by 1024 (`kwds.get("slurm_mem",12)*1024`), `slurm_account` defaults to `"def-hinat"`, and `slurm_mail_user` defaults to a hardcoded address. Then `executor.submit(run_script, ...)`. Importantly, `cli_input` passed to the job is just `cc_config["job_name"]` (`submit.py:167`), that is, the experiment selector string handed to the target script.
- `run_script(...)` runs *on the cluster*. For `cluster == "narval"` it loads a hardcoded Compute Canada module stack (`StdEnv/2023`, `python/3.10`, CUDA 12.2, NCCL, cuDNN, and so on, `submit.py:32-44`), otherwise `module load {modules}`. Then `source ~/<virtual_env>/bin/activate` and runs `python <script_path> <cli_input>` (or `bash` for `.sh`) through `subprocess.run(..., shell=True)`, logging stdout/stderr through `rich`.
- The real experiment grid lives under `emforecaster/jobs/hpc/rf_emf/` (per-model dirs: `tsmixer`, `timesnet`, `patchtst`, `univar`, and `baselines/{transformer,cnn,mlp,...}`), each pointing `script_path` at `emforecaster/tuning/grid_search/grid_search.py`. Fragile and surprising: `logdir`/`script_path` are absolute paths baked to one user's `/home/xmootoo/...` tree, and `test1`/`test2` configs are stale (for example `test1/compute_canada.yaml` is missing the `cluster` and `job_name` keys that `submit.py` requires).

## Part II. The Model Zoo

EMForecaster ships **ten** model implementations. One is the flagship, and the rest are benchmarks spanning every major time-series architecture family (Transformer, CNN/TCN, RNN, state-space, MLP-mixer, linear). Every model is selected purely by the `exp.model_id` string in the YAML config and built by the `get_model` factory in `emforecaster/utils/models.py`.

### Model Comparison Table

| Model | Family | Selected by (`exp.model_id`) | Core idea |
| --- | --- | --- | --- |
| **EMForecaster** | MLP-Mixer | `EMForecaster` | The paper's flagship single-variable forecaster. A patch-then-embed model that cuts the input series into overlapping patches, embeds each patch with a small linear-style "patch model" (DLinear or TSMixer), adds positional encoding, then forecasts with a linear head, all wrapped in RevIN. |
| **PatchTST** | Transformer | `PatchTST` | A faithful re-implementation of PatchTST (Nie et al., ICLR 2023): a channel-independent patch-based Transformer encoder for long-horizon time-series forecasting, with RevIN instance normalization and a flatten-linear prediction head. |
| **TimesNet** | CNN-TCN | `TimesNet` | A baseline time-series model that reshapes each 1D series into 2D tensors along FFT-discovered dominant periods and applies 2D Inception convolutions, capturing intra- and inter-period variations for forecasting and classification. |
| **ModernTCN** | CNN-TCN | `ModernTCN` | A pure-convolutional time-series model that replaces self-attention with re-parameterized depthwise large-kernel 1D convolutions plus grouped 1x1 ConvFFNs, operating channel-independently over patched, RevIN-normalized inputs. |
| **DLinear** | Decomposition | `DLinear` | A decomposition-based linear forecaster that splits the series into trend and seasonal components through a moving-average kernel, then maps each with a single linear layer per component and sums them. |
| **TSMixer** | MLP-Mixer | `TSMixer` | An all-MLP time-series forecaster that alternately mixes information along the time axis and the channel axis using residual MLP blocks, wrapped in reversible instance normalization (RevIN). |
| **Mamba** | SSM | `RecurrentModel` plus `backbone_id=Mamba` | A selective state-space (Mamba) sequence backbone wrapped with optional RevIN, PatchTST-style patching, and a linear/MLP forecasting head. In practice it is reached through the unified RecurrentModel class with backbone_id="Mamba". |
| **Recurrent (RNN / LSTM / GRU)** | RNN | `RecurrentModel` plus `backbone_id=RNN/LSTM/GRU` | A single unified recurrent backbone wrapper (`RecurrentModel`) that hosts vanilla RNN, LSTM, GRU, or a Mamba SSM behind an optional RevIN plus patching plus positional-encoding front end and a linear/MLP forecasting head, selected at runtime by `backbone_id`. |
| **Linear** | Linear | `Linear` | A single fully-connected layer that maps the entire lookback window directly to the forecast horizon, optionally preceded by a normalization layer. The simplest "Linear" forecasting baseline in the style of LTSF-Linear (Zeng et al., AAAI 2023). |
| **RF-EMF Baselines (CNN / MLP / Transformer / LSTM)** | Transformer / MLP / CNN-TCN / RNN (a bundle of four small single-variable baselines) | `RF_EMF_{MLP,CNN,LSTM,Transformer}` | A bundle of four deliberately small, single-variable forecasting baselines (MLP, 1D-CNN, single-layer LSTM, and a tiny convolutional Transformer) used as comparison points against the flagship EMForecaster model on RF-EMF data. |

> **Shared scaffolding.** Most models share the same front-end and back-end machinery, toggled by config: **RevIN** reversible instance normalization (`sl.revin/revout/revin_affine`), **patching** (`data.patch_dim/patch_stride`, through `layers/patcher.py`), **learnable positional encoding**, and a flatten-then-`Linear` forecasting head. The interesting differences are in the *backbone* that sits between them. The tensor-shape convention below is `(B, C, L)` in to `(B, C, H)` out, where `B`=batch, `C`=channels (=1 for single-variable EMF), `L`=`seq_len`=336, `H`=`pred_len`=96.

### EMForecaster

The flagship model of the paper (IEEE TNSE 2025). Defined in `emforecaster/models/emforecaster.py:21` as the class `EMForecaster(nn.Module)`.

**What it is.** EMForecaster is a *patch-based* single-variable time-series forecaster built entirely from linear and MLP-mixing pieces (no attention). The core mechanism splits the forecasting problem into two stages:
1. A **patch model** that independently or jointly embeds short overlapping windows ("patches") of the series into a higher-dimensional representation.
2. A **backbone head** that consumes the flattened patch embeddings (plus positional encoding) and produces the forecast.

The design borrows the *patching* idea from PatchTST (Nie et al., 2023, see the RevIN provenance comment in `emforecaster/layers/patchtst/revin.py:9` citing PatchTST and Kim et al. 2022) but replaces the Transformer encoder with a cheap linear-style embedder. Both the patch model and the backbone come from **DLinear** (Zeng et al., 2022, "Are Transformers Effective for Time Series Forecasting?", arxiv 2205.13504, see `emforecaster/models/dlinear.py:9`) and **TSMixer** (Chen et al., 2023, see `emforecaster/models/tsmixer.py:6`). Family: **MLP-Mixer / linear** (patching plus per-patch MLP/linear embedding plus an MLP-mixer or DLinear backbone). It is *not* a Transformer despite the patching heritage.

**Role in this repo.** Forecasting only (the constructor docstring says "Patched-based forecasting model (univariate only)", `emforecaster/models/emforecaster.py:50`, `num_channels=1`). It is the **flagship**, not a baseline. It has its own job directory `emforecaster/jobs/exp/rf_emf/emforecaster/args.yaml` plus an `ablation.yaml`, while DLinear/TSMixer/PatchTST/TimesNet/etc. are baselines.

**How it's built.**
- Class: `EMForecaster` (`emforecaster/models/emforecaster.py:21`). It composes sub-modules rather than defining new low-level layers.
- Selected by the factory in `emforecaster/utils/models.py:103` (`elif args.exp.model_id == "EMForecaster":`), which maps `Global` config keys to constructor args (`emforecaster/utils/models.py:104-123`).
- Config class: the top-level `Global` (`emforecaster/config/config.py:719`), pulling from sub-configs `exp` (`Experiment`), `data` (`Data`), `sl` (`SL`), `dlinear` (`DLinear`), `tsmixer` (`TSMixer`). The whole `args: BaseModel` object is passed in (constructor arg `args`, `emforecaster/models/emforecaster.py:24`) so the model can later select the patch-model sub-config through `get_patch_model_config()` (`:188`), which maps `{"DLinear": self.args.dlinear, "TSMixer": self.args.tsmixer}`.
- Constructor args and defaults (`emforecaster/models/emforecaster.py:22-45`): `seq_len=336`, `pred_len=96`, `num_channels=1`, `revin=True`, `revout=True`, `revin_affine=True`, `eps_revin=1e-5`, `patch_model_id="DLinear"`, `backbone_id="DLinear"`, `patch_norm="layer"`, `patch_act="GeLU"`, `patch_dim=24`, `patch_stride=12`, `patch_embed_dim=128`, `independent_patching=False`, `pos_enc="learnable"`, `patching_on=True`, `num_patches=0`, `return_head=True`.
- Derived hyperparameters: `patch_stride==-1` gives `patch_dim//2` (50% overlap), `patch_stride==-2` gives `patch_dim` (no overlap) (`:86-91`). `num_patches = int((seq_len - patch_dim)/stride) + 2` (`:93`).
- Config defaults (`config.py`): `patch_norm="layer"`, `patch_act="gelu"`, `patch_embed_dim=128`, `independent_patching=False`, `pos_enc="learnable"` (`config.py:427-445`), and `patch_dim` default 16 (`config.py:198`).
- Realistic config (`emforecaster/jobs/exp/rf_emf/emforecaster/args.yaml`): `patch_model_id: TSMixer`, `backbone_id: Linear`, `patch_dim: 8`, `patch_stride: -1`, `seq_len: 336`, `pred_len: 96`, `patch_embed_dim: 256`, `patch_norm: none`, `patch_act: gelu`, `pos_enc: none`, `independent_patching: False`, `tsmixer: {num_enc_layers: 2, d_model: 256, dropout: 0.3}`, `dlinear.final_moving_avg: 25`, RevIN all on. So the *shipped* flagship config is the **TSMixer-patch plus Linear-backbone** variant with positional encoding and patch-norm disabled, not the constructor defaults.
- All weights are Xavier-initialized through `self.apply(lambda m: xavier_init(m, seed=seed))` (`:140`, helper in `emforecaster/utils/weight_init.py:48`). DLinear sub-modules override their linear weights to a uniform `1/seq_len` moving-average init (`dlinear.py:59-74`).

**How it runs (the forward pass).** `forward(self, x)` at `emforecaster/models/emforecaster.py:142`. Input `x` is `(B, C=1, L=seq_len)`.
1. **RevIN normalize** (`:147-148`): if RevIN is enabled, `x = self.revin(x, mode="norm")`, a per-instance z-score over the time dim with optional affine. Shape unchanged `(B, 1, L)`.
2. **Patching** (`:151`): `x = self.patcher(x).squeeze()`. `Patcher` (`emforecaster/layers/patcher.py:5`) right-pads the series by `stride` copies of the last value (`patcher.py:31-33`), then `unfold`s into `N = int((L-patch_dim)/stride)+2` patches, returning `(B, 1, N, P=patch_dim)`. The `.squeeze()` drops the channel dim to `(B, N, P)`.
3. **Patch embedding** (`:155-161`):
   - If `independent_patching=True`: reshape to `(B*N, 1, P)` and run the patch model per patch, reshape back to `(B, N, patch_embed_dim)`, channel-independent over patches.
   - Else (default): `x = self.patch_model(x)` on `(B, N, P)`, treating the N patches as the channel axis (`get_patch_model` sets `num_channels=num_patches`, `:219`), output `(B, N, patch_embed_dim)`. For `patch_model_id="DLinear"` the embedder is a `DLinear(task="forecasting", seq_len=patch_dim, pred_len=patch_embed_dim, individual=False, return_head=False)` (`:224-233`). For `"TSMixer"` it is `nn.Sequential(nn.Linear(patch_dim, patch_embed_dim), TSMixer(seq_len=patch_embed_dim, pred_len=1, num_enc_layers, d_model, dropout, revin/revout=False, return_head=False))` (`:234-248`).
   - Note: the DLinear branch does NOT use `final_moving_avg` here. It uses `self.patch_model_config.moving_avg` (`:230`).
4. **Activation** (`:164`): `self.patch_act`, one of `nn.ReLU`/`nn.GELU`/`nn.Identity` (built at `:123-128`). Note the string check is lowercase `"relu"`/`"gelu"`, so the default `"GeLU"` falls through to **Identity**.
5. **Patch norm** (`:167`): `self.patch_norm` is `nn.LayerNorm(patch_embed_dim)` only when `patch_norm=="layer"` AND `independent_patching` is True (`:131-134`), otherwise `nn.Identity` (so in the default/joint-patching path it is a no-op).
6. **Head path** (`:169-184`, only if `return_head=True`):
   - Flatten: `x.view(B, 1, N*patch_embed_dim)` (`:172`).
   - Add learnable positional encoding `x = x + self.pos_enc` where `pos_enc` is `nn.Parameter(torch.randn(1, 1, N*patch_embed_dim))` for `"learnable"`, or the scalar `0` for `"none"` (`:115-120`).
   - **Backbone**: `x = self.base_model(x)` to `(B, 1, pred_len)`. `get_base_model` (`:199`) returns either `DLinear(seq_len=N*patch_embed_dim, pred_len, num_channels=1, moving_avg=args.dlinear.final_moving_avg, individual=True, return_head=True)` for `backbone_id="DLinear"`, or `nn.Linear(N*patch_embed_dim, pred_len)` for `backbone_id="Linear"`.
   - **RevIN denormalize** (`:183-184`): if `revout`, `x = self.revin(x, mode="denorm")`, restoring the original scale.
   Output: `(B, 1, pred_len)`.

   If `return_head=False`, the forward returns the patch-embedded tensor before flattening/backbone. This is used by extensions (for example CyclicalEMForecaster) that supply their own head.

**Custom layers it depends on.**
- `Patcher` (`emforecaster/layers/patcher.py:5`): last-value-padded `unfold` patcher returning `(B, M, N, P)`. (The sibling `VerticalPatcher` at `:49` is unused by EMForecaster.)
- `RevIN` (`emforecaster/layers/patchtst/revin.py:5`): reversible instance norm (Kim et al. 2022), per-instance mean/std over the time axis with learnable affine, with `norm`/`denorm` modes.
- `DLinear` (`emforecaster/models/dlinear.py:7`): series-decomposition (trend plus seasonal through `series_decomp`/moving average, `emforecaster/layers/dlinear/series_decomp.py`) followed by two linear maps. Used both as the patch embedder (`return_head=False`) and optionally as the backbone (`individual=True`, `return_head=True`).
- `TSMixer` (`emforecaster/models/tsmixer.py:47`): a stack of `ResBlock`s, each a residual temporal MLP plus residual channel MLP (`tsmixer.py:9-44`). Used as the alternative patch embedder.
- `xavier_init` (`emforecaster/utils/weight_init.py:48`). `Transpose`/`Unsqueeze` (`emforecaster/utils/utils.py:14,23`) are imported but not used in `forward`.
- The `emforecaster/layers/wavelets/*` files (`transform.py`, `dwt.py`, `cwt.py`, `purturbation.py`) are **standalone experimental/scratch scripts** (top-level `plt.show()`, demo code) for a wavelet-perturbation uncertainty idea. They are **not imported by `emforecaster.py`** and are not part of this model's graph.

**Notable details / gotchas.**
- **Single-variable by design:** `num_channels=1` throughout. RevIN runs on the single channel and the backbone is hard-coded to `num_channels=1` (`:205`, `:211`).
- **Channel/patch axis trick:** with `independent_patching=False` the N patches play the role of "channels" inside the DLinear/TSMixer patch model (`:219`). This is patch-*dependent* embedding. `independent_patching=True` switches to patch-*independent* (each patch reshaped to `(B*N,1,P)`).
- **`patch_act` casing bug:** the default `patch_act="GeLU"` (and the YAML/constructor strings) only matches lowercase `"relu"`/`"gelu"` in `:123-126`, so `"GeLU"` silently becomes `nn.Identity`. With the literal default, no activation is applied. The shipped job sets `"gelu"` (lowercase), which does activate.
- **`patch_norm` only fires with independent patching:** `nn.LayerNorm` is applied only when `patch_norm=="layer"` AND `independent_patching` (`:131`). In the default joint path it is Identity regardless.
- **`num_patches` off-by-design `+2`:** both `EMForecaster` (`:93`) and `Patcher` (`patcher.py:28`) use `int((L-patch_dim)/stride)+2`, reflecting the one-stride last-value padding added before unfolding.
- **Series decomposition is internal to DLinear** (trend through moving average plus residual seasonal), applied inside both the patch embedder and the DLinear backbone. EMForecaster itself does no top-level decomposition.
- **Two-level RevIN:** EMForecaster owns the outer RevIN. When TSMixer is the patch model, its internal RevIN is explicitly disabled (`revin=False, revout=False`, `:243-246`) to avoid double normalization.
- **`patching_on=False`** branch (`:94`, `:152`) is a hook for extensions that patch externally and pass pre-patched tensors plus `num_patches`. In that path `B` is never set inside `forward` before the `independent_patching` reshape, so that combination would error. It is intended for the `return_head=False`, externally-managed use case.
- `final_moving_avg` (DLinear backbone) vs `moving_avg` (DLinear patch embedder) are distinct config knobs (`config.py:633,642`). The flagship sets `moving_avg: 3`, `final_moving_avg: 25`, but with `backbone_id: Linear` the `final_moving_avg` is unused.

### PatchTST

**What it is.** A from-scratch re-implementation of **PatchTST**, "A Time Series is Worth 64 Words: Long-term Forecasting with Transformers" (Nie, Nguyen, Sinthong, Kalagnanam, ICLR 2023, original code at `github.com/yuqinie98/PatchTST`). The code explicitly references that origin: `emforecaster/layers/patchtst/revin.py:8-11` cites the RevIN source repo, and `pos_enc.py` (commented-out test block, lines 37-98) compares "PatchtstOG vs ours" to verify equivalence to the original projection plus positional-encoding math (`W_P`, `W_pos`). The core mechanism is the PatchTST recipe: **(1)** reversible instance normalization (RevIN), **(2)** split each single channel into overlapping **patches**, **(3)** linearly embed each patch and add a learnable positional encoding, **(4)** process patches with a vanilla Transformer encoder applied **channel-independently** (all channels share weights, the channel dim is folded into the batch dim), **(5)** flatten the per-channel token sequence and apply a linear head to produce the forecast, **(6)** denormalize with RevIN. Family: **Transformer** (patch-tokenized, encoder-only, channel-independent).

**Role in this repo.** A forecasting **baseline** (the flagship is `EMForecaster`). It is registered as a supported model in `emforecaster/utils/models.py` (model lists at lines 334, 397, 425) and has its own experiment config at `emforecaster/jobs/exp/rf_emf/patchtst/args.yaml` (`exp.model_id: "PatchTST"`, `task: "forecasting"`). Note this is distinct from `PatchTSTOG` (a separate entry that wraps the original authors' code). The `forward` accepts `y` and `ch_ids` arguments but they are unused for plain forecasting. They are passed through to the backbone head, which ignores them too. Only the `target_channels` path (never wired up by the factory) would use channel selection.

**How it's built.**
- Top-level class: `PatchTST(nn.Module)` in `emforecaster/models/patchtst.py:12`. Constructor args (with defaults): `num_enc_layers, d_model, d_ff, num_heads, num_channels, seq_len, pred_len, attn_dropout=0.0, ff_dropout=0.0, pred_dropout=0.0, batch_first=True, norm_mode="batch1d", revin=True, revout=True, revin_affine=True, eps_revin=1e-5, patch_dim=16, stride=1, return_head=True, head_type="linear", channel_independent=False, target_channels=None`.
- The number of patches is precomputed: `self.num_patches = int((seq_len - patch_dim)/stride) + 2` (`patchtst.py:41`). Note the `+2` (one extra patch vs the usual `+1` because the Patcher pads `stride` extra steps, see gotchas).
- Sub-modules built in `__init__`: `RevIN` (if `revin`), `Patcher`, `PositionalEncoding`, `PatchTSTBackbone`. Weights are then initialized with `self.apply(xavier_init)` (`patchtst.py:77`, `xavier_init` in `emforecaster/utils/weight_init.py:48` does Xavier-uniform on Linear weights, zeros biases, ones/zeros on LayerNorm, special init for LSTM).
- **Config/factory.** Selected when `args.exp.model_id == "PatchTST"` and built in `emforecaster/utils/models.py:42-63`. The factory pulls hyperparameters from the shared `SL` config (`args.sl.*`) and the data config (`args.data.*`), **not** primarily from the dedicated `PatchTST` config block. The dedicated config class is `config.PatchTST` (`emforecaster/config/config.py:606`, mounted as `patchtst` at line 730) with model-specific defaults `num_enc_layers=3, d_model=16, d_ff=128, num_heads=4, attn_dropout=0.3, ff_dropout=0.3, norm_mode="batch1d"`. Relevant `SL` defaults actually consumed by the factory (`config.py:306+`): `pred_dropout=0.1, batch_first=True, norm_mode="batch1d", revin=True, revout=True, revin_affine=True, eps_revin=1e-5, head_type="linear"`. The factory does **not** pass `channel_independent`, `target_channels`, or `return_head`, so those fall back to the constructor defaults (`channel_independent=False`, `target_channels=None`, `return_head=True`).
- **Realistic hyperparameters** from `emforecaster/jobs/exp/rf_emf/patchtst/args.yaml`: single-variable EMF (`num_channels: 1`), `seq_len: 20`, `pred_len: 20`, `patch_dim: 16`, `patch_stride: 8`, `num_enc_layers: 3`, `d_model: 16`, `d_ff: 32`, `num_heads: 2`, `attn_dropout: 0.0`, `ff_dropout: 0.3`, `pred_dropout: 0.0`, OneCycle scheduler, Adam, MSE, batch 512, 100 epochs.

**How it runs (the forward pass).** `PatchTST.forward(x, y=None, ch_ids=None)` (`patchtst.py:88`). The logical input shape is `(B, C, L)` with `C = num_channels`, `L = seq_len` (the channel dim is dim 1, and the Patcher/RevIN docstrings treat the last dim as the time axis):
1. **RevIN norm** (`patchtst.py:92`): `x = self.revin(x, mode="norm")`. `RevIN.forward` (`revin.py:26`) computes per-instance, per-channel mean/std over the time axis (`dim=2`), subtracts/divides, then applies a learnable affine `weight`/`bias` of shape `(num_channels,)`. Statistics are cached on the module for denorm. Output `(B, C, L)`.
2. **Patcher** (`patchtst.py:95`): `Patcher.forward` (`emforecaster/layers/patcher.py:15`) pads the series by repeating the last timestep `stride` times (`patcher.py:31-33`), then `x.unfold(dim=2, size=patch_dim, step=stride)` extracts overlapping patches, reshaping to `(B, C, N, P)` with `N = int((L - patch_dim)/stride) + 2` and `P = patch_dim`. (It also computes a `patches_combined` `(B*C, N, P)` view but returns only `patches`.)
3. **Project plus Positional Encoding** (`patchtst.py:98`): `PositionalEncoding.forward` (`pos_enc.py:18`) reshapes `(B,C,N,P)` to `(B*C,N,P)`, applies `nn.Linear(patch_dim, d_model)` (the `P x D` projection `W_P`), adds a learnable `nn.Parameter` positional matrix of shape `(num_patches, d_model)` (init `uniform_(-0.02, 0.02)`), then reshapes back to `(B, C, N, d_model)`.
4. **Backbone** (`patchtst.py:101`): `PatchTSTBackbone.forward` (`backbone.py:84`) reshapes to `(B*C, N, d_model)` (folding channels into the batch dim is what makes the encoder **channel-independent**), runs `self.enc`, an `nn.Sequential` of `num_enc_layers` `EncoderBlock`s. Each `EncoderBlock` (`enc_block.py:36`) does: MultiheadAttention (`enc_block.py:86`, `attn(x,x,x)`, `need_weights=False`), residual add plus `Norm`, position-wise FFN `Linear(d_model,d_ff) -> GELU -> Dropout -> Linear(d_ff,d_model)`, residual add plus `Norm`. Then with `return_head=True` it reshapes to `(B, C, N*d_model)` (`backbone.py:96`), optionally slices `target_channels`, and applies the head. With `head_type="linear"` and `channel_independent=False` (the default factory path), the head is `Reshape(-1, C*N*d_model) -> Linear(C*N*d_model, C*pred_len) -> Reshape(-1, C, pred_len)` (`backbone.py:65-73`) to output `(B, C, pred_len)`. (`head_type="mlp"` swaps in a 2-layer GELU MLP. `channel_independent=True` uses a per-channel `Linear(N*d_model, pred_len)`.)
5. **RevIN denorm** (`patchtst.py:105`): `x = self.revin(x, mode="denorm")` reverses the affine transform and re-applies the cached std/mean. Final output `(B, C, pred_len)`, which is `(B, H, C)` after transposition by the caller.

**Custom layers.**
- `emforecaster/layers/patchtst/revin.py`: `RevIN`, reversible per-instance/per-channel normalization (Kim et al. 2022), with learnable affine params and cached mean/std for the inverse.
- `emforecaster/layers/patcher.py`: `Patcher`, unfolds the series into overlapping patches with last-value padding (also defines an unused `VerticalPatcher`).
- `emforecaster/layers/patchtst/pos_enc.py`: `PositionalEncoding`, per-patch linear embedding `P` to `d_model` plus a learnable additive positional matrix.
- `emforecaster/layers/patchtst/backbone.py`: `PatchTSTBackbone`, stacks `EncoderBlock`s and owns the flatten plus prediction head.
- `emforecaster/layers/patchtst/enc_block.py`: `EncoderBlock` (attention plus FFN with residual plus Norm), `_MultiheadAttention` (thin wrapper over `nn.MultiheadAttention`), and an unused `SupervisedHead`.
- `emforecaster/utils/utils.py`: `Norm` (`utils.py:41`, selects BatchNorm1d through Transpose / BatchNorm2d / LayerNorm by `norm_mode`) and `Reshape` (`utils.py:32`).

**Notable details / gotchas.**
- **Channel independence** is implemented by reshaping `(B,C,...)` to `(B*C,...)` before the encoder (`pos_enc.py:28`, `backbone.py:90`). All channels share encoder weights. But the **default `head_type="linear"` head is NOT channel-independent**, because `channel_independent=False` by default, so the head flattens **across channels** (`C*N*d_model` to `C*pred_len`, `backbone.py:65-73`), mixing channels at the output. True per-channel heads require passing `channel_independent=True`, which the factory never does. For the single-variable EMF config (`num_channels: 1`) this distinction does not matter.
- **RevIN normalizes over `dim=2`** (`revin.py:50`), that is, the time axis when input is `(B, C, L)`. Affine params are sized `(num_channels,)`. RevIN also auto-unsqueezes a missing channel dim (`revin.py:37-38`).
- **Patch count `+2`, not `+1`.** Because `Patcher` pads `stride` extra steps with the repeated last value (`patcher.py:31-33`), `num_patches = (L - patch_dim)/stride + 2`. The model and Patcher must agree (both use `+2`), so changing the padding logic would silently break positional-encoding/head dimensions.
- **`norm_mode="batch1d"`** (the default) uses BatchNorm over `d_model` through `Transpose(1,2) -> BatchNorm1d -> Transpose(1,2)` (`utils.py:50-51`), matching PatchTST's use of BatchNorm rather than LayerNorm. `"layer"` and `"batch2d"` are alternatives.
- **Latent bug, RevIN signature mismatch.** `PatchTST._init_revin` (`patchtst.py:81-86`) constructs `RevIN(num_channels=..., eps=..., affine=..., target_channels=self.target_channels)`, but `RevIN.__init__` (`revin.py:6`) only accepts `(num_channels, eps=1e-5, affine=True)`. There is no `target_channels` parameter (confirmed: only one `RevIN` class in this package). Since `revin` defaults to `True` (constructor, `SL` config, and the args.yaml all enable it), building `PatchTST` would raise `TypeError: __init__() got an unexpected keyword argument 'target_channels'`. This path is exercised by the standard factory call, so either this baseline is currently not runnable as-is or it is only run with `revin=False`. Flagging it as a genuine inconsistency rather than guessing intent.
- The `forward` signature accepts `y` and `ch_ids` for API uniformity with other models, but for forecasting they are unused (passed to `backbone.forward`, which also ignores them).

### TimesNet

**What it is.** A faithful port of **TimesNet** (Wu et al., ICLR 2023, "TimesNet: Temporal 2D-Variation Modeling for General Time Series Analysis"). The class docstring cites the paper directly (`emforecaster/models/timesnet.py:13`, `https://openreview.net/pdf?id=ju_Uqw384Oq`) and the layer files note they were taken from the THU `Time-Series-Library` repo (for example `inception_block.py:1`, `embed.py:1`, `timesblock.py:13`). The core idea: a 1D time series has several periodicities. TimesNet uses an **FFT** to find the `top_k` dominant periods, **folds the 1D sequence into a 2D tensor** of shape `(num_periods, period)` for each period, and processes that 2D map with **2D Inception (parallel multi-kernel) convolutions**. This turns intra-period (within one cycle) and inter-period (across cycles) variation into spatial structure a 2D CNN can model. By family it is a **convolutional / 2D-CNN** model (closest to the CNN-TCN bucket in this zoo), not a Transformer.

**Role in this repo.** A **baseline** (the flagship is the EMForecaster model plus conformal prediction). It is general-purpose: a single `task` switch supports both **forecasting** and **classification** (`forward`, `timesnet.py:151-161`). It appears in the model-name lists in `emforecaster/utils/models.py` at lines 341, 404, 431 alongside the other baselines, and there is an HPC grid-search launcher at `emforecaster/jobs/hpc/rf_emf/timesnet/compute_canada.yaml` (note: that file only holds `logdir`/`script_path`/`job_name`, not hyperparameters, and there is no per-model `args.yaml` for TimesNet under `jobs/exp/`).

**How it's built.**
- Class `TimesNet(nn.Module)` in `emforecaster/models/timesnet.py:11`.
- Constructor args (`timesnet.py:16-34`): `seq_len, pred_len, num_channels, d_model, d_ff, num_enc_layers, num_kernels, c_out, top_k, dropout, task, revin=False, revin_affine=False, revout=False, eps_revin=1e-5, return_head=True`.
- Built by the factory at `emforecaster/utils/models.py:147-165` under `model_id == "TimesNet"`. The args are pulled from **two** config groups: model-specific `args.timesnet.*` (`d_model`, `d_ff`, `num_kernels`, `c_out`, `top_k`) and the shared **`args.sl.*` / `args.data.*` / `args.exp.*`** groups for the rest. Notably `num_enc_layers` comes from `args.sl.num_enc_layers` and `dropout` from `args.sl.dropout`, so the `num_enc_layers`/`dropout` fields on the `TimesNet` config class are effectively **shadowed and unused** by the factory.
- Config class `TimesNet(BaseModel)` (`emforecaster/config/config.py:661-683`) defaults: `num_enc_layers=2`, `d_model=16`, `d_ff=128`, `num_kernels=6`, `c_out=1`, `top_k=3`, `dropout=0.3`. Bound to `args.timesnet` (`config.py:733`).
- Effective shared defaults from `SL` (`config.py:306-376`): `num_enc_layers=3`, `dropout=0.05`, `revin=revout=revin_affine=True`, `eps_revin=1e-5`, `return_head=False`. `task` defaults to `"classification"` (`config.py:115-118`). So in practice the model runs with `d_model=16`, `d_ff=128`, `num_kernels=6`, `top_k=3`, 3 encoder layers, RevIN on.
- Submodules built in `__init__`: `self.backbone` = `nn.ModuleList` of `num_enc_layers` `TimesBlock`s (`timesnet.py:54-66`), `self.enc_embedding = DataEmbedding(...)` with `max_len = max(seq_len, 5000)` (`timesnet.py:69-72`), `self.layer_norm = nn.LayerNorm(d_model)`, and task-specific heads (`timesnet.py:78-84`): forecasting builds `self.predict_linear = nn.Linear(seq_len, pred_len + seq_len)` and `self.head = nn.Linear(d_model, c_out)`. Classification builds `self.act = F.gelu`, `self.dropout`, and `self.head = nn.Linear(d_model * seq_len, pred_len)` (here `pred_len` is repurposed as the **number of classes**). For classification, `self.pred_len` is forced to 0 (`timesnet.py:40`).

**How it runs (the forward pass).** The input convention is **`(B, num_channels, seq_len)`** (channels-second), confirmed by the `forecast` docstring and `__main__` test (`timesnet.py:95`, `timesnet.py:198`).

Forecasting path (`forecast`, `timesnet.py:92-124`):
1. Optional **RevIN normalize** on `(B, C, L)` (`timesnet.py:99-100`).
2. **Embedding:** `x.permute(0,2,1)` to `(B, L, C)`, then `enc_embedding` to `(B, L, d_model)` (`timesnet.py:103-105`).
3. **Temporal extension:** `predict_linear` is applied over the time axis. `predict_linear(x.permute(0,2,1)).permute(0,2,1)` maps `L` to `L+pred_len`, giving `(B, L+H, d_model)` (`timesnet.py:106-108`). This is TimesNet's trick of padding the horizon before the backbone rather than after.
4. **Backbone:** for each of `num_enc_layers` blocks, `x = layer_norm(backbone[i](x))`, shape stays `(B, L+H, d_model)` (`timesnet.py:111-112`).
5. **Head:** `head` (Linear `d_model` to `c_out`) to `(B, L+H, c_out)` (`timesnet.py:115-116`).
6. Optional **RevIN denormalize:** `revin(out.permute(0,2,1), mode="denorm")` to `(B, c_out, L+H)` (`timesnet.py:119-122`).
7. `forward` then **slices the last `pred_len` steps:** `forecast(x)[:, :, -pred_len:]` to `(B, c_out, H)` (`timesnet.py:153-155`).

Classification path (`classification`, `timesnet.py:126-149`): RevIN norm, embed (`enc_embedding(x.permute(0,2,1), None)`), backbone plus layernorm, `GELU`, dropout, flatten to `(B, L*d_model)`, `head`, to `(B, num_classes)`.

Inside one **TimesBlock** (`emforecaster/layers/timesnet/timesblock.py:37-97`), input `(B, T, N)` where `T=L+H`, `N=d_model`:
- `compute_periods(x, k)` (`timesblock.py:9-34`) takes `rfft` over the time axis, averages amplitudes across batch and channels, zeros the DC component, takes the `top_k` frequencies, and returns `periods = T // top_freqs` plus per-sample amplitude weights.
- For each period, **zero-pad** the time axis up to a multiple of the period if needed (`timesblock.py:67-75`), then **reshape to 2D:** `(B, T//period, period, N)` permuted to `(B, N, T//period, period)` (`timesblock.py:78-82`).
- Run `self.conv` = `InceptionBlock(d_model to d_ff) -> GELU -> InceptionBlock(d_ff to d_model)` (`timesblock.py:52-56`), reshape back to `(B, T, N)`, and crop to `seq_len+pred_len`.
- **Adaptive aggregation:** stack the `top_k` reconstructions, `softmax` the FFT amplitudes into weights, weighted-sum (`timesblock.py:88-93`), then add the **residual** `res + x` (`timesblock.py:96`).

**Custom layers.**
- `emforecaster/layers/timesnet/timesblock.py`: `TimesBlock` and the FFT period-finder `compute_periods`. This is the heart of the model.
- `emforecaster/layers/timesnet/inception_block.py`: `InceptionBlock` (v1), `num_kernels` parallel `nn.Conv2d` layers with kernel sizes `2*i+1` and matching padding, Kaiming-init, outputs averaged across kernels (`inception_block.py:8-39`). (The same file also defines unused `InceptionBlock1D`/`InceptionNet1D` helpers. TimesNet uses only the 2D `InceptionBlock`.)
- `emforecaster/layers/timesnet/embed.py`: `DataEmbedding` (`embed.py:119-143`) = `TokenEmbedding` (a circular-padded `Conv1d` value projection) plus `PositionalEmbedding` (sinusoidal) plus optional temporal embedding. In this repo it is called with `x_mark=None`, so only value plus positional embeddings are used.
- `emforecaster/layers/patchtst/revin.py`: `RevIN` (Kim et al. 2022 reversible instance norm), shared with the PatchTST family, operates on `(B, C, L)`.

**Notable details / gotchas.**
- **Input is channels-second `(B, C, L)`** and the model permutes internally. Do not pass `(B, L, C)`.
- **RevIN** is the only instance-norm here. There is **no series decomposition and no patching** (unlike DLinear / PatchTST). RevIN denorm is gated on the separate `revout` flag, while `revin` gates the norm. Both default True through `SL`.
- **`c_out` vs `num_channels`:** the forecasting output channel count is `c_out` (default 1), independent of the input `num_channels`, so a multivariate input can be mapped to a single-variable forecast.
- **`pred_len` is overloaded:** it is the horizon for forecasting but the **number of classes** for classification (`head = Linear(d_model*seq_len, pred_len)`, `timesnet.py:84`). `self.pred_len` is internally set to 0 for classification (`timesnet.py:40`).
- **Config shadowing:** the factory reads `num_enc_layers` and `dropout` from `args.sl`, not from `args.timesnet`, so the `TimesNet` config class's own `num_enc_layers`/`dropout` defaults (2 / 0.3) are not what actually runs (3 / 0.05).
- **`return_head=False` default** (from `SL`): if `return_head` is False, `forecast` never assigns `out` and would raise a `NameError` (`out = self.head(x)` is inside `if self.return_head:` at `timesnet.py:115-116`, but `out` is returned/denormed afterward). In practice this model must be run with `return_head=True` for forecasting to work end-to-end. Worth flagging as a latent bug when using the bare `SL` default.
- **`predict_linear` operates in embedding space** (it extends the time axis after embedding, before the conv backbone), which is the published TimesNet design and differs from "predict from a pooled representation" baselines.

### ModernTCN

**What it is.** A faithful port of **ModernTCN** (Luo & Wang, ICLR 2024, "ModernTCN: A Modern Pure Convolution Structure for General Time Series Analysis", OpenReview `vpJMJerXHU`). Both source files carry the comment `# Taken from: https://github.com/luodhhh/ModernTCN/tree/main/ModernTCN-classification/models`, and the docstring at `emforecaster/models/modern_tcn.py:40` cites `https://openreview.net/forum?id=vpJMJerXHU`. ModernTCN is a **pure-convolutional (CNN-TCN family)** architecture that argues a modernized ConvNet can match Transformers on time series. Its core mechanism replaces self-attention with:
- a **re-parameterized large-kernel depthwise convolution** (`ReparamLargeKernelConv`, `backbone.py:89`) that mixes information across time, structurally re-parameterized RepLKNet-style from a large kernel plus a parallel small kernel that can be fused at inference (`merge_kernel`, `backbone.py:180`).
- two **grouped 1x1 "ConvFFN" blocks** (`backbone.py:225-268`). `convffn1` mixes across the feature (`d_model`) dimension per variable, and `convffn2` mixes across the variable (`M`) dimension, giving cheap channel- and variable-mixing without attention.

This is specifically the **classification** variant of the upstream repo. The backbone's `forward` only implements the classification path (`backbone.py:501-504`), and the model docstring states "This is configured for classification purposes" (`modern_tcn.py:42`).

**Role in this repo.** A **baseline** forecasting/classification model (the flagship is EMForecaster). It is registered in the SL training loop alongside the other baselines in `forward_pass`, `compute_loss`, and `model_update` (`emforecaster/utils/models.py:340`, `:403`, `:430`). Notably, although the code is the classification variant, **this repo repurposes it for forecasting**: the factory at `models.py:131` passes `num_classes=args.data.pred_len`, so the classification `nn.Linear` head (`backbone.py:464`) emits `pred_len` values per sample. The default `args.exp.task` is `"classification"` (`config.py:115-118`), which is also the only branch the backbone's `forward` handles. For any other task string the backbone returns the input unchanged (`backbone.py:501-504`), a real gotcha.

**How it's built.** The outer class `ModernTCN(nn.Module)` (`modern_tcn.py:6`) is a thin wrapper that forwards all args into `ModernTCNBackbone` (`backbone.py:340`) and exposes `self.head_in_dim`. Construction is dispatched in `make_model`/`get_model` when `args.exp.model_id == "ModernTCN"` (`models.py:125-145`). Constructor args and their config sources:
- `seq_len=args.data.seq_len`, `pred_len=args.data.pred_len` (the latter becomes `target_window` and `num_classes`).
- `patch_dim=args.data.patch_dim`, `patch_stride=args.data.patch_stride`, the patch (kernel) size and stride of the Conv1d stem.
- `num_channels=args.data.num_channels` to `nvars`/`c_in`.
- `task=args.exp.task`, `return_head=args.sl.return_head`, `dropout=args.sl.dropout` (backbone dropout), `class_dropout=args.moderntcn.class_dropout`.
- `ffn_ratio=args.moderntcn.ffn_ratio`, `num_enc_layers=args.moderntcn.num_enc_layers`, `large_size=args.moderntcn.large_size`, `small_size=args.moderntcn.small_size`, `d_model=args.moderntcn.d_model`, `dw_dims=args.moderntcn.dw_dims`.
- `revin=args.sl.revin`, `affine=args.sl.revin_affine`.

Config class `ModernTCN(BaseModel)` (`config.py:686`, accessed as `args.moderntcn`, registered at `config.py:734`). Defaults: `num_enc_layers=[2]`, `d_model=[16]`, `ffn_ratio=1`, `dropout=0.0`, `class_dropout=0.0`, `large_size=[9]`, `small_size=[5]`, `dw_dims=[256]`. (Note the config default `d_model=[16]`/`large_size=[9]` differs from the model docstring's suggested `[32]`/`[13]`. The lists are per-stage, length == `len(num_enc_layers)`.) Several constructor args are hard-coded "non-modifiable" in the wrapper (`modern_tcn.py:27-35`): `stem_ratio=6`, `downsample_ratio=2`, `small_kernel_merged=False`, `head_dropout=0.0`, `use_multi_scale=False`, `subtract_last=False`, `individual=False`. **No `args.yaml` for ModernTCN exists** under `emforecaster/jobs/exp/`, so the config defaults above are the only realistic hyperparameters available.

**How it runs (the forward pass).** `ModernTCN.forward(x)` just calls the backbone (`modern_tcn.py:133-135`). The backbone routes to `classification` to `forward_feature` (`backbone.py:491`, `:466`). Note the input layout: the `__main__` test feeds `x = torch.randn(B, num_channels, seq_len)`, that is, **(B, M, L)** with channels before time (not (B, L, C)).
1. `forward_feature` (`backbone.py:466`): `x` is `(B, M, L)`, `x.unsqueeze(-2)` to `(B, M, 1, L)`. (No RevIN is applied here, see gotchas.)
2. Per stage `i` (loop `backbone.py:472`): reshape `(B, M, D, N)` to `(B*M, D, N)` so every channel is processed independently (**channel-independence**).
   - Stage 0: if `patch_dim != patch_stride`, edge-pad the time axis by repeating the last step `patch_dim - patch_stride` times (`backbone.py:476-480`). The stem `nn.Sequential(Conv1d(1, dims[0], kernel_size=patch_dim, stride=patch_stride), BatchNorm1d)` (`backbone.py:385-388`) **patches** each channel: `(B*M, 1, L)` to `(B*M, d_model, patch_num)`.
   - Stages > 0: if `N % downsample_ratio != 0`, pad, then a `Conv1d(dims[i], dims[i+1], kernel_size=stride=downsample_ratio)` downsamples (`backbone.py:394-402`).
   - Reshape back to `(B, M, D_, N_)` and run `self.stages[i]` (`backbone.py:488`).
3. Each `Stage` (`backbone.py:301`) runs `num_enc_layers[i]` `Block`s. Each `Block.forward` (`backbone.py:272-298`):
   - **DWConv:** reshape to `(B, M*D, N)`, apply `ReparamLargeKernelConv` (`self.dw`, groups = `nvars*dmodel`, so depthwise) for temporal mixing, reshape to `(B*M, D, N)`, `BatchNorm1d(dmodel)`.
   - **ConvFFN1** (`groups=nvars`): 1x1 conv `nvars*dmodel` to `nvars*dff`, GELU, 1x1 conv back, dropout, mixing the `d_model` feature dim per variable.
   - **ConvFFN2** (`groups=dmodel`): permute so variables are grouped, 1x1 conv `nvars*dmodel` to `nvars*dff`, GELU, back, mixing across the `M` variable dim.
   - Residual: `x = input + x` (`backbone.py:297`).
4. `classification` (`backbone.py:491`): the `(B, M, D, N)` feature map is GELU'd (`F.gelu`), `class_dropout`'d, then if `return_head`: flattened to `(B, M*D*N)` and `head_class = nn.Linear(n_vars*head_nf, num_classes)` (`backbone.py:464`) to output **(B, num_classes)**. Since the factory sets `num_classes = pred_len`, this is effectively **(B, pred_len)**, a flat forecast over the horizon. `head_nf` is computed at `backbone.py:443-450` from `patch_num = seq_len // patch_stride` and the downsample factor.

**Custom layers.**
- `emforecaster/layers/modern_tcn/backbone.py`: the entire network. `LayerNorm`, conv/bn helpers (`get_conv1d`, `conv_bn`, `fuse_bn`), `ReparamLargeKernelConv` (large plus small kernel reparam, `merge_kernel`/`structural_reparam` for inference fusion), `Block`, `Stage`, and `ModernTCNBackbone`.
- `emforecaster/layers/modern_tcn/utils.py`: `series_decomp` plus `moving_avg` (a trend/seasonal decomposition block, **imported but never used** in the classification backbone), `Flatten_Head` (the forecasting head, built but unused on the classification path), and `RevIN` (reversible instance norm).

**Notable details / gotchas.**
- **RevIN is constructed but never applied.** The backbone builds `self.revin_layer = RevIN(...)` when `revin=True` (`backbone.py:380-381`), but `forward_feature` never calls it for norm/denorm. So despite `revin`/`affine` being wired from config, instance normalization is effectively dead code in this classification-only path. Within `Block`, normalization is `BatchNorm1d` instead (`backbone.py:222`, `:280`).
- **Forecasting through the classification head.** Forecasts come out of `head_class` (a single Linear to `pred_len`), not the `Flatten_Head` forecasting head. `Flatten_Head` is built (`backbone.py:435`/`:452`) but only `head_class` is used by `classification`.
- **Channel-independence plus per-channel patching.** The stem `Conv1d(1, dims[0], ...)` operates on each channel separately (in=1, after reshaping `(B,M,..)` to `(B*M,..)`), so all channels share one stem and are processed independently. Cross-variable mixing happens only inside ConvFFN2.
- **Input layout is (B, M, L)** (channels-first), unlike many other models here that use (B, L, C). The `forward_pass` helper feeds `batch[0]` directly (`models.py:359`), and `args.data.difference_input` may diff along `dim=1`.
- **Multi-stage** is supported through list args (`num_enc_layers`, `large_size`, `d_model`, `dw_dims` must be equal-length lists). `downsample_ratio=2` halves the temporal length each stage after the first.
- **Bug in `LayerNorm`:** `backbone.py:14` calls `nn.Layernorm` (typo, the correct name is `nn.LayerNorm`). It would raise `AttributeError` if built, but `LayerNorm` is never used in the active code path, so it never triggers. `PaddingTwoEdge1d` similarly uses `dims=` instead of `dim=` (`backbone.py:163-164`) but is only reachable through the rarely-used `merge_kernel` reparam path.
- **Structural reparameterization** (`structural_reparam`, `backbone.py:506`) can fuse each large plus small DWConv into a single equivalent kernel for faster inference, but it is not called during normal training/eval here.

### DLinear

**What it is**

DLinear is the "Decomposition-Linear" forecaster introduced in *"Are Transformers Effective for Time Series Forecasting?"* (Zeng et al., AAAI 2023). The docstring in `emforecaster/models/dlinear.py:9-10` cites the origin paper `https://arxiv.org/pdf/2205.13504.pdf` and states it was taken from the Time-Series-Library (`thuml/.../models/DLinear.py`).

The core mechanism is deliberately simple: decompose the input series into a **trend** component (moving average) and a **seasonal/remainder** component (input minus trend), pass each through a *single* linear layer that maps the lookback window directly to the forecast horizon, then add the two outputs. There is no attention, no recurrence, and no nonlinearity in the forecasting path. The model is essentially two learned linear projections over time. Its family is **Decomposition / Linear** (a decomposition front-end feeding linear heads).

**Role in this repo**

DLinear is a **baseline**, not the flagship (the flagship is the EMForecaster model with conformal prediction). The class supports both `task="forecasting"` and `task="classification"` (selected in the constructor, `dlinear.py:39-44`), but in this repo it is wired up for forecasting: the dedicated experiment `emforecaster/jobs/exp/rf_emf/dlinear/args.yaml` sets `task: "forecasting"`. Notably, DLinear's `series_decomp` block is also reused in spirit by the flagship (the config class even carries a `final_moving_avg` field "for the EMForecaster model", `config.py:642-645`), so DLinear's decomposition idea feeds the headline model.

**How it's built**

- Class `DLinear(nn.Module)` in `emforecaster/models/dlinear.py:7`. Constructor args (`dlinear.py:13-23`): `task="forecasting"`, `seq_len=512`, `pred_len=96`, `num_channels=7`, `num_classes=2`, `moving_avg=25`, `individual=False`, `return_head=True`.
- The factory `get_model` dispatches on `args.exp.model_id == "DLinear"` (`emforecaster/utils/models.py:92`) and builds it (`models.py:93-102`) with `task=args.exp.task`, `seq_len=args.data.seq_len`, `pred_len=args.data.pred_len`, `num_channels=args.data.num_channels`, `num_classes=args.data.pred_len` (note: `num_classes` is fed the pred_len here), `moving_avg=args.dlinear.moving_avg`, `individual=args.dlinear.individual`, `return_head=args.sl.return_head`.
- Config class `DLinear(BaseModel)` at `config.py:632-645`, mounted as the `dlinear` field of the top-level args (`config.py:731`). Keys: `moving_avg` (default 25), `individual` (default False, "Whether to use model channels together or separately"), and `final_moving_avg` (default 25, used by EMForecaster rather than DLinear itself).
- Realistic hyperparameters from `emforecaster/jobs/exp/rf_emf/dlinear/args.yaml`: `dlinear.moving_avg: 3`, `data.seq_len: 20`, `data.pred_len: 20`, `data.num_channels: 1` (single-variable RF-EMF), `sl.criterion: "MSE"`, `sl.lr: 1e-5`, `sl.batch_size: 512`, `sl.epochs: 100`. Since `num_channels: 1` and `individual` is not overridden, the default shared-weight path is used.
- Weight initialization is non-default: both linear layers' weights are explicitly set to `(1/seq_len) * ones([pred_len, seq_len])` (`dlinear.py:69-74`, and per-channel at `59-64`), so each output step starts as the simple average of the input window.

**How it runs (the forward pass)**

`forward(self, x_enc)` (`dlinear.py:130-141`) branches on `self.task`.

Forecasting path (`forecast` to `encoder`):
1. Input `x_enc` is `(B, num_channels, seq_len)` = `(B, C, L)`. (`forecast`, `dlinear.py:107-119`, also handles a 2D `(B, L)` input by unsqueezing to `(B, 1, L)` and asserting `seq_len == self.seq_len`, `dlinear.py:109-116`.)
2. `encoder` (`dlinear.py:79-105`) permutes to `(B, L, C)` (`x.permute(0, 2, 1)`, `dlinear.py:80-82`), then calls `self.decomposition(x)` (`series_decomp`), returning `seasonal_init, trend_init`, each `(B, L, C)`.
3. Both are permuted to `(B, C, L)` (`dlinear.py:84-86`).
4. Linear projection over the time axis (`seq_len` to `pred_len`):
   - Shared (`individual=False`, `dlinear.py:101-103`): `seasonal_output = self.Linear_Seasonal(seasonal_init)` and `trend_output = self.Linear_Trend(trend_init)`, each `(B, C, pred_len)`.
   - Per-channel (`individual=True`, `dlinear.py:87-100`): loops over channels, applying a separate `Linear_Seasonal[i]` / `Linear_Trend[i]` to each channel slice and writing into pre-allocated `(B, C, pred_len)` tensors.
5. Sum the components: `x = seasonal_output + trend_output` (`dlinear.py:104`), then permute back to `(B, pred_len, C)` (`dlinear.py:105`).
6. `forward` slices `dec_out[:, -self.pred_len:, :]` and permutes to `(B, num_channels, pred_len)` = `(B, C, H)` (`dlinear.py:133-135`).

Classification path (`classification`, `dlinear.py:121-128`): runs the same `encoder` (with `self.pred_len` set to `seq_len`, `dlinear.py:40`), and if `return_head` is True, flattens to `(B, C*seq_len)` and applies `self.projection = nn.Linear(num_channels*seq_len, num_classes)` (`dlinear.py:76-77`) to get `(B, num_classes)`. If `return_head` is False, it returns the raw encoder output (embedding) instead.

**Custom layers**

`emforecaster/layers/dlinear/series_decomp.py` defines two small modules:
- `moving_avg(nn.Module)` (`series_decomp.py:5-22`): an `nn.AvgPool1d(kernel_size, stride=1, padding=0)` trend extractor. It replicate-pads the series on both ends by `(kernel_size-1)//2` steps (front = first timestep repeated, end = last timestep repeated, `series_decomp.py:17-19`) so the output length equals the input length, then average-pools along the time axis.
- `series_decomp(nn.Module)` (`series_decomp.py:25-37`): computes `moving_mean = self.moving_avg(x)` (trend) and `res = x - moving_mean` (seasonal/remainder), returning `(res, moving_mean)`. This is the "Series decomposition block from Autoformer" referenced in `dlinear.py:46`.

**Notable details / gotchas**

- **No RevIN inside DLinear.** Although `args.yaml` sets `sl.revin: True`, `sl.revin_affine: True`, `sl.revout: True` (these are shared `sl` flags), the DLinear factory branch (`models.py:92-102`) does **not** pass `revin`/`revin_affine` to the constructor, and the `DLinear` class has no RevIN code. RevIN (`emforecaster/layers/patchtst/revin.py`) is wired into other models (for example PatchTST at `models.py:56`) but not this one. So the `revin` flags in the DLinear job are effectively inert for the model itself.
- **Padding gotcha for even `moving_avg`.** The replicate padding uses `(kernel_size-1)//2` on each side. With `kernel_size=3` (the job's value) this is 1+1=2 padding, and AvgPool stride 1 returns exactly `seq_len` outputs. For an *even* kernel the integer-divided padding (2*((k-1)//2) = k-2) plus AvgPool would yield `seq_len-1` outputs, mismatching the linear layers' `in_features=seq_len`. Odd `moving_avg` is the safe/intended choice (the test in `__main__` uses 13 for classification).
- **Channel-independence is opt-in.** With `individual=False` (the default and the job's setting), one shared linear maps all channels. `individual=True` gives each channel its own `Linear_Seasonal`/`Linear_Trend`. In the RF-EMF job the data is single-variable (`num_channels: 1`), so this distinction does not matter there.
- **Custom weight init** to a uniform-average matrix (`1/seq_len`) means the model starts as a plain moving-average forecaster and learns deviations from there.
- **`num_classes` overload in the factory.** For DLinear the factory passes `num_classes=args.data.pred_len` (`models.py:98`), which is harmless in forecasting (the classification head is never built since `task != "classification"`, `dlinear.py:76`).
- Only `forecasting` and `classification` are implemented. The constructor raises `ValueError` for any other `task` (`dlinear.py:43-44`), despite the docstring also mentioning `anomaly_detection`/`imputation`.

### TSMixer

**What it is.** A faithful re-implementation of **TSMixer** (Chen et al., 2023, "TSMixer: An All-MLP Architecture for Time Series Forecasting"), an MLP-Mixer-style model. The header comment at `emforecaster/models/tsmixer.py:6` attributes the port to the Time-Series-Library (`thuml/Time-Series-Library`, written here as `thuml/emforecaster-Library` after a project-wide rename). Its core mechanism is to **alternately mix two axes with MLPs:** a *temporal* MLP that mixes across time steps and a *channel* (feature) MLP that mixes across variables, each wrapped in a residual connection. There is no attention and no convolution. It is purely linear layers plus GELU plus dropout, which puts it in the **MLP-Mixer family**.

**Role in this repo.** A **forecasting baseline** (not the flagship). It competes against the flagship EMForecaster model and other baselines (PatchTST, DLinear, TimesNet, and so on). It is registered for forecasting only. It appears in the model lists at `emforecaster/utils/models.py:336`, `:399`, and `:426`. The matching experiment (`emforecaster/jobs/exp/rf_emf/tsmixer/args.yaml`) runs `task: "forecasting"`, `learning_type: "sl"`, on the single-variable `rf_emf` dataset.

**How it's built.** Two classes in `emforecaster/models/tsmixer.py`:
- `ResBlock` (`tsmixer.py:9`): one mixer block. Holds `self.temporal = Sequential(Linear(seq_len, d_model), GELU, Linear(d_model, seq_len), Dropout)` and `self.channel = Sequential(Linear(num_channels, d_model), GELU, Linear(d_model, num_channels), Dropout)`.
- `TSMixer` (`tsmixer.py:47`): the model. Constructor args: `seq_len, pred_len, num_enc_layers, d_model, num_channels, dropout=0.0, revin=True, revin_affine=True, revout=False, eps_revin=1e-5, return_head=True`. It builds `self.backbone = ModuleList([ResBlock(...) for _ in range(num_enc_layers)])` and a single linear forecasting head `self.head = nn.Linear(seq_len, pred_len)` (`tsmixer.py:81`). All `nn.Linear` weights are Xavier-uniform initialized with zero bias through `self.apply(xavier_init)` (`tsmixer.py:90`, from `emforecaster/utils/weight_init.py:48`).

**Construction / config.** The factory dispatches at `emforecaster/utils/models.py:185` (`elif args.exp.model_id == "TSMixer":`), building `TSMixer` with `seq_len/pred_len/num_channels` from `args.data` and `num_enc_layers/d_model/dropout/revin/revin_affine/revout/eps_revin` from `args.sl` (`models.py:186-197`). The config class is `TSMixer(BaseModel)` at `emforecaster/config/config.py:648`, registered as the field `tsmixer: TSMixer = TSMixer()` (`config.py:732`). Its own fields default to `num_enc_layers=2`, `dropout=0.3`, `d_model=16`. The RevIN knobs (`revin=True`, `revout=True`, `revin_affine=True`, `eps_revin=1e-5`) are inherited from `BaseModel` (`config.py:341-353`). The `rf_emf` `args.yaml` overrides these to `d_model: 32`, `num_enc_layers: 1`, `dropout: 0.3`, `seq_len: 20`, `pred_len: 20`, `num_channels: 1`, with `revin/revin_affine/revout` all `True`.

**How it runs (the forward pass).** `forward(x)` at `tsmixer.py:98`, with input `x` shaped `(B, C, L) = (batch, num_channels, seq_len)` (see the test block at `tsmixer.py:136`, `torch.rand(32, 7, 512)`):
1. **RevIN norm** (`tsmixer.py:101-102`): if RevIN is enabled, `x = self.revin(x, mode="norm")`. RevIN computes mean/std over `dim=2` (the time axis) and normalizes, keeping shape `(B, C, L)`.
2. **Permute** (`tsmixer.py:104-106`): `x = x.permute(0, 2, 1)` to `(B, L, C)`, the layout the ResBlocks expect.
3. **Mixer backbone** (`tsmixer.py:107-108`): loop over `num_enc_layers` ResBlocks. Inside each `ResBlock.forward` (`tsmixer.py:33`): `x = x + self.temporal(x.transpose(1, 2)).transpose(1, 2)`, transposing to `(B, C, L)` so the temporal MLP mixes across the `L` time steps, then adding residually. Then `x = x + self.channel(x)`, the channel MLP mixing across the `C` features on `(B, L, C)`, added residually. Shape stays `(B, L, C)`.
4. **Head** (`tsmixer.py:110-113`): if `return_head` (default True), `out = self.head(x.transpose(1, 2))`, transposing to `(B, C, L)` and applying `Linear(seq_len to pred_len)` along the last (time) axis, giving `out` of shape `(B, C, H) = (batch, num_channels, pred_len)`. If `return_head=False`, it returns the transposed features `(B, C, L)` as an embedding (no head).
5. **RevIN denorm** (`tsmixer.py:116-117`): if `revout`, `out = self.revin(out, mode="denorm")` re-applies the stored mean/std (and inverse affine), restoring the original scale. Final output: `(B, C, pred_len)`.

**Custom layers.**
- `RevIN` (`emforecaster/layers/patchtst/revin.py:5`). Reversible instance normalization (Kim et al., 2022). Expects `(B, C, L)`, computes detached per-instance, per-channel mean and `sqrt(var + eps)` over the time axis `dim=2` (`revin.py:49-53`), normalizes, and applies optional learnable per-channel affine (`affine_weight`, `affine_bias`). `mode="denorm"` inverts it. Note the denorm divides by `affine_weight + eps*eps` (`revin.py:71`), a slightly different epsilon than the forward path.
- `xavier_init` (`emforecaster/utils/weight_init.py:48`). A module-wise initializer applied through `model.apply(...)`. Xavier-uniform for Linear weights (zero bias), with LayerNorm/LSTM branches that are unused here.

**Notable details / gotchas.**
- **Channel mixing (NOT channel-independent).** Unlike PatchTST/DLinear-style channel-independence, the `channel` MLP explicitly mixes across the `num_channels` features. With the repo's `rf_emf` config (`univariate: True`, `num_channels: 1`) the channel MLP degenerates to a `1 to d_model to 1` mapping, so cross-channel mixing is effectively a no-op there. Useful to know if you reuse this model on multivariate data.
- **Input layout is `(B, C, L)`, not `(B, L, C)`.** The model takes channels-first input (matching RevIN and the test harness) and permutes internally. Do not pre-permute.
- **The inline comment at `tsmixer.py:104-106` is consistent** with the code (`permute(0,2,1)` turns `(B, C, L)` into `(B, L, C)`), but note the ResBlock docstring describes its own input as `(batch_size, seq_len, num_vars)`, that is, `(B, L, C)`. That is the post-permute layout, so the two descriptions agree.
- **RevIN is on by default** (config defaults plus the experiment yaml all set `revin/revout/revin_affine=True`). Statistics are stored on the module during `norm` and reused at `denorm`, so forward order matters.
- **No patching, no series decomposition.** TSMixer here is a plain time/channel MLP-Mixer. It does not import `series_decomp` or any patcher.
- **Architecturally light.** With `num_enc_layers=1` and small `d_model` (16 to 32), this is a deliberately compact baseline. The only forecasting projection is the single `Linear(seq_len, pred_len)` head.

### Mamba

**What it is.** This is a wrapper around the **Mamba selective state-space model (SSM)** of Gu & Dao, *"Mamba: Linear-Time Sequence Modeling with Selective State Spaces"* (2023). The actual SSM kernel is **not** implemented in this repo. It is imported from the external **`mambapy`** package (pinned `mambapy==1.2.0` in `pyproject.toml:27` and `requirements.txt:21`), through `from mambapy.mamba import Mamba as MambaBackbone` and `from mambapy.mamba import MambaConfig` (`emforecaster/models/mamba.py:3-4`, `emforecaster/models/recurrent.py:5-6`). `mambapy` is a pure-PyTorch reimplementation that "closely follows the mamba_simple.py from the official Mamba implementation, and the mamba-minimal by @johnma2006" (its module docstring), doing the 1-D conv with `torch.nn.Conv1d` and the **selective scan in PyTorch** through a parallel scan (`from mambapy.pscan import pscan`). The core mechanism is the Mamba block: expand `x` to `(B, L, 2*ED)` (expand factor E=2), split into a content and a gate stream, apply a short causal `d_conv=4` 1-D conv plus SiLU, run the input-dependent (selective) SSM with state size `d_state=16`, then gate by `SiLU(z)`. A full backbone is `n_layers` of `ResidualBlock` where `ResidualBlock(x) = MambaBlock(RMSNorm(x)) + x` (see `mambapy/mamba.py:20-29, 99-112`). Family: **SSM** (a linear-time recurrent/selective-scan sequence model), not attention-based.

**Role in this repo.** A forecasting and classification baseline. EMForecaster's flagship is the `EMForecaster` model plus conformal prediction. Mamba is one of several recurrent/sequence **baselines** selectable through the unified `RecurrentModel` class (alongside `LSTM`, `RNN`, `GRU`). The patched mode targets EMF time-series forecasting `(B, M, L)` to `(B, M, pred_len)`. The non-patched mode targets single-variable variable-length classification (the code comments reference SOZ localization, `pred_len=1` for binary). The two `__main__` blocks demonstrate both modes.

**How it's built, IMPORTANT routing caveat.** There are two classes here, and only one is actually wired:
- `class RecurrentModel(nn.Module)` in `emforecaster/models/recurrent.py:16`. **This is the one the factory uses.** In `emforecaster/utils/models.py:64-85`, `get_model` dispatches when `args.exp.model_id == "RecurrentModel"`, building `RecurrentModel(...)` and passing `backbone_id=args.exp.backbone_id`. Inside the constructor, `backbone_id == "Mamba"` builds `MambaConfig(d_model=d_model, n_layers=num_enc_layers)` then `self.backbone = MambaBackbone(config)` (`recurrent.py:127-132`). The valid options are documented as `'LSTM', 'RNN', 'GRU', 'Mamba'`.
- `class Mamba(nn.Module)` in `emforecaster/models/mamba.py:11`. This is a Mamba-only standalone version with the same patching/RevIN/head scaffolding. **A grep shows it is never imported or instantiated anywhere** (no `from emforecaster.models.mamba import` and no `Mamba(` construction outside the file itself). It appears to be an earlier/dead-code variant that was generalized into `RecurrentModel`. The model behavior is documented from `mamba.py`, but note that in this codebase Mamba is reached through `RecurrentModel` plus `backbone_id="Mamba"`, so the live hyperparameter surface is `RecurrentModel`'s.

Config class and keys (`emforecaster/config/config.py`):
- `Experiment.model_id` (`config.py:25`, default `"PatchTST"`) must be `"RecurrentModel"`.
- `Experiment.backbone_id` (`config.py:29`, default `"LSTM"`) must be `"Mamba"`. Its description literally lists `'LSTM', 'RNN', 'GRU', 'Mamba'`.
- Supervised-learning block `SL` (`config.py:306`): `d_model` (default 128, `config.py:318`), `num_enc_layers` (default 3, becomes Mamba `n_layers`, `config.py:315`), `dropout` (default 0.05), `head_type` (`"linear"`/`"mlp"`, default `"linear"`), `norm_mode` (default `"batch1d"`, `"layer"` enables the final `LayerNorm`), `revin`/`revout`/`revin_affine` (all default True), `eps_revin` (1e-5), `bidirectional` (default False, **note Mamba ignores this**, since `compute_backbone` never doubles dims for Mamba even though `head`/`norm` would size for `2*d_model` if it were True), `last_state` (default True), `avg_state` (default False).
- Data block: `seq_len`, `pred_len`, `num_channels`, `patching` (default False, `config.py:195`, commented "LSTM only"), `patch_dim` (16), `patch_stride` (8).

Mamba-specific defaults come from `mambapy.MambaConfig` (`mambapy/mamba.py:31-67`). Only `d_model` and `n_layers` are set by this repo. The rest stay at library defaults: `d_state=16`, `expand_factor=2`, `d_conv=4`, `dt_rank='auto'` (= `ceil(d_model/16)`), `dt_min=0.001`, `dt_max=0.1`, RMSNorm, `pscan=True`, `use_cuda=False`.

No Mamba-specific `args.yaml` exists under `emforecaster/jobs/exp/**` (grep found `mamba`/`backbone_id`/`RecurrentModel` only in EMForecaster/test yamls, not as a Mamba run). The closest realistic recurrent-baseline numbers are in `jobs/exp/rf_emf/baselines/lstm/args.yaml`, but that file actually drives a different class (`model_id: "RF_EMF_LSTM"` to `rf_emf_baselines.LSTM`, not `RecurrentModel`). It is useful for ballpark hyperparameters only (`seq_len: 336`, `pred_len: 96`, `num_channels: 1`, `d_model: 32`, `epochs: 100`, `batch_size: 512`, `lr: 1e-4`).

**How it runs (the forward pass).** Stepping through `RecurrentModel.forward` (`recurrent.py:188-252`), with legend `B`=batch, `M`=channels, `L`=seq_len, `N`=num_patches, `P`=patch_dim, `D`=d_model:
1. Shape fix: a 2-D `(B, L)` input is unsqueezed to `(B, 1, L)` if patching else `(B, L, 1)` (`recurrent.py:207-210`).
2. RevIN norm (if enabled): `self.revin(x, mode="norm")`, instance-normalizing per channel over the time axis (`recurrent.py:213-216`).
3. Patching (forecasting path, fixed length only): `Patcher` to `(B, M, N, P)` then `PositionalEncoding` projects `P` to `D` and adds a learned positional matrix to `(B, M, N, D)`, reshaped to `(B*M, N, D)` for channel-independent processing (`recurrent.py:219-223`). `num_patches = int((seq_len - patch_dim)/patch_stride) + 2`.
4. Backbone: `compute_backbone` (`recurrent.py:171-186`). For Mamba it calls `out = self.backbone(x)` returning `(B*M, N, D)` (patched) or `(B, L, D)` (non-patched), and sets `last_hn = out[-1]`. **Gotcha:** for LSTM/RNN/GRU `last_hn` is the true final hidden state `hn[-1]`, but for Mamba there is no hidden-state tuple, so the code uses `out[-1]`, indexing the **first dimension (batch)**, that is, the last *sample in the batch*, not the last time step. So `last_state=True` is semantically wrong for the Mamba backbone. Mamba runs should use `last_state=False` with `avg_state=True` (which is exactly what the `recurrent.py` `__main__` example does, `recurrent.py:322-323`).
5. Pooling plus final norm (`recurrent.py:229-238`): `last_state` gives `norm(last_hn)`, `avg_state` gives `norm(mean(out, dim=1))` averaging over the sequence (recommended for Mamba), else `norm(out)` keeps the full sequence. `norm` is `LayerNorm(d_model)` when `norm_mode=="layer"`, else `Identity`.
6. Patch reshape (`recurrent.py:241-242`): `avg_state` gives `(B, M, D)`, non-avg gives `(B, M, N*D)`.
7. Head (`recurrent.py:245-246`): `Dropout` then `Linear`/`MLP` of size `head_dim` to `pred_len`. `head_dim = num_patches*d_model` (patched, non-avg) or `d_model` (avg/non-patched), doubled if `bidirectional`.
8. RevIN denorm (if `revout`): `self.revin(x, mode="denorm")` adds back per-channel mean/std (`recurrent.py:249-250`). Output `(B, M, pred_len)` (forecasting) or `(B, pred_len)` (classification).

The standalone `Mamba.forward` (`mamba.py:84-142`) is simpler: it always takes the last *time* step (`x[:, -1, :]`, `mamba.py:128-129`) when `last_state and not patching` (note it indexes the time dimension, unlike `RecurrentModel`'s `out[-1]`), and uses no dropout before the head and no `avg_state` option.

**Custom layers / dependencies.**
- `emforecaster/layers/patcher.py`: `Patcher`, pads the series by repeating the last value `stride` times, then `unfold`s the time axis into overlapping patches `(B, M, N, P)`. (`VerticalPatcher` also lives here but is unused by Mamba.)
- `emforecaster/layers/patchtst/pos_enc.py`: `PositionalEncoding`, a `Linear(patch_dim to d_model)` projection plus a learnable `(num_patches, d_model)` positional parameter (xavier/uniform init), applied per channel. In `RecurrentModel`, if `avg_state` is set, this is swapped for a bare `nn.Linear(patch_dim, d_model)` (no positional embedding), since position is irrelevant when averaging (`recurrent.py:91-95`).
- `emforecaster/layers/patchtst/revin.py`: `RevIN` (Kim et al. 2022, reversible instance norm), detached per-channel mean/var normalization with optional learnable affine, reversible at the output.
- `emforecaster/utils/weight_init.py`: `xavier_init`, applied through `self.apply(...)`. Xavier-uniform for `Linear`, ones/zeros for `LayerNorm`, xavier/orthogonal for `LSTM`. It does **not** touch Mamba's internal parameters (no Mamba-specific branch), so the SSM keeps `mambapy`'s own initialization (for example `dt` init, A_log, RMSNorm).
- `mambapy.mamba` (external): provides `Mamba`/`MambaConfig`/`ResidualBlock`/`MambaBlock`/`RMSNorm` and the PyTorch parallel-scan `pscan`.

**Notable details / gotchas.**
- **The standalone `mamba.py` `Mamba` class is dead code** in this repo (never imported/instantiated). Live Mamba = `RecurrentModel` plus `backbone_id="Mamba"`.
- **`last_state` is unsafe with Mamba** because `compute_backbone` sets `last_hn = out[-1]` (the last *batch element*, not the last time step). Use `avg_state=True` (and `last_state=False`) for Mamba, as the in-file example does.
- **Channel independence:** patches are reshaped to `(B*M, N, D)`, so each channel is processed independently by one shared Mamba backbone, then recombined.
- **RevIN/RevOUT** provide instance normalization against distribution shift. **Patching only works for fixed `seq_len`** (comment at `mamba.py:47` / `recurrent.py:87`). The non-patched mode supports variable, padded sequence lengths for classification.
- **`bidirectional`** is meaningful only for the `nn.LSTM/RNN/GRU` backbones. For Mamba it is ignored at the backbone (Mamba is causal/unidirectional) yet would still inflate `head_dim`/`norm_dim` sizing if set True, so leave it False for Mamba.
- **`d_state`, `expand_factor`, `d_conv`** are not exposed through this repo's config. Only `d_model` (to `D`) and `num_enc_layers` (to `n_layers`) are configurable. Everything else uses `mambapy` defaults.
- mambapy runs the selective scan in pure PyTorch (`pscan`, `use_cuda=False`), so this is the CUDA-free reference Mamba, not the fused official kernel.

### Recurrent (RNN / LSTM / GRU)

**File:** `emforecaster/models/recurrent.py` (single class `RecurrentModel`, defined at `recurrent.py:16`). This is one polymorphic wrapper, not four separate classes. The recurrent cell type is chosen by a string argument `backbone_id`.

#### What it is
`RecurrentModel` is a generic sequence-model backbone wrapper. Per its own docstring (`recurrent.py:42-65`): *"A Recurrent Neural Network (RNN) class that host a variety of different recurrent architectures including LSTM, Mamba, GRU, and the classic RNN."* It does not implement a novel recurrent cell. It builds PyTorch's stock cells:
- `"LSTM"` to `nn.LSTM` (`recurrent.py:101`), Hochreiter & Schmidhuber long short-term memory.
- `"RNN"` to `nn.RNN` (`recurrent.py:110`), the classic Elman recurrent network.
- `"GRU"` to `nn.GRU` (`recurrent.py:119`), Cho et al. gated recurrent unit.
- `"Mamba"` to `MambaBackbone` from the third-party `mambapy` package (`recurrent.py:5`, `127-132`), a selective state-space model (SSM). Despite the family being recurrent, Mamba is included here as a drop-in backbone.

Any other `backbone_id` raises `ValueError("Invalid backbone_id. Options: 'LSTM', 'RNN', 'GRU', 'Mamba'.")` (`recurrent.py:133-136`). **Family: RNN** (with an SSM option). The novelty in this file is not the cell but the surrounding pipeline (RevIN plus patching plus positional encoding plus flexible state-pooling head), which mirrors the PatchTST-style preprocessing used elsewhere in the repo.

#### Role in this repo
Dual-purpose, and a **baseline** (the repo flagship is `EMForecaster`). The `forward` docstring (`recurrent.py:188-204`) documents two modes:
- **Patched forecasting:** single-variable/multivariate time-series forecasting, input `(B, M, L)`, output `(B, M, pred_len)`.
- **Non-patched classification:** single-variable variable-length classification (the docstring names "SOZ localization", seizure-onset-zone, a holdover from the codebase's origins, Neptune project `xmootoo/soz-localization` per `config.py:57`), input `(B, L, 1)` with variable, padded `L`, output `(B, pred_len)` with `pred_len=1` for binary classification. The bidirectional/last_state/avg_state knobs are described in config as "typically for classification" (`config.py:393-404`).

Note: the related but distinct `RF_EMF_LSTM` (`emforecaster/models/rf_emf_baselines.py`, dispatched at `models.py:211`) is a *separate* LSTM baseline. The `jobs/exp/rf_emf/baselines/lstm/args.yaml` job uses `RF_EMF_LSTM`, not this class.

#### How it's built
- **Class:** `RecurrentModel(nn.Module)`.
- **Constructor args** (`recurrent.py:17-39`): `d_model`, `num_enc_layers`, `pred_len`, `backbone_id` (required), then `bidirectional=False`, `dropout=0.0`, `seq_len=512`, `patching=False`, `patch_dim=16`, `patch_stride=8`, `num_channels=1`, `head_type="linear"`, `norm_mode="layer"`, `revin=False`, `revout=False`, `revin_affine=False`, `eps_revin=1e-5`, `last_state=True`, `avg_state=False`, `return_head=True`.
- **Factory dispatch:** `emforecaster/utils/models.py:64-85`, branch `elif args.exp.model_id == "RecurrentModel":`. The mapping of config keys to constructor args:
  - `backbone_id=args.exp.backbone_id`
  - `d_model=args.sl.d_model`, `num_enc_layers=args.sl.num_enc_layers`, `dropout=args.sl.dropout`, `bidirectional=args.sl.bidirectional`, `head_type=args.sl.head_type`, `norm_mode=args.sl.norm_mode`, `last_state=args.sl.last_state`, `avg_state=args.sl.avg_state`
  - `revin/revout/revin_affine/eps_revin` from `args.sl.*`
  - `pred_len=args.data.pred_len`, `seq_len=args.data.seq_len`, `num_channels=args.data.num_channels`, `patching=args.data.patching`, `patch_dim=args.data.patch_dim`, `patch_stride=args.data.patch_stride`
  - **Gotcha:** the factory does **not** pass `return_head`, so it keeps the constructor default `True` (the head always runs) regardless of `args.sl.return_head` (whose config default is `False`, `config.py:374-376`).
- **Config classes/keys:** `backbone_id` lives on the `Experiment` config (`config.py:29-32`, default `"LSTM"`, options `'LSTM','RNN','GRU','Mamba'`). `bidirectional`/`last_state`/`avg_state` live in the supervised-learning (`sl`) config under a "# Recurrent Model" header (`config.py:392-404`, defaults `False`/`True`/`False`). Shared `sl` defaults: `d_model=128` (`config.py:318`), `num_enc_layers=3` (`config.py:315`), `head_type="linear"` (`config.py:369`), `norm_mode` default `"batch1d"` in config (`config.py:336`), but only `"layer"` produces a real norm in this model (see gotchas). `data` defaults: `patching=False` (`config.py:195`), `patch_dim=16` (`config.py:198`).
- **No dedicated job:** a repo-wide search found **no `args.yaml` with `model_id: RecurrentModel`**. The only file referencing `backbone_id` for this style is `jobs/exp/test/args.yaml`, but its `model_id` is `EMForecaster`. So there are no committed "realistic" recurrent hyperparameters. The `__main__` smoke test at `recurrent.py:294-334` uses `backbone_id="Mamba"`, `d_model=128`, `num_enc_layers=5`, `patch_dim=64`, `patch_stride=16`, `seq_len=16031`, `patching=True`, `avg_state=True`, `revin/revout=True`.

#### How it runs (the forward pass)
`forward(self, x)` (`recurrent.py:188-252`). Legend: `B`=batch, `M`=channels, `L`=seq_len, `N`=num_patches, `P`=patch_dim, `D`=d_model.

1. **Shape coercion** (`recurrent.py:207-210`): if `x` is 2-D `(B, L)`, unsqueeze to `(B, L, 1)` (or `(B, 1, L)` when patching).
2. **RevIN normalize** (`recurrent.py:213-216`): if `self._revin`, `x = self.revin(x, mode="norm")` (instance-wise mean/std normalization with optional affine).
3. **Patching path** (`recurrent.py:219-223`, only if `patching=True`):
   - `Patcher` to `(B, M, N, P)` through `unfold` with last-value padding.
   - `pos_enc` to `(B, M, N, D)`. This is a plain `nn.Linear(patch_dim, d_model)` when `avg_state=True`, else a `PositionalEncoding` (linear projection **plus** a learnable `(N, D)` positional table), chosen at `recurrent.py:91-95`.
   - Reshape `(B, M, N, D)` to `(B*M, N, D)` (**channel-independent**, each channel is a separate sequence in the batch).
   - The recurrent `input_size` is `d_model` when patching, else `num_channels` (`recurrent.py:77`).
4. **Backbone** (`compute_backbone`, `recurrent.py:171-186`):
   - RNN/GRU: `out, hn = backbone(x)`, `last_hn = hn[-1]`.
   - LSTM: `out, (hn, _) = backbone(x)`, `last_hn = hn[-1]`.
   - Mamba: `out = backbone(x)`, `last_hn = out[-1]` (note: indexes the *first* dim of the output tensor, see gotchas).
   - `out` is the full sequence `(., T, D)`. `last_hn` is the final-layer hidden state.
5. **State pooling plus norm** (`recurrent.py:229-238`): `last_state` gives `norm(last_hn)` `(B*M or B, D)`, elif `avg_state` gives `norm(mean(out, dim=1))` (mean over time), else `norm(out)` (full sequence kept).
6. **Un-patch reshape** (`recurrent.py:241-242`): if patching, `x.view(B, M, -1)` to `(B, M, D)` for avg_state or `(B, M, N*D)` otherwise.
7. **Head** (`recurrent.py:245-246`): if `return_head`, `x = self.head(self.dropout(x))`. The head is `nn.Linear(head_dim, pred_len)` or an MLP (`Linear -> GELU -> Linear`). `head_dim = num_patches*d_model` for non-avg patched, else `d_model`, doubled if `bidirectional` (`recurrent.py:141-146`).
8. **RevIN denormalize** (`recurrent.py:249-250`): if `revout`, `x = self.revin(x, mode="denorm")`.

Output: `(B, M, pred_len)` (patched forecasting) or `(B, pred_len)` (unpatched).

#### Custom layers (dependencies)
- `emforecaster/layers/patcher.py`: `Patcher`, `unfold`-based patch extraction `(B,M,L)` to `(B,M,N,P)`, with `stride`-length last-value right padding, `N = (L - patch_dim)//stride + 2`. (A `VerticalPatcher` also exists in the file but is unused here.)
- `emforecaster/layers/patchtst/pos_enc.py`: `PositionalEncoding`, `Linear(patch_dim to d_model)` plus a learnable `nn.Parameter` positional table of shape `(num_patches, d_model)` (init `uniform_(-0.02, 0.02)`), applied channel-independently.
- `emforecaster/layers/patchtst/revin.py`: `RevIN` (Kim et al. 2022, reversible instance normalization), per-instance mean/std normalize with optional learnable affine and a reversible denorm.
- `emforecaster/utils/weight_init.py`: `xavier_init`, applied through `self.apply(xavier_init)` (`recurrent.py:163`). Xavier-uniform for `Linear`, ones/zeros for `LayerNorm`, and for `LSTM` specifically Xavier on input weights, **orthogonal** on hidden weights, zeros on biases (RNN/GRU/Mamba weights are not specially initialized).

#### Notable details / gotchas
- **`norm_mode` is effectively layer-or-nothing.** The final norm is built as `nn.LayerNorm(norm_dim) if norm_mode == "layer" else nn.Identity()` (`recurrent.py:159-160`). The config default for `args.sl.norm_mode` is `"batch1d"` (`config.py:336`), which here silently becomes `nn.Identity`. You must set `norm_mode="layer"` to get any normalization.
- **`last_state` vs `avg_state` precedence:** `last_state` is checked first (`recurrent.py:229`). Both default arrangements can coexist. If `last_state=True`, `avg_state` is ignored. With patching plus `last_state`, `pos_enc` is a full `PositionalEncoding`, but with `avg_state` it degrades to a bare `Linear` (no positional table), chosen at `recurrent.py:91-95`.
- **Mamba `last_hn = out[-1]` is suspicious.** For nn.RNN/LSTM/GRU, `hn[-1]` selects the last *layer*'s hidden state correctly. For Mamba, `out` is `(B, T, D)`, so `out[-1]` indexes the **last batch element**, not a hidden state. This is likely only sensible when combined with `avg_state`/non-`last_state` modes (the `__main__` demo uses Mamba with `last_state=False, avg_state=True`). Treat `backbone_id="Mamba"` with `last_state=True` as a latent bug.
- **Channel independence:** patching collapses channels into the batch (`B*M`), so the recurrent model processes every channel as an independent single-variable sequence, then reshapes back (`recurrent.py:223`, `242`). RevIN's `num_channels` is the true `M` (`recurrent.py:169`).
- **Patching requires a fixed `seq_len`** (comment at `recurrent.py:87`) because `num_patches` is precomputed in `__init__` (`recurrent.py:69`) and drives the head dimension. Variable-length inputs are only supported in the non-patched classification path.
- **Bidirectional doubling:** when `bidirectional=True`, both `head_dim` and the LayerNorm `norm_dim` double to `2*d_model` (`recurrent.py:146`, `159`).
- **`return_head` is hard-on through the factory** (not wired from config), so the model always emits `pred_len` outputs in normal runs.
- **External dependency:** importing this module requires `mambapy` (`from mambapy.mamba import Mamba, MambaConfig`, `recurrent.py:5-6`) even if you only intend to use LSTM/RNN/GRU.

### Linear

**What it is.** `emforecaster/models/linear.py` defines a `Linear(nn.Module)` (lines 5-22) that is a *direct multi-step* (DMS) linear forecaster: a single `nn.Linear(in_features, out_features)` projects the whole input lookback window of length `in_features = seq_len` straight onto the forecast horizon of length `out_features = pred_len`, with an optional normalization layer applied first. There is no paper cited in the file itself, but this is the canonical "Linear" baseline popularized by LTSF-Linear / `Are Transformers Effective for Time Series Forecasting?` (Zeng et al., AAAI 2023), a one-layer linear map over the temporal dimension. Family: **Linear** (a single dense temporal projection, not a Transformer/CNN/RNN).

**Role in this repo.** A pure **forecasting baseline**. It is one of the simplest models in the zoo and exists for comparison against the flagship `EMForecaster`. It is registered alongside `DLinear` in every model-handling set in `emforecaster/utils/models.py`, the `forward_pass`, `compute_loss`, and `model_update` dispatch sets (lines 338, 401, 428), so it is trained through the standard supervised-learning loop. It is **not** a flagship model and has no classification path of its own (the shared loop supports binary/multi tasks, but `Linear` is configured purely as a `seq_len` to `pred_len` regressor).

**How it's built.** Class: `Linear` (`emforecaster/models/linear.py:5`). Constructor signature: `__init__(self, in_features, out_features, norm_mode="layer")`.
- Selected by the factory in `emforecaster/utils/models.py:86-91` when `args.exp.model_id == "Linear"`:
  ```python
  model = Linear(
      in_features=args.data.seq_len,
      out_features=args.data.pred_len,
      norm_mode=args.sl.norm_mode,
  )
  ```
- Config sources: there is **no dedicated `Linear` config class** in `emforecaster/config/config.py`. Its three args are pulled from existing groups:
  - `in_features` = `args.data.seq_len` (default `512`, config.py:159)
  - `out_features` = `args.data.pred_len` (config.py:163)
  - `norm_mode` = `args.sl.norm_mode` (default **`"batch1d"`**, config.py:336-339, the docstring lists `'batch1d'`, `'batch2d'`, or `'layer'`)
- The constructor builds two submodules:
  1. `self.norm`, chosen by `norm_mode`: `nn.LayerNorm(self.d_model)` if `"layer"` (line 11), `nn.BatchNorm1d(self.d_model)` if `"batch1d"` (line 13), else `nn.Identity()` (line 15).
  2. `self.linear = nn.Linear(in_features, out_features)` (line 17).

**How it runs (the forward pass).** `forward(self, x)` (lines 19-22):
1. Input `x` is expected as `(B, L, C)` with `L = seq_len`. `x.squeeze()` (line 20) removes singleton dims. With `C = 1` (the configured EMF case, `num_channels = 1`) this yields `(B, L)`.
2. `x = self.norm(x.squeeze())`, normalizing the `(B, L)` tensor (LayerNorm/BatchNorm1d over the feature/time axis, or Identity).
3. `out = self.linear(x)` (line 21), the dense map `L` to `H` giving `(B, H)` with `H = pred_len`.
4. Returns `out`, shape `(B, pred_len)`. (The shared `prepare_output_and_target` later `.squeeze()`es as needed.)

So the data flow is `(B, L, 1) --squeeze--> (B, L) --norm--> (B, L) --Linear--> (B, pred_len)`.

**Custom layers.** None. The file imports only `torch` and `torch.nn`. Everything is stock PyTorch (`nn.Linear`, `nn.LayerNorm`, `nn.BatchNorm1d`, `nn.Identity`). It does not depend on any layer files in the repo.

**Notable details / gotchas.**
- **Bug, `self.d_model` is undefined.** Lines 11 and 13 reference `self.d_model`, but `d_model` is never assigned in `__init__` (it is not even a constructor argument). So building `Linear` with `norm_mode="layer"` *or* `norm_mode="batch1d"` raises `AttributeError: 'Linear' object has no attribute 'd_model'`. Crucially, the config default fed in by the factory is `args.sl.norm_mode = "batch1d"` (config.py:337), so **building this model with default settings crashes**. Only `norm_mode` values that hit the `else` branch (for example `"none"`, or the documented-but-unhandled `"batch2d"`), that is, those that resolve to `nn.Identity()`, let the model build successfully. (Even if `d_model` were defined, the dimension is also questionable: `LayerNorm`/`BatchNorm1d` would need to match the squeezed feature size, which is `L = seq_len`, not a separate `d_model`.)
- **No RevIN / no series decomposition / no patching.** Unlike `DLinear` (trend/seasonal decomposition) and `EMForecaster` (RevIN plus patching), this plain `Linear` has no instance normalization or decomposition, only the optional `norm` layer above.
- **Effectively single-variable / channel-independence-by-squeeze.** The unconditional `x.squeeze()` assumes the channel axis is 1. With `num_channels = 1` (the EMF setup) it works, but for `C > 1` the squeeze would leave a `(B, L, C)` tensor and `self.linear` (which maps the last dim of size `L`) would mismatch. So this implementation is only correct for single-channel input.
- **No dedicated experiment config.** No `emforecaster/jobs/exp/**/args.yaml` targets `model_id: "Linear"` (the `args.yaml` files that mention "Linear" do so through `patch_model_id: "DLinear"` / DLinear configs, not this model). Realistic hyperparameters therefore fall back to the config defaults: `seq_len = 512`, `pred_len` per experiment, `norm_mode = "batch1d"` (which, per the bug above, must be overridden to a no-norm value to actually run).

### RF-EMF Baselines (CNN / MLP / Transformer / LSTM)

**What it is.** A bundle of four small, classic deep-learning forecasters defined together in `emforecaster/models/rf_emf_baselines.py`: a 2-layer `MLP`, a single-conv `CNN`, a single-layer `LSTM`, and a compact convolutional `Transformer`. There is no single "published architecture" being reproduced here and the file carries no paper citation. These are textbook reference implementations (multi-layer perceptron, 1D convolutional net, vanilla LSTM, and an encoder-only Transformer) deliberately kept tiny to serve as comparison baselines for the flagship EMForecaster. Each one is single-variable (single channel) and forecasts a fixed horizon directly. Families: MLP (Linear/MLP), CNN (CNN-TCN), LSTM (RNN), Transformer (Transformer encoder). The Transformer's `EncoderBlock` reuses the same `_MultiheadAttention` wrapper as PatchTST, so it shares lineage with the standard "Attention Is All You Need" encoder block (MHA, Add&Norm, FFN, Add&Norm).

**Role in this repo.** Pure forecasting baselines (no classification path). They are explicitly *baselines*, not the flagship. Their job files live under `emforecaster/jobs/exp/rf_emf/baselines/{cnn,mlp,lstm,transformer}/` and all share `project_name: "rf-emf-forecasting/RF-EMF"`, `dataset: rf_emf`, `univariate: True`, `num_channels: 1`, `seq_len: 336`, `pred_len: 96`. They participate in the same supervised-learning train/loss/update loop as every other model (their `model_id`s appear in the dispatch sets at `emforecaster/utils/models.py:342`, `:405`, `:432`).

**How it's built.** Four classes in `rf_emf_baselines.py`:
- `MLP(seq_len=20, pred_len=20, d_model=32)` (line 7): `fc1: Linear(seq_len, d_model)`, `Tanh`, `fc2: Linear(d_model, pred_len)`.
- `CNN(seq_len=20, num_channels=1, pred_len=20, num_kernels=32)` (line 21): `conv1: Conv1d(num_channels, num_kernels, kernel_size=seq_len)`, `Tanh`, `fc: Linear(num_kernels, pred_len)`. Note `kernel_size = seq_len`, so the conv spans the entire input window and produces exactly one output position per kernel.
- `LSTM(seq_len=20, pred_len=20, d_model=32)` (line 41): `nn.LSTM(seq_len, d_model)` then `fc: Linear(d_model, pred_len)`. The LSTM's `input_size` is `seq_len` (it does not treat the sequence as the time axis, see gotchas).
- `Transformer(seq_len=20, pred_len=20, d_model=32, num_heads=2, num_enc_layers=2)` (line 87): a `ModuleList` of local `EncoderBlock`s plus `fc1: Linear(1, d_model)`, `conv1/conv2: Conv1d(seq_len, seq_len, kernel_size=2/1)`, `fc2: Linear(seq_len, pred_len)`, `Tanh`.

They are built in the factory `emforecaster/utils/models.py` (lines 199-224), dispatched on `args.exp.model_id`:
- `RF_EMF_MLP` to `RF_EMF_MLP(seq_len=args.data.seq_len, pred_len=args.data.pred_len, d_model=args.sl.d_model)`
- `RF_EMF_CNN` to `RF_EMF_CNN(seq_len, pred_len, num_kernels=args.sl.num_kernels)`
- `RF_EMF_LSTM` to `RF_EMF_LSTM(seq_len, pred_len, d_model=args.sl.d_model)`
- `RF_EMF_Transformer` to `RF_EMF_Transformer(seq_len, pred_len, d_model=args.sl.d_model, num_heads=args.sl.num_heads, num_enc_layers=args.sl.num_enc_layers)`

The imports alias the classes: `from emforecaster.models.rf_emf_baselines import MLP as RF_EMF_MLP, CNN as RF_EMF_CNN, LSTM as RF_EMF_LSTM, Transformer as RF_EMF_Transformer` (`emforecaster/utils/models.py:13`).

The config class is the shared `SL` (`emforecaster/config/config.py:306`). Relevant defaults: `d_model=128` (`:318`), `num_heads=16` (`:322`), `num_enc_layers=3` (`:315`), `num_kernels=32` (`:373`). The realistic job-file values are much smaller: MLP/LSTM use `d_model: 32`, CNN uses `num_kernels: 32`, and the Transformer uses `d_model: 2`, `num_heads: 2`, `num_enc_layers: 2` (from the respective `args.yaml`). All run `seq_len: 336` to `pred_len: 96`, `batch_size: 512`, `lr: 1e-4`, `criterion: MSE`, `scheduler: onecycle`.

**How it runs (the forward pass).** The input from the loop is `x = batch[0]` of shape `(B, 1, L)` with `L = seq_len` (single-variable, `num_channels=1`). `forward_pass` may apply `x.diff(dim=1)` first if `args.data.difference_input` (`emforecaster/utils/models.py:353`), but no RevIN is applied. The `__main__` block confirms the contract with `x = torch.randn(32, 1, 20)`.

- **MLP** (line 14): `fc1` maps the last dim `(B,1,L)` to `(B,1,d_model)`, `Tanh`, `fc2` to `(B,1,pred_len)`, then `.squeeze()` to `(B, pred_len)`.
- **CNN** (line 31): it acts on the `(B,1,L)` input by adding a dim to get the conv's expected `(B, in_channels=1, L)`. `conv1` with `kernel_size=L` yields `(B, num_kernels, 1)`, `Tanh`, `view(B,-1)` to `(B, num_kernels)`, `fc` to `(B, pred_len)`.
- **LSTM** (line 47): `nn.LSTM(input_size=seq_len, hidden=d_model)` is fed `(B,1,L)` and returns `(B,1,d_model)`, `fc` to `(B,1,pred_len)`, `.squeeze()` to `(B, pred_len)`.
- **Transformer** (line 104): `x.squeeze()` to `(B,L)`, `unsqueeze(-1)` to `(B,L,1)`, `fc1` (Linear 1 to d_model) to `(B,L,d_model)`, loop over `num_enc_layers` `EncoderBlock`s (MHA over the length axis, each preserving `(B,L,d_model)`), `conv1: Conv1d(seq_len,seq_len,k=2)` over the L axis treating length as channels to `(B,L,d_model-1)`, `Tanh`, `conv2(k=1)` to the same width, `Tanh`, finally `fc2(x.squeeze())` maps `seq_len` to `pred_len`. Output is `(B, pred_len)`. (The Transformer's `EncoderBlock` is the local one at line 53, distinct from the PatchTST `EncoderBlock`. It uses `Tanh` in its FFN and a single shared `LayerNorm`.)

**Custom layers.** The only external dependency is `_MultiheadAttention` imported from `emforecaster/layers/patchtst/enc_block.py` (line 4), used by the Transformer's `EncoderBlock`. `_MultiheadAttention` (enc_block.py:98) is a thin wrapper around `nn.MultiheadAttention(embed_dim=d_model, num_heads, dropout, batch_first=True)` that returns `self.attn(Q,K,V, need_weights=False)`. MLP, CNN, and LSTM use only stock `torch.nn` modules.

**Notable details / gotchas.**
- **No RevIN, no decomposition, no patching, no channel-independence machinery.** Unlike every other model in the factory (PatchTST, EMForecaster, TSMixer, and so on, which receive `revin=args.sl.revin` and friends, see `emforecaster/utils/models.py:56`, `:79`), the four RF-EMF baseline constructors receive *no* RevIN/affine/eps arguments. They operate on raw (optionally differenced and pre-scaled) inputs. Normalization is only the dataset-level `scale: True` standardization.
- **Single-variable-only by construction.** Every model assumes `num_channels=1`. Multiple `.squeeze()` calls (MLP line 18, LSTM line 50, Transformer lines 105/121) would collapse or mangle dimensions if a real channel axis (>1) were present, and the LSTM/Transformer treat `seq_len` itself as a feature/channel dimension rather than the recurrence/sequence axis. These are not safe to run multivariate.
- **CNN kernel equals the whole window.** `kernel_size = seq_len` (line 24) means the conv is effectively a global filter bank producing one value per kernel, closer to a learned linear projection over the window than a sliding-window TCN.
- **Transformer conv quirk.** After attention, the model applies two `Conv1d`s along the length axis (`in/out_channels = seq_len`). `conv1` has `kernel_size=2`, which shrinks `d_model` by 1. With the job-file `d_model: 2` this leaves width 1, after which `fc2` maps the squeezed length axis `seq_len` to `pred_len`. The architecture is sensitive to `d_model` being large enough for the kernel-2 conv.
- **Tiny by design.** The shipped Transformer config (`d_model: 2`, `num_heads: 2`, `num_enc_layers: 2`) is intentionally minimal. These are weak baselines, not tuned competitors.

## Part III. Operating the Framework

## How to Run: Quick Reference

### Install

```bash
# Python >= 3.10, < 3.14 (numba/numba-llvmlite cap the upper bound)
python3.12 -m venv venv && source venv/bin/activate
pip install -r requirements.txt   # exact pinned versions (reproducible)
pip install -e .                  # register the editable `emforecaster` package
```

### Run an experiment

```bash
python main.py <job_name>
```

- `<job_name>` is a **directory under `emforecaster/jobs/exp/`**, not a model name. `python main.py rf_emf/emforecaster` loads `emforecaster/jobs/exp/rf_emf/emforecaster/args.yaml`.
- The model itself is chosen *inside* that YAML through `exp.model_id`. To switch models, either point at a different job dir or edit `model_id`.
- Each entry in `exp.seed_list` produces an independent run with its own log directory.

### Hyperparameter tuning (grid search)

```bash
python emforecaster/tuning/tune.py <model>
```

It reads the model's `ablation.yaml`, enumerates the grid, and re-runs once per cell, injecting overrides as a dotted-key JSON `--ablation` string (for example `{"sl.lr": 0.01}`). Each cell is stamped with an incrementing `exp.ablation_id`.

### Turn on uncertainty quantification

Set this in the YAML under the conformal block:

```yaml
conf:
  conf: true          # enable the conformal pipeline
  alpha: 0.1          # 1 - target coverage (0.1 gives 90% prediction intervals)
  validation_eval: false
```

### Online logging (optional)

```bash
export NEPTUNE_API_TOKEN='your-token'
```

Then set `exp.neptune: true` in the YAML. With Neptune off, everything is written offline instead.

### Where results land

```text
logs/<ablation_id>_<model_id>_<exp.id>/<seed>/
├── args.yaml          # the fully-resolved config that produced this run
├── log.json           # metrics per epoch plus final test MSE/MAE (offline logger)
├── supervised.pth     # best-model checkpoint (by exp.best_model_metric)
└── plot_*.png         # per-sample forecast-vs-truth plots on the test set
```

(The shipped example run `logs/1_EMForecaster_3oWOFEniR3/69423/` reached **test MSE about 0.299, MAE about 1.165** with a roughly 2.4M-param model on CPU. See `PLOT_NOTES.md`.)

## Caveats, Sharp Edges, and Dead Code

Things worth knowing before you trust or extend this code, surfaced directly from reading the source:

- **Config-path discovery is fragile.** `main.py:67-68` climbs two directories above `main.py` and then re-appends a **doubled** `emforecaster/emforecaster/...`. It only works because the repo root folder is itself named `emforecaster`. Renaming the root, moving `main.py`, or vendoring the package elsewhere silently breaks config loading.
- **"Distributed" is cosmetic.** Despite `print_master` ("prints to the rank 0 node") and an `"Evaluation loss after all_reduce"` log string (`exp.py:843`), there is **no** `DistributedDataParallel`, process group, or real `dist.all_reduce`. Runs are single-device (`main.py:88` logs "Using single device"). The big "after all_reduce" numbers in logs are summed totals. The real per-sample loss is the small number beneath them (per `PLOT_NOTES.md`).
- **`models/mamba.py` is dead code.** The standalone `Mamba` class is not wired into the `get_model` factory. Mamba is actually reached through `RecurrentModel` with `backbone_id="Mamba"`. It also depends on the external `mambapy` package.
- **Forecast plots are unreadable as shipped.** Per `PLOT_NOTES.md`, the saved `plot_*.png` draw prediction and ground truth in the **same color with no legend**, so you cannot tell them apart. Flat straight segments in some plots are likely interpolated missing data, worth verifying before drawing conclusions.
- **Two different "grid searches" exist.** `model_id="KNN"` wraps the estimator in sklearn's `GridSearchCV` (`models.py:168-175`), which is internal cross-validation. That is different from the repo's own ablation grid search in `tuning/tune.py`, which sweeps configs across full runs.
- **The models are effectively single-variable.** Several (`Linear`, the RF-EMF baselines) `squeeze()` the channel dim and assume `C=1`. The framework is built around single-channel EMF exposure, so multivariate use would need care.
- **There is an `__init__ copy.py`** under `utils/`, a stray duplicate, not an import target.

## Glossary

| Term | Meaning in this repo |
| --- | --- |
| **EMF exposure** | Electromagnetic-field strength in volts/meter (V/m), the quantity being forecast. |
| **`seq_len` / lookback** | Number of past timesteps fed in (default 336, about 33.6 h at 6-min cadence). |
| **`pred_len` / horizon (H)** | Number of future timesteps predicted (default 96, about 9.6 h). |
| **RevIN** | Reversible Instance Normalization. Normalize each instance by its own mean/std before the model, denormalize after. Stabilizes training under distribution shift. Toggled by `sl.revin/revout/revin_affine`. |
| **Patching** | Slicing the series into overlapping windows (`patch_dim` long, `patch_stride` apart) so a model processes patch tokens instead of raw timesteps. Implemented in `layers/patcher.py`. |
| **Channel-independence** | Treating each channel as its own single-variable sequence sharing one set of weights (PatchTST/ModernTCN style), implemented by folding channels into the batch dim. |
| **Conformal prediction** | A distribution-free method that converts point forecasts into prediction intervals with a finite-sample coverage guarantee, by calibrating nonconformity scores on a held-out split. |
| **Empirical coverage** | Fraction of test targets that actually fall inside the predicted interval. Should be at least `1 - alpha`. |
| **Interval width** | Average size of the prediction interval. Smaller is more informative (at fixed coverage). |
| **Trade-off score** | The paper's proposed single metric that unifies width (minimize) and coverage (maximize) into one number. See `conformal/tradeoff.py`. |
| **`alpha`** | Miscoverage level. `alpha=0.1` targets 90% intervals. |
| **SSM / Mamba** | Selective state-space model, a linear-time sequence backbone alternative to attention. |
| **TCN** | Temporal Convolutional Network, causal/dilated 1D convolutions instead of recurrence or attention. |
| **MLP-Mixer** | An all-MLP architecture that alternately mixes across time and across channels (TSMixer, EMForecaster's patch model). |
| **Ablation** | A hyperparameter sweep cell. `exp.ablation_id` tags each, driven by `ablation.yaml` plus `tuning/tune.py`. |
| **`model_id` / `backbone_id`** | Config strings that select the model class and (for `RecurrentModel`) its inner recurrent/SSM backbone. |

---

*End of guide. Regenerate it by re-reading the source. Every claim above is grounded in a specific file and line, so if the code moves, re-verify before trusting.*
