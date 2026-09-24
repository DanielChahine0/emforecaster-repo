import os
import sys
import datetime
import gc
import json
import warnings
import shutil
import random
import matplotlib.pyplot as plt

warnings.filterwarnings(
    "ignore", message="h5py not installed, hdf5 features will not be supported."
)
warnings.filterwarnings("ignore", message=".*omp_set_nested routine deprecated.*")


# Rich console
from rich.console import Console
from rich.pretty import pprint

# Pydantic
from pydantic import BaseModel
from typing import Any

# Torch
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Custom Modules
from forecasting.multivariate.utils.train import EarlyStopping
from forecasting.multivariate.utils.dataloading import get_loaders
from forecasting.multivariate.utils.models import (
    get_criterion,
    get_model,
    get_optim,
    get_scheduler,
    compute_loss,
    compute_aux_loss,
    model_update,
    forward_pass,
    model_input,
)
from forecasting.multivariate.utils.classification import (
    get_logger_mapping,
    update_stats,
    binary_classification_metrics,
    multi_classification_metrics,
    get_metrics,
)
from forecasting.multivariate.conformal.coverage import get_all_critical_scores, get_coverage
from forecasting.run_path import build_run_dir, PER_CHANNEL_SETUP
from forecasting import provenance
from forecasting import hardware

# Logger
import neptune
from forecasting.multivariate.utils.logger import (
    log_pydantic,
    epoch_logger,
    format_time_dynamic,
    global_to_yaml,
)

# Timing
import time


class Experiment:
    def __init__(self, args):
        self.args = args
        self.start_time = time.time()

        # Add this to suppress OpenMP warnings
        import os
        os.environ["KMP_WARNINGS"] = "off"
        os.environ["OMP_MAX_ACTIVE_LEVELS"] = "1"

    def run(self):
        # Rich Console
        self.console = Console()

        # Reproducibility
        self.generator = torch.Generator().manual_seed(self.args.exp.seed)
        torch.manual_seed(self.args.exp.seed)  # CPU
        np.random.seed(self.args.exp.seed)  # Numpy
        random.seed(self.args.exp.seed)  # Python
        if torch.cuda.is_available():
            torch.cuda.manual_seed(self.args.exp.seed)  # GPU
            torch.cuda.manual_seed_all(self.args.exp.seed)  # multi-GPU

        # Logging
        self.init_logger()

        # Initialize Device
        self.init_device()

        # Learning Type
        self.supervised_train()

        # Stop Logger
        if self.args.exp.neptune:
            self.logger.stop()
        else:
            # Save offline logging to JSON file
            with open(self.log_file, "w") as f:
                json.dump(self.logger, f, indent=2)

    def init_device(self):
        """
        Initialize CUDA (or MPS) devices.
        """
        if self.args.exp.mps:
            self.device = torch.device(
                "mps" if torch.backends.mps.is_available() else "cpu"
            )
            self.console.log(f"MPS hardware acceleration activated.")
        elif not torch.cuda.is_available():
            self.device = torch.device("cpu")
            self.console.log("CUDA not available. Running on CPU.")
        else:
            self.device = torch.device(f"cuda:{self.args.exp.gpu_id}")
            # perf: TF32 tensor-core matmuls + cuDNN autotune. Safe: run() seeds
            # RNGs but never requests deterministic algorithms.
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.set_float32_matmul_precision("high")
            torch.backends.cudnn.benchmark = True

        self.console.log(f"Device initialized to: {self.device}")

    def init_dataloaders(self, learning_type="sl", loader_type="train"):
        """
        Initialize the dataloaders depending on the learning type and loader type.

        Args:
            loader_type (str): Options: "train", "test", "all". "train" returns train and val loaders. "test" returns test loader. "all" returns all loaders.
            learning_type (str): Options: "sl", "ssl", "downstream". "sl" is supervised learning. "ssl" is self-supervised learning. "downstream" is downstream learning.
        """

        # Scikit-learn pipeline
        if self.args.exp.sklearn:
            self.seq_load(loader_type="all", learning_type="sl")
            self.print_master(
                f"{len(self.train_loader)} train samples. {len(self.val_loader)} validation samples. {len(self.test_loader)} test samples."
            )
            return

        # Deep learning (PyTorch) models
        if self.args.data.seq_load:
            self.seq_load(loader_type, learning_type)
        else:
            raise ValueError(
                f"Invalid dataloading option. Please set either data.seq_load or data.rank_seq_load to {True}."
            )

    def seq_load(self, loader_type="train", learning_type="sl"):
        self.console.log(
            f"Running sequential dataloading ({loader_type})."
        )
        self.free_memory()
        loaders = get_loaders(
            self.args,
            learning_type,
            self.generator,
            self.args.sl.dataset_class,
            loader_type,
        )

        if loader_type == "train":
            self.train_loader, self.val_loader = loaders[:2]
            self.print_master(
                f"{len(self.train_loader.dataset)} train samples. {len(self.val_loader.dataset)} validation samples."
            )
        elif loader_type == "test":
            self.test_loader = loaders[0]
            self.print_master(f"{len(self.test_loader.dataset)} test samples.")
        elif loader_type == "all":
            self.train_loader, self.val_loader, self.test_loader = loaders[:3]
            if not self.args.exp.sklearn:
                self.print_master(
                    f"{len(self.train_loader.dataset)} train samples. {len(self.val_loader.dataset)} validation samples. {len(self.test_loader.dataset)} test samples."
                )
        else:
            raise ValueError("Invalid loader type.")

        if self.args.data.median_seq_len:
            self.args.data.seq_len = loaders[-1]
            self.logger["parameters/data/seq_len"] = loaders[-1]
            self.print_master(f"Sequence set to the median: {self.args.data.seq_len}.")

        # get_loaders infers num_channels from the data, but init_logger has
        # already dumped the config's advisory value by then. Re-log it so the
        # run records the channel count it actually trained on: harmless while
        # every dataset happened to carry 27, actively misleading once the VoIP
        # files (6 channels) run through job configs that still say 27.
        self.logger["parameters/data/num_channels"] = self.args.data.num_channels

        # How many windows each split gave the run, so that runs compared
        # across paradigms can be checked to have fitted and scored the same
        # data from their logs alone (finding PE-9 had to recompute the counts
        # with the loader). test() loads the test split before it logs any
        # test metric, so these always precede the grid's resume marker.
        if not self.args.exp.sklearn:
            for split in ("train", "val", "test"):
                loader = getattr(self, f"{split}_loader", None)
                if loader is not None:
                    self.logger[f"parameters/data/n_{split}_windows"] = len(
                        loader.dataset
                    )

    def free_memory(self):
        for k in ["train", "val", "test"]:
            if hasattr(self, f"{k}_loader"):
                loader = getattr(self, f"{k}_loader")
                if hasattr(loader.dataset, "close"):
                    loader.dataset.close()
                del loader
                delattr(self, f"{k}_loader")

        gc.collect()
        # perf: no cuda.empty_cache() here — these datasets are tiny, the cache
        # never needs flushing, and empty_cache() device-syncs then forces slow
        # cudaMalloc re-allocation in the next phase.
        if torch.backends.mps.is_available():
            # MPS has its own caching allocator; cuda.empty_cache() is a no-op here,
            # so freed loader memory was never returned between phases on the 16GB box.
            torch.mps.empty_cache()

    def init_model(self):
        """
        Initialize the model.
        """
        self.model = get_model(self.args, self.generator)
        if self.args.exp.sklearn:
            self.print_master(
                f"Scikit-learn model {self.args.exp.model_id} initialized."
            )
            return
        else:
            self.model.to(self.device)

        # Optional torch.compile. self.model stays the eager module for every
        # checkpoint save/load and the optimizer wiring (so state_dict keys keep
        # their clean, prefix-free names and compiled/eager runs are
        # checkpoint-compatible). self.model_fwd is the handle used for the
        # forward pass only; it wraps the same module, so it shares parameters
        # with the optimizer and picks up load_state_dict updates in place.
        # Numerically equivalent to eager (float32-rounding-level differences).
        if getattr(self.args.exp, "compile", False):
            self.model_fwd = torch.compile(self.model)
            self.print_master("torch.compile enabled for the forward pass.")
        else:
            self.model_fwd = self.model

        num_params = self.count_parameters()
        self.logger["parameters/sl/num_params"] = num_params
        self.print_master(
            f"{self.args.exp.model_id} model initialized with {num_params:,} parameters."
        )

    def init_optimizer(self):
        """
        Initialize the optimizer
        """
        self.optimizer = get_optim(self.args, self.model, self.args.sl.optimizer)
        self.print_master(f"{self.args.sl.optimizer} optimizer initialized.")

    def init_logger(self):
        """
        Initialize the logger
        """

        # Create log directory
        self.log_dir = os.path.join(
            build_run_dir(
                # This package serves both sweeps: the joint grid (all bands in
                # one model) and the per-band grid, which runs these same configs
                # one channel at a time. Split them into logs/multivariate/ and
                # logs/univariate (per channel)/ so a per-band sweep cannot be
                # mistaken for, or collide with, the joint results.
                setup=(
                    "cluster"
                    if self.args.cluster.enabled
                    else PER_CHANNEL_SETUP
                    if self.args.data.single_channel
                    else "multivariate"
                ),
                model_id=self.args.exp.model_id,
                dataset=self.args.data.dataset,
                seq_len=self.args.data.seq_len,
                pred_len=self.args.data.pred_len,
                conf_on=self.args.conf.conf,
                ablation_id=self.args.exp.ablation_id,
                tag=self.args.exp.id,
            ),
            str(self.args.exp.seed),
        )
        if os.path.exists(self.log_dir):
            self.print_master(f"Using existing log directory: {self.log_dir}")
        else:
            os.makedirs(self.log_dir, exist_ok=True)
            self.print_master(f"Created new log directory: {self.log_dir}")

        # Save arguments to YAML file
        yaml_path = os.path.join(self.log_dir, "args.yaml")
        global_to_yaml(self.args, yaml_path)

        if self.args.exp.neptune:
            # Initialize Neptune run with the time-based ID
            self.logger = neptune.init_run(
                project=self.args.exp.project_name,
                api_token=self.args.exp.api_token,
                custom_run_id=self.args.exp.time,
            )
            self.print_master("Neptune logger initialized.")
            self.logger["config"].upload(yaml_path)
        else:
            self.logger = dict()
            self.log_file = os.path.join(self.log_dir, "log.json")
            self.print_master("Offline logger initialized.")

        log_pydantic(self.logger, self.args, "parameters")

        # Provenance: which code, which machine, and how many trainings shared
        # the GPU. A run that lacks these cannot be placed in a paradigm
        # comparison (see .personal/PLAN.md, contract C1/C7).
        for key, value in provenance.run_provenance(self.args).items():
            self.logger[f"parameters/exp/{key}"] = value

        # Conformal prediction (logdir)
        if self.args.conf.conf:
            self.score_path = os.path.join(self.log_dir, "scores")

    def init_earlystopping(self, path: str):
        self.early_stopping = EarlyStopping(
            patience=self.args.early_stopping.patience,
            verbose=self.args.early_stopping.verbose,
            delta=self.args.early_stopping.delta,
            path=path,
        )

    def sklearn_train(self, model, train_data, train_targets):
        self.print_master(f"Training scikit-learn model {self.args.exp.model_id}.")
        model.fit(train_data, train_targets)
        self.print_master("Sklearn training completed.")

    def sklearn_eval(
        self,
        model,
        criterion,
        inputs,
        targets,
        acc=False,
        mae=False,
        ch_acc=False,
        flag="sl",
    ):
        stats = dict()

        # Compute predictions
        num_examples = inputs.shape[0]
        preds = torch.from_numpy(model.predict(inputs))
        probs = torch.from_numpy(model.predict_proba(inputs)).to(self.device)

        if self.args.open_neuro.task == "binary":
            probs = probs[
                :, 1
            ]  # Get positive class value. Otherwise use multiclass values

        targets = torch.from_numpy(targets).to(self.device)

        # Loss
        if isinstance(criterion, nn.BCELoss):
            probs = probs.double()
            targets = targets.double()
        elif self.args.open_neuro.task == "multi":
            targets = targets.long()

        print(f"Probs: {probs.shape}, Targets: {targets.shape}")

        stats["loss"] = criterion(probs, targets).item()

        if mae:
            mae_loss = nn.L1Loss()
            stats["mae"] = mae_loss(preds, targets).item()

        if self.args.exp.other_metrics:
            if self.args.open_neuro.task == "binary":
                single_metrics = binary_classification_metrics(
                    preds, targets, probs, self.device
                )  # Shape: (7,)
            elif self.args.open_neuro.task == "multi":
                single_metrics = multi_classification_metrics(
                    preds, targets, probs, self.device
                )  # Shape: (15,)
            else:
                raise ValueError("Invalid task.")
        elif acc:
            single_metrics = ((preds == targets).float().mean()).unsqueeze(
                0
            )  # Shape: (1,)
        else:
            single_metrics = None

        # Window Metrics
        if acc or ch_acc:
            ch_acc = True if self.args.data.full_channels else ch_acc
            acc = False if self.args.data.full_channels else acc
            update_stats(
                stats,
                single_metrics,
                self.args.open_neuro.task,
                self.args.exp.other_metrics,
                ch_acc,
            )
            self.log_stats(stats, flag, mae, acc, ch_acc, mode="test")

        # Clear all tensors before returning stats
        del single_metrics
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif torch.backends.mps.is_available():
            torch.mps.empty_cache()

    def train(
        self,
        model,
        model_id,
        optimizer,
        train_loader,
        best_model_path,
        criterion,
        val_loader=None,
        scheduler=None,
        flag="sl",
        ema=None,
        mae=False,
        acc=False,
        early_stopping=False,
        ch_acc=False,
        conformal=False,
        test_loader=None,
    ):
        """
            Trains a model.

        Args:
            model (nn.Module): The model to train.
            model_id (str): The model ID.
            optimizer (torch.optim): The optimizer to use.
            train_loader (torch.utils.data.DataLoader): The training data.
            best_model_path (str): The path to save the best model.
            criterion (torch.nn): The loss function.
            val_loader (torch.utils.data.DataLoader): The validation data.
            scheduler (torch.optim.lr_scheduler): The learning rate scheduler.
            flag (str): The type of learning. Options: "sl", "ssl", "downstream".
        """

        # Scikit-learn pipeline
        if self.args.exp.sklearn:
            self.sklearn_train(model, train_loader.x, train_loader.y)
            return

        # Fit-once, non-gradient pipeline (ARIMA). There is no epoch loop and no
        # checkpoint, so return before both. Unlike the sklearn branch above this
        # does NOT short-circuit test(): the conformal recalibration on val and
        # the test evaluation still run against the same fixed predictor, so
        # every sl_test/* key -- including the grid's resume marker -- is written
        # normally. Returning here also keeps the model from forecasting the
        # train split, which it never needs and which would cost more calls than
        # val and test combined.
        if self.args.exp.no_train:
            self.print_master(
                f"{self.args.exp.model_id}: fit-once model, skipping the epoch loop."
            )
            return

        # Deep learning (PyTorch) pipeline
        num_examples = len(train_loader.dataset)
        self.print_master(f"Training on {num_examples} examples...")

        # Gradient accumulation: backward every micro-batch, step the optimizer
        # (and scheduler) once per group of accum_steps, so batch_size 128 x
        # accum 4 follows the same optimization trajectory as batch_size 512.
        accum_steps = max(1, getattr(getattr(self.args, flag), "accum_steps", 1))
        num_batches = len(train_loader)

        if early_stopping:
            self.init_earlystopping(best_model_path)
            self.print_master("Early stopping initialized.")

        # Synchronize before training starts

        self.best_val_metric = float("inf")
        train_start_time = time.time()

        # <--------------- Training --------------->
        for epoch in range(eval(f"self.args.{flag}.epochs")):
            model.train()
            total_loss = torch.tensor(0.0, device=self.device)
            total_aux_loss = torch.tensor(0.0, device=self.device)
            running_loss = torch.tensor(0.0, device=self.device)
            running_num_examples = torch.tensor(0.0, device=self.device)
            start_time = time.time()
            epoch_start_time = time.time()

            for i, batch in enumerate(train_loader):
                if i % accum_steps == 0:
                    optimizer.zero_grad()
                output = forward_pass(self.args, self.model_fwd, batch, model_id, self.device)
                loss = compute_loss(
                    output, batch, criterion, model_id, self.args, self.device
                )
                # Extra objective terms that are not a function of the
                # predictions -- currently only CCM's similarity loss. Kept out
                # of `loss` on purpose: `loss` is what gets logged and what
                # selects the best epoch, and it has to stay the forecast loss so
                # a cluster cell is comparable to every other cell. self.model
                # (eager), not self.model_fwd: it reads module attributes the
                # torch.compile wrapper does not forward.
                aux_loss = compute_aux_loss(self.model, batch, self.args, self.device)

                # Metrics
                num_batch_examples = batch[0].shape[0]  # perf: plain int; int * device-tensor stays on-device and skips a per-step host->device copy
                total_loss += loss * num_batch_examples
                running_loss += loss * num_batch_examples
                running_num_examples += num_batch_examples
                if aux_loss is not None:
                    total_aux_loss += aux_loss.detach() * num_batch_examples

                # Update model parameters. The backward loss is scaled by the
                # accumulation factor so summed micro-batch gradients match the
                # full-batch gradient; logging above uses the unscaled loss.
                alpha = next(ema) if ema else 0.966
                do_step = ((i + 1) % accum_steps == 0) or (i + 1) == num_batches
                step_loss = loss if aux_loss is None else loss + aux_loss
                model_update(
                    model, step_loss / accum_steps, optimizer, model_id, alpha,
                    step=do_step,
                )

                # Periodic Logging
                if (i + 1) % 100 == 0:
                    loss_tensor = running_loss.to(self.device)
                    num_examples_tensor = running_num_examples.to(self.device)

                    end_time = time.time()

                    # Only rank 0 prints and logs details
                    average_loss = loss_tensor.item() / num_examples_tensor.item()
                    self.print_master(
                        f"[Epoch {epoch}, Batch ({i+1}/{len(train_loader)})]: {end_time - start_time:.3f}s. Loss: {average_loss:.6f}"
                    )
                    (
                        self.print_master(f"EMA decay rate: {alpha}")
                        if flag == "ssl"
                        else None
                    )

                    # Reset trackers

                    running_loss = torch.tensor(0.0, device=self.device)
                    running_num_examples = torch.tensor(0.0, device=self.device)
                    start_time = time.time()

                # Scheduler advances once per optimizer step (not per micro-batch)
                # so the onecycle trajectory is independent of accum_steps.
                if scheduler and do_step:
                    scheduler.step()

            epoch_loss = total_loss.item() / num_examples
            epoch_time = time.time() - epoch_start_time
            self.print_master(
                f"Epoch {epoch}. Training loss: {epoch_loss:.6f}. Time: {epoch_time:.2f}s."
            )
            epoch_logger(self.args, self.logger, f"{flag}_train/loss", epoch_loss)
            epoch_logger(self.args, self.logger, f"{flag}_train/epoch_time", epoch_time)
            # Only on a run that has an auxiliary term (CCM), so every other
            # run's log.json keeps exactly the keys it had before. This is the
            # curve to watch to see whether the cluster assignment converged.
            if self.args.cluster.enabled and self.args.cluster.beta:
                epoch_logger(self.args, self.logger, f"{flag}_train/similarity_loss",
                             total_aux_loss.item() / num_examples)

            # <--------------- Validation --------------->
            if val_loader:

                # <------------Calibration (CP)----------->
                if conformal:
                    if self.args.conf.validation_eval:  # Calibrates on every epoch
                        self.evaluate(
                            model=model,
                            model_id=model_id,
                            loader=val_loader,
                            criterion=criterion,
                            flag=flag,
                            mae=mae,
                            acc=acc,
                            ch_acc=ch_acc,
                            conformal_calibrate=True,
                            coverage=False,
                        )
                    elif (
                        epoch == eval(f"self.args.{flag}.epochs") - 1
                    ):  # Calibrates only on last epoch
                        self.evaluate(
                            model=model,
                            model_id=model_id,
                            loader=val_loader,
                            criterion=criterion,
                            flag=flag,
                            mae=mae,
                            acc=acc,
                            ch_acc=ch_acc,
                            conformal_calibrate=True,
                            coverage=False,
                        )

                # Validate
                self.validate(
                    model,
                    val_loader,
                    model_id,
                    criterion,
                    flag,
                    mae,
                    acc,
                    epoch,
                    best_model_path,
                    early_stopping,
                    ch_acc,
                )

            # <--------------- Per-epoch test loss --------------->
            # Logged for the train/val/test loss curve. Model selection still uses
            # the validation loss only, so this never feeds into best-model choice.
            if test_loader is not None:
                test_stats = self.evaluate(
                    model=model,
                    model_id=model_id,
                    loader=test_loader,
                    criterion=criterion,
                    flag=flag,
                    mae=False,
                    acc=False,
                    ch_acc=False,
                    conformal_calibrate=False,
                    coverage=False,
                )
                self.print_master(f"Epoch {epoch}. Test loss: {test_stats['loss']:.6f}.")
                epoch_logger(
                    self.args, self.logger, f"{flag}_test/loss", test_stats["loss"]
                )

            # Early stopping
            if early_stopping:
                if self.early_stopping.early_stop:
                    self.print_master(f"EarlyStopping activated, ending training.")

                    # Calibrate model on last epoch if early stopping triggered
                    if conformal and not self.args.conf.validation_eval:
                        self.evaluate(
                            model=model,
                            model_id=model_id,
                            loader=val_loader,
                            criterion=criterion,
                            flag=flag,
                            mae=mae,
                            acc=acc,
                            ch_acc=ch_acc,
                            conformal_calibrate=True,
                            coverage=False,
                        )

                    break

            # Logging Checkpoint
            if self.args.exp.neptune:
                pass
            else:
                run_time = time.time() - self.start_time
                self.logger["parameters/running_time"] = format_time_dynamic(
                    run_time
                )

        # <--------------- Total training time --------------->
        # Logged to whichever backend is active (not only the offline branch).
        total_training_time = time.time() - train_start_time
        self.logger["parameters/total_training_time"] = format_time_dynamic(
            total_training_time
        )
        self.logger["parameters/total_training_seconds"] = total_training_time
        self.print_master(
            f"Total training time: {format_time_dynamic(total_training_time)}."
        )


    def validate(
        self,
        model,
        val_loader,
        model_id,
        criterion,
        flag,
        mae,
        acc,
        epoch,
        best_model_path,
        early_stopping,
        ch_acc,
    ):
        """
        Validate the model.
        """
        stats = self.evaluate(
            model=model,
            model_id=model_id,
            loader=val_loader,
            criterion=criterion,
            flag=flag,
            mae=mae,
            acc=acc,
            ch_acc=ch_acc,
            conformal_calibrate=False,
            coverage=self.args.conf.validation_eval,
        )

        # Synchronize before validation
        ch_acc = True if self.args.data.full_channels else ch_acc
        acc = False if self.args.data.full_channels else acc
        val_loss = stats["loss"]
        self.log_stats(
            stats,
            flag,
            mae,
            acc,
            ch_acc,
            mode="val",
            coverage=self.args.conf.validation_eval,
        )

        if self.args.exp.best_model_metric == "loss":
            val_metric = val_loss
        elif self.args.exp.best_model_metric in {"acc", "ch_acc", "ch_f1"}:
            val_metric = -stats[self.args.exp.best_model_metric]
        else:
            raise ValueError(
                f"Invalid best model metric: {self.args.exp.best_model_metric}"
            )

        # Save best model and apply early stopping
        if early_stopping:
            self.early_stopping(val_metric, model)

        else:
            if val_metric < self.best_val_metric:
                if self.args.exp.best_model_metric == "loss":
                    self.print_master(
                        f"Validation loss decreased ({self.best_val_metric:.6f} --> {val_metric:.6f})."
                    )
                elif self.args.exp.best_model_metric in {"acc", "ch_acc", "ch_f1"}:
                    self.print_master(
                        f"Validation {self.args.exp.best_model_metric} increased ({-self.best_val_metric*100:.3f}% --> {-val_metric*100:.3f}%)."
                    )
                path_dir = os.path.abspath(os.path.dirname(best_model_path))

                # Save model
                if not os.path.isdir(path_dir):
                    os.makedirs(path_dir)
                torch.save(model.state_dict(), best_model_path)
                self.print_master(f"Saving Model Weights at: {best_model_path}...")

                # Save conformal calibration scores
                if self.args.conf.conf and self.args.conf.validation_eval:
                    self.print_master(
                        f"Saving Model Calibration Scores at: {self.score_path}..."
                    )
                    torch.save(self.calibration_scores, self.score_path)

                self.best_val_metric = val_metric



        self.print_master("Validation complete")

    def supervised_train(self):
        """
        Train the model in supervised mode.
        """

        # Load train + val loaders. When per-epoch test logging is enabled, build the
        # test loader up front too so the test loss can be tracked each epoch; the
        # final test() still reloads and evaluates it on the best (val-selected)
        # model. Otherwise the test loader is built once in test().
        if self.args.exp.eval_test_per_epoch:
            self.init_dataloaders(loader_type="all", learning_type="sl")
        else:
            self.init_dataloaders(loader_type="train", learning_type="sl")

        # Dump args.yaml again now that the loaders have set num_channels: the
        # copy init_logger wrote still has the job config's 27, and a 29-band or
        # VoIP run rebuilt from it does not fit its own checkpoint (finding C7).
        yaml_path = os.path.join(self.log_dir, "args.yaml")
        global_to_yaml(self.args, yaml_path)
        if self.args.exp.neptune:
            # Replace the copy init_logger uploaded, which has the same stale 27.
            self.logger["config"].upload(yaml_path)

        # Initialize Model and Optimizer
        self.init_model()
        self.init_optimizer()

        # Get supervised criterion

        self.criterion = get_criterion(self.args, self.args.sl.criterion)
        self.print_master(f"{self.args.sl.criterion} initialized.")

        # Get supervised scheduler. Per-epoch step count is optimizer steps, which
        # under gradient accumulation is ceil(batches / accum_steps).
        if self.args.sl.scheduler != "None":
            accum_steps = max(1, self.args.sl.accum_steps)
            self.sl_scheduler = get_scheduler(
                self.args,
                self.args.sl.scheduler,
                "supervised",
                self.optimizer,
                -(-len(self.train_loader) // accum_steps),
            )
        else:
            self.sl_scheduler = None
        self.print_master("Starting Supervised Training...")

        # Supervised Training
        best_model_path = os.path.join(self.log_dir, f"supervised.pth")
        self.train(
            model=self.model,
            model_id=self.args.exp.model_id,
            optimizer=self.optimizer,
            train_loader=self.train_loader,
            best_model_path=best_model_path,
            criterion=self.criterion,
            val_loader=self.val_loader,
            scheduler=self.sl_scheduler,
            flag="sl",
            mae=self.args.exp.mae,
            acc=self.args.exp.acc,
            early_stopping=self.args.sl.early_stopping,
            ch_acc=self.args.exp.ch_acc,
            conformal=self.args.conf.conf,
            test_loader=(
                self.test_loader if self.args.exp.eval_test_per_epoch else None
            ),
        )

        # Upload best model to Neptune
        if self.args.exp.neptune and not self.args.exp.sklearn:
            self.print_master(f"Uploading best model to Neptune.")
            self.logger[f"model_checkpoints/{self.args.exp.model_id}_sl"].upload(
                best_model_path
            )

        # Test model
        self.print_master("Starting Supervised Testing...")

        self.test(
            model=self.model,
            model_id=self.args.exp.model_id,
            best_model_path=best_model_path,
            criterion=self.criterion,
            flag="sl",
            mae=self.args.exp.mae,
            acc=self.args.exp.acc,
            ch_acc=self.args.exp.ch_acc,
        )

    def test(
        self,
        model,
        model_id,
        best_model_path,
        criterion,
        flag="sl",
        mae=False,
        acc=False,
        ch_acc=False,
    ):

        # Load data
        if self.args.exp.sklearn:
            pass
        elif self.args.exp.calibrate:
            self.init_dataloaders(loader_type="all", learning_type="sl")
        elif self.args.cluster.enabled or (
            self.args.conf.conf and not self.args.conf.validation_eval
        ):
            # Conformal recalibration below needs the val set, and a cluster
            # run's fixed membership the train split too. seq_load frees any
            # existing loaders, so build train/val/test together to keep them.
            self.init_dataloaders(loader_type="all", learning_type="sl")
        else:
            self.init_dataloaders(loader_type="test", learning_type="sl")

        # <---Scikit-learn pipeline--->
        if self.args.exp.sklearn:
            self.sklearn_eval(
                model,
                criterion,
                self.test_loader.x,
                self.test_loader.y,
                acc,
                mae,
                ch_acc,
                flag,
            )
            return

        # <---Deep learning (PyTorch) pipeline--->
        # Load best model. A no_train model wrote no checkpoint (and has no
        # val-selected "best" epoch to reload); it is already the fixed predictor
        # the conformal recalibration below assumes, so the round-trip is skipped
        # rather than faked.
        if not self.args.exp.no_train:
            model_weights = torch.load(best_model_path)
            model.load_state_dict(model_weights)

        # <---Conformal recalibration on the best (val-selected) model--->
        # Split conformal requires the calibration residuals and the test residuals to
        # come from the same fixed predictor. During training (validation_eval=False)
        # the scores were calibrated on the live last/early-stop-epoch model, but the
        # weights just loaded are the best checkpoint; whenever the best epoch is not
        # the final/trigger epoch these differ and the coverage guarantee is void.
        # Recalibrate on the val set with the best model so both predictors match.
        # The validation_eval=True path already persists/reloads scores tied to the
        # best checkpoint, so it is left untouched.
        if self.args.conf.conf and not self.args.conf.validation_eval:
            self.evaluate(
                model=model,
                model_id=model_id,
                loader=self.val_loader,
                criterion=criterion,
                flag=flag,
                mae=mae,
                acc=acc,
                ch_acc=ch_acc,
                conformal_calibrate=True,
                coverage=False,
            )

        # Test set evaluation

        stats = self.evaluate(
            model=model,
            model_id=model_id,
            loader=self.test_loader,
            criterion=criterion,
            flag=flag,
            mae=mae,
            acc=acc,
            ch_acc=ch_acc,
            conformal_calibrate=False,
            coverage=self.args.conf.conf,
            plot_results=True,
            num_plots=10,
        )

        # <---CCM: the same checkpoint under one fixed membership--->
        # The primary cluster evaluation (models/ccm.py); the batch-protocol
        # stats above stay the sensitivity row. Runs after them so they are
        # computed exactly as before, and the conformal scores it recalibrates
        # are its own.
        if self.args.cluster.enabled:
            self.fit_fixed_membership(model)
            # Rewrite the checkpoint with the membership beside the weights it
            # was fitted for, so supervised.pth alone reproduces the fixed stats.
            torch.save(model.state_dict(), best_model_path)
            val_fixed, test_fixed = self.evaluate_fixed_membership(
                model, model_id, criterion, flag, mae, acc, ch_acc
            )

        ch_acc = True if self.args.data.full_channels else ch_acc
        acc = False if self.args.data.full_channels else acc
        # Logged before the batch-protocol test stats: the grid's resume marker
        # (sl_test/interval_width_mean) must stay the last thing a run logs.
        if self.args.cluster.enabled:
            self.log_stats(val_fixed, flag, mae, acc, ch_acc, mode="val_fixed")
            self.log_stats(
                test_fixed, flag, mae, acc, ch_acc, mode="test_fixed",
                coverage=self.args.conf.conf,
            )
        self.log_stats(
            stats, flag, mae, acc, ch_acc, mode="test", coverage=self.args.conf.conf
        )

        # A fit-once model leaves no supervised.pth behind, so the estimated
        # parameters would otherwise be unrecoverable from the run. Persist
        # whatever provenance the model recorded (ARIMA: per-band order, params,
        # AIC and convergence flag) beside its log.
        if self.args.exp.no_train and hasattr(model, "fit_info"):
            info = model.fit_info()
            if info is not None:
                with open(os.path.join(self.log_dir, "fit_info.json"), "w") as f:
                    json.dump(info, f, indent=2)
        self.tuning_score = stats[
            f"{self.args.exp.tuning_metric}"
        ]  # For hyperparameter tuning

    def fit_fixed_membership(self, model):
        """CCM: fit the run's fixed membership on the training split.

        Every training window once, in order, through a fresh loader: the
        training loader shuffles, and may drop its last batch. Stored on the
        model (CCMForecaster.fit_fixed_membership) and returned.
        """
        loader = DataLoader(
            self.train_loader.dataset,
            batch_size=self.train_loader.batch_size,
            shuffle=False,
            num_workers=self.train_loader.num_workers,
            pin_memory=self.train_loader.pin_memory,
        )
        return model.fit_fixed_membership(
            model_input(self.args, batch, self.device) for batch in loader
        )

    def evaluate_fixed_membership(
        self, model, model_id, criterion, flag="sl", mae=False, acc=False,
        ch_acc=False,
    ):
        """CCM: validation and test under the model's fixed membership.

        Mirrors test(): the validation stats, then split-conformal recalibrated
        on validation with this same membership, then the test stats scored
        against it. Needs fit_fixed_membership first, and self.model_fwd to
        be model (or its compiled wrapper). Overwrites self.calibration_scores.

        Returns:
            (val_stats, test_stats), as evaluate() returns them.
        """
        # evaluate() forwards through self.model_fwd, not model: were they
        # different modules, the batch protocol would be scored as the fixed one.
        if getattr(self.model_fwd, "_orig_mod", self.model_fwd) is not model:
            raise RuntimeError(
                "evaluate_fixed_membership needs self.model_fwd to run model; "
                "set it to the model the fixed membership was fitted on."
            )
        with model.fixed_membership():
            val_stats = self.evaluate(
                model=model,
                model_id=model_id,
                loader=self.val_loader,
                criterion=criterion,
                flag=flag,
                mae=mae,
                acc=acc,
                ch_acc=ch_acc,
                conformal_calibrate=False,
                coverage=False,
            )
            if self.args.conf.conf:
                self.evaluate(
                    model=model,
                    model_id=model_id,
                    loader=self.val_loader,
                    criterion=criterion,
                    flag=flag,
                    mae=mae,
                    acc=acc,
                    ch_acc=ch_acc,
                    conformal_calibrate=True,
                    coverage=False,
                )
            test_stats = self.evaluate(
                model=model,
                model_id=model_id,
                loader=self.test_loader,
                criterion=criterion,
                flag=flag,
                mae=mae,
                acc=acc,
                ch_acc=ch_acc,
                conformal_calibrate=False,
                coverage=self.args.conf.conf,
            )
        return val_stats, test_stats

    def evaluate(
        self,
        model: nn.Module,
        model_id: str,
        loader: DataLoader,
        criterion: nn.Module,
        flag: str,
        mae=False,
        acc=False,
        ch_acc=False,
        calibrate=False,
        conformal_calibrate=False,
        coverage=False,
        plot_results=False,
        num_plots=10
    ):
        """
        Evaluate the model return evaluation loss and/or evaluation accuracy.
        """
        self.print_master("Evaluating...")
        stats = dict()
        mae_loss = nn.L1Loss()
        num_examples = len(loader.dataset)  # Total number of examples across all ranks


        model.eval()
        with torch.no_grad():

            # Initialize Metrics
            total_loss = torch.tensor(0.0, device=self.device)
            total_mae = torch.tensor(0.0, device=self.device)
            all_logits = []
            all_labels = []
            all_ch_ids = []
            all_u = []

            n = len(loader.dataset)
            # Get num_plots random indices for plotting
            plotting_indices = random.sample(
                range(n), min(num_plots, n)
            ) if plot_results else []

            for i, batch in enumerate(loader):
                output = forward_pass(self.args, self.model_fwd, batch, model_id, self.device)
                y_hat, y, ch_ids, u = self.parse_output(output, batch)
                all_logits.append(y_hat)
                all_labels.append(y)
                all_ch_ids.append(ch_ids)
                all_u.append(u)

                # Loss. Forecast loss only, including on a cluster run: this
                # feeds stats["loss"], which is both the reported
                # {flag}_{val,test}/loss and the best-model / early-stopping
                # metric. See compute_aux_loss for why the similarity term is
                # deliberately not in it.
                loss = compute_loss(
                    output, batch, criterion, model_id, self.args, self.device
                )
                num_batch_examples = batch[0].shape[0]  # perf: plain int (see train loop)
                total_loss += loss * num_batch_examples

                # MAE
                # output and target are both (B, num_channels, pred_len) for
                # forecasting. Do NOT squeeze: squeeze(1) would collapse a size-1
                # channel axis (univariate, C=1) to (B, pred_len) and silently
                # broadcast against the (B, 1, pred_len) target, corrupting the MAE.
                if mae:
                    total_mae += (
                        mae_loss(output, hardware.to_device(batch[1], self.device)) * num_batch_examples
                    )

        preds = torch.cat(all_logits)
        targets = torch.cat(all_labels)

        if plot_results:
            self.plot(preds, targets, plotting_indices)

        if calibrate:
            return preds, targets, torch.cat(all_ch_ids)

        if conformal_calibrate:  # Conformal calibration
            return self.coverage(preds, targets, mode="calibrate")

        # Loss
        stats["loss"] = total_loss.item() / num_examples

        # Per-channel MSE (forecasting): compute MSE per channel, then average across
        # channels for the reported value. For equal-length channels this equals the
        # flattened loss above; the per-channel vector is logged so per-band
        # performance stays visible. preds/targets are (n_samples, num_channels, O).
        if self.args.exp.task == "forecasting" and preds.dim() == 3:
            per_channel_mse = ((preds - targets) ** 2).mean(dim=(0, 2))
            stats["mse"] = per_channel_mse.mean().item()
            stats["mse_per_channel"] = [round(float(v), 10) for v in per_channel_mse]

        self.print_master(
            f"Evaluation loss after all_reduce: {total_loss.item()}."
        )

        # MAE
        if mae:
            stats["mae"] = total_mae.item() / num_examples

        # Window Metrics
        if acc:
            window_metrics = get_metrics(
                self.args,
                all_logits,
                all_labels,
                mode="window",
            )
            channel = True if self.args.data.full_channels else False
            update_stats(
                stats,
                window_metrics,
                self.args.open_neuro.task,
                self.args.exp.other_metrics,
                channel,
            )

        # Channel Metrics
        if ch_acc:
            ch_metrics = get_metrics(
                self.args,
                all_logits,
                all_labels,
                all_ch_ids,
                all_u,
                mode="channel",
            )
            update_stats(
                stats,
                ch_metrics,
                self.args.open_neuro.task,
                self.args.exp.other_metrics,
                True,
            )

        # Coverage (conformal prediction)
        if coverage:
            preds = torch.cat(all_logits)
            targets = torch.cat(all_labels)
            coverage_metrics = self.coverage(
                preds,
                targets,
                mode="inference",
                return_intervals=self.args.conf.intervals,
            )
            stats["ic"] = coverage_metrics[0].item()
            stats["jc"] = coverage_metrics[1].item()

            if self.args.conf.intervals:
                stats["interval_width_mean"] = coverage_metrics[2].item()
                stats["interval_width_std"] = coverage_metrics[3].item()


        return stats

    def plot(self, preds, targets, indices):
        """
        Plot the predictions and targets for a given set of indices.
        """

        self.print_master("Plotting results...")

        for idx in indices:
            plt.figure(figsize=(10, 5))
            # preds[idx]/targets[idx] are (num_channels, pred_len); transpose so the
            # horizon is the x-axis and each channel is one line. Without the .T a
            # single-channel (1, pred_len) forecast makes plt.plot draw pred_len
            # single-point series at x=0 -- a blank frame with a degenerate x-axis
            # (this silently blanked every per-channel plot). A 1-D (pred_len,)
            # forecast is unaffected: numpy .T on a 1-D array is a no-op.
            plt.plot(preds[idx].cpu().numpy().T, label="Predictions")
            plt.plot(targets[idx].cpu().numpy().T, label="Targets")
            plt.title(f"Sample {idx}")
            plt.xlabel("Time (6min)")
            plt.ylabel("Normalized EMF Exposure (V/m)")

            # Save the plot to the log directory
            plot_path = os.path.join(self.log_dir, f"plot_{idx}.png")
            plt.savefig(plot_path)
            self.print_master(f"Saved plot {idx} to {plot_path}")

    def coverage(self, preds, targets, mode="calibrate", return_intervals=False):
        """
        For conformal time series forecasting. Obtain the critical nonconformity scores.

            preds: Model predictions (n_samples, pred_len, num_channels).
            targets: Target forecasts (n_samples, pred_len, num_channels).

        """
        preds = preds.detach().cpu()
        targets = targets.detach().cpu()

        # The conformal stack expects (n_samples, pred_len, num_channels). The models
        # emit (n_samples, num_channels, pred_len), so move the channel axis last; a
        # bare 2D (n_samples, pred_len) tensor just gains a singleton channel. This
        # makes the critical scores, coverage, and Bonferroni correction per
        # horizon-step and per channel.
        if preds.dim() == 2:
            preds = preds.unsqueeze(-1)
        elif preds.dim() == 3:
            preds = preds.transpose(1, 2).contiguous()

        if targets.dim() == 2:
            targets = targets.unsqueeze(-1)
        elif targets.dim() == 3:
            targets = targets.transpose(1, 2).contiguous()

        if mode == "calibrate":
            self.print_master("Calibrating conformal prediction model...")
            critical_scores, corrected_critical_scores = get_all_critical_scores(
                preds, targets, self.args.conf.alpha
            )
            self.calibration_scores = {
                "uncorrected": critical_scores,
                "corrected": corrected_critical_scores,
            }

            # Save calibration scores for 1st epoch (if validation tracking on)
            if not os.path.isfile(self.score_path) and self.args.conf.validation_eval:
                torch.save(self.calibration_scores, self.score_path)

        elif mode == "inference":
            self.print_master("Using calibration scores to obtain coverage values...")

            if self.args.conf.validation_eval:
                calibration_scores = torch.load(
                    self.score_path
                )  # Load calibration scores for best model
            else:
                calibration_scores = self.calibration_scores
            scores = (
                calibration_scores["corrected"].detach().cpu()
                if self.args.conf.corrected
                else calibration_scores["uncorrected"].detach().cpu()
            )

            self.print_master(
                f"Preds: {preds.device}. Targets: {targets.device}. Scores: {scores.device}."
            )

            ic, jc, intervals = get_coverage(
                preds, targets, scores
            )  # (independent_coverages, joint_coverages, intervals)
            ic_percent = ic.float().mean() * 100
            jc_percent = jc.float().mean() * 100

            # Compute interval widths
            interval_widths = (
                intervals[:, 1] - intervals[:, 0]
            )  # shape: [n_samples, pred_len, 1]

            print(f"Interval width shape: {interval_widths.shape}")
            mean_width = interval_widths.mean()
            std_width = interval_widths.std()

            print(f"Mean width: {mean_width.shape}. Std width: {std_width.shape}")

            if return_intervals:
                return torch.stack([ic_percent, jc_percent, mean_width, std_width]).to(
                    self.device
                )
            else:
                return torch.stack([ic_percent, jc_percent]).to(self.device)

    def parse_output(self, output, batch):
        ch_ids = torch.tensor(0, device=self.device).unsqueeze(-1)
        u = torch.tensor(0, device=self.device).unsqueeze(-1)
        y_hat, y = (output, hardware.to_device(batch[1], self.device))

        if y_hat.dim() > 1 and y_hat.size(-1) == 1:
            y_hat = y_hat.unsqueeze(-1)

        return (y_hat, y.to(self.device), ch_ids.to(self.device), u.to(self.device))

    def log_stats(
        self, stats, flag, mae, acc, ch_acc, mode, task="binary", coverage=False
    ):
        modes = {
            "val": "Validation",
            "test": "Test",
            "val_fixed": "Validation (fixed membership)",
            "test_fixed": "Test (fixed membership)",
        }
        Mode = modes[mode]
        mapping = get_logger_mapping(self.args.open_neuro.task)

        loss = stats["loss"]
        self.print_master(f"Model {Mode} Loss: {loss:.6f}")
        epoch_logger(self.args, self.logger, f"{flag}_{mode}/loss", loss)

        # Channel-averaged MSE (MSE per channel, then averaged across channels). The
        # per-channel vector is stored once at test time for per-band inspection.
        if "mse" in stats:
            self.print_master(f"Model {Mode} MSE (channel-averaged): {stats['mse']:.6f}")
            epoch_logger(self.args, self.logger, f"{flag}_{mode}/mse", stats["mse"])
            if (
                mode in ("test", "test_fixed")
                and "mse_per_channel" in stats
                and not self.args.exp.neptune
            ):
                self.logger[f"{flag}_{mode}/mse_per_channel"] = stats["mse_per_channel"]

        if mae:
            mae_value = stats["mae"]
            self.print_master(f"Model {Mode} MAE: {mae_value:.6f}")
            epoch_logger(self.args, self.logger, f"{flag}_{mode}/mae", mae_value)
        if acc and not self.args.open_neuro.ch_aggr:
            acc_value = stats["acc"] * 100
            self.print_master(f"Model {Mode} Accuracy: {acc_value:.2f}%")
            epoch_logger(self.args, self.logger, f"{flag}_{mode}/accuracy", acc_value)

            if self.args.exp.other_metrics:
                for key, value in mapping.items():
                    self.logger[f"{flag}_{mode}/{value}"] = stats[key]
        if ch_acc:
            ch_acc_value = stats["ch_acc"] * 100
            self.print_master(f"Model {Mode} Channel Accuracy: {ch_acc_value:.2f}%")
            epoch_logger(
                self.args, self.logger, f"{flag}_{mode}/channel_accuracy", ch_acc_value
            )

            if self.args.exp.other_metrics:
                for key, value in mapping.items():
                    self.logger[f"{flag}_{mode}/channel_{value}"] = stats[f"ch_{key}"]

        if coverage:
            self.print_master(f"Model {Mode} Independent Coverage: {stats['ic']:.2f}%")
            self.print_master(f"Model {Mode} Joint Coverage: {stats['jc']:.2f}%")
            epoch_logger(self.args, self.logger, f"{flag}_{mode}/ic", stats["ic"])
            epoch_logger(self.args, self.logger, f"{flag}_{mode}/jc", stats["jc"])

            if self.args.conf.intervals:
                self.print_master(
                    f"Model {Mode} Interval Width Mean: {stats['interval_width_mean']:.2f}"
                )
                self.print_master(
                    f"Model {Mode} Interval Width Std: {stats['interval_width_std']:.2f}"
                )
                epoch_logger(
                    self.args,
                    self.logger,
                    f"{flag}_{mode}/interval_width_mean",
                    stats["interval_width_mean"],
                )
                epoch_logger(
                    self.args,
                    self.logger,
                    f"{flag}_{mode}/interval_width_std",
                    stats["interval_width_std"],
                )

    def count_parameters(self):
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)

    def print_master(self, message):
        """
        Prints statements to the rank 0 node.
        """
        self.console.log(message)
