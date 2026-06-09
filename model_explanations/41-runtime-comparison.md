# Runtime Comparison: All Models in One Table

Here is the timing for every model in the rf_emf experiments, side by side. These numbers were **measured** on a laptop CPU. Each model used its real config. Do you want the per-component breakdowns and the full method and caveats? See [40, Runtime breakdown](40-runtime-breakdown.md).

> **Read this first.** The numbers come from a **laptop CPU** (Mac, PyTorch 2.11,
> 4 threads). The batch size was 512. The window was 336 steps, and the model
> predicted 96 steps ahead, with 1 channel. The data was `DET_Monitoring`
> (56 train, 8 val, and 16 test batches per epoch). On a
> **GPU/MPS** the heavy models get *much* faster. The order stays the same.
> "Full run" means 100 epochs times 5 seeds. That is an **upper bound**, because
> early stopping usually ends training sooner. You can re-run it yourself with
> [`benchmark_runtime.py`](benchmark_runtime.py).

## Main comparison (fastest to slowest)

| Rank | Model | Weights | Forward /batch | Train step /batch | ~Per epoch | ~Per seed (100 ep) | ~Full run (5 seeds) | Tier |
|----:|-------|--------:|---------------:|------------------:|-----------:|-------------------:|--------------------:|:----:|
| 1 | [MLP](10-mlp.md) | 13.9k | 0.3 ms | 0.9 ms | 0.05 s | ~6 s | **~25 s** | ⚡ instant |
| 2 | [DLinear](21-dlinear.md) | 64.7k | ~3 ms | ~3 ms | 0.2 s | ~18 s | **~1.5 min** | ⚡ instant |
| 3 | [CNN](11-cnn.md) | 13.9k | 1.9 ms | 3.4 ms | 0.2 s | ~21 s | **~1.7 min** | ⚡ instant |
| 4 | [TSMixer](22-tsmixer.md) | 54.3k | 11.8 ms | 35.7 ms | 2.1 s | ~3.5 min | **~17 min** | 🟢 light |
| 5 | [LSTM](12-lstm.md) | 50.5k | 7.1 ms | 73 ms | 4.2 s | ~7 min | **~35 min** | 🟡 medium |
| 6 | [PatchTST](23-patchtst.md) | 72.1k | 48.8 ms | 104 ms | 6.2 s | ~10 min | **~52 min** | 🟡 medium |
| 7 | [EMForecaster](30-emforecaster.md) | **2.42M** | 351 ms | 933 ms | 55 s | ~1.5 h | **~7.7 h** | 🔴 heavy |
| 8 | [Transformer](13-transformer.md) | 373k | 913 ms | 1140 ms | 71 s | ~2 h | **~9.9 h** | 🔴 heavy |

## Where each model spends its time

| Model | Biggest time sink | Share | In plain words |
|-------|-------------------|:-----:|----------------|
| MLP | the linear layers | ~100% | three tiny steps, basically free |
| CNN | the convolution (`conv1`) | 85% | one full-width filter does almost everything |
| LSTM | the recurrence (`lstm`) | 99% | it walks one step at a time with memory; the **backward pass is the real cost** |
| Transformer | the **tail convolutions** (`conv1`+`conv2`) | **76%** | two wide conv layers cost *more than the attention*, so the design is wasteful |
| DLinear | the moving-average split | 73% | even that is only ~3 ms; the rest is two small linears |
| TSMixer | the mixing blocks (`backbone`) | 83% | the "stir time, stir channels" layers |
| PatchTST | the transformer (`backbone`) | 77% | attention over the patches |
| EMForecaster | the **TSMixer patch-encoder** (`patch_model`) | **92%** | it sums up ~84 patches with a full TSMixer, which is where its 2.4M weights live |

## How to read this

- **Cheap yardsticks (ranks 1 to 3).** MLP, DLinear, and CNN finish a *whole*
  experiment in under ~2 minutes. Run these freely.
- **Mid-weight (ranks 4 to 6).** TSMixer, LSTM, and PatchTST take minutes to under
  an hour for the full 5-seed experiment.
- **Heavy (ranks 7 to 8).** These are EMForecaster and the Transformer baseline.
  They are heavy for two very different reasons.
  - **EMForecaster** is heavy because it is **genuinely big and powerful**.
    It has 2.4M weights and a TSMixer that runs over every patch. It is also the
    model that benefits **most from a GPU**, so expect a large speed-up there.
  - **The Transformer baseline** is heavy because of an **inefficient design**
    (wide tail convolutions). It is not heavy because it is powerful. It is the
    slowest model, yet it is only a simple baseline.

## A few honest caveats

- **CPU vs GPU.** These are CPU seconds. The configs train on GPU/MPS. There the
  heavy models (EMForecaster, Transformer, PatchTST, and LSTM) speed up a lot.
  The tiny ones barely change. They are already limited by overhead, not math.
- **Early stopping** (patience 20, and 10 for the Transformer) usually stops
  before 100 epochs. So real runs are often **shorter** than the "full run" column.
- **Not included.** Data loading, logging and checkpointing, and the conformal
  uncertainty step at the end are left out. For the heavy models, the training
  math dominates anyway. For the tiny models, this fixed overhead may actually be
  the larger part.
- **Parallelism.** The 5 seeds are independent. So if you run them on 5 GPUs at
  once, the wall-clock time is roughly the per-seed number, not the full-run number.
