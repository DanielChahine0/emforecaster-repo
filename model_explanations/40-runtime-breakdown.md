# How Long Each Model Takes, and Which Part Is Slow

This file gives a **rough, measured estimate** of how long each model takes to
run. It also **breaks down which inside part (component) eats the time**. For a
side by side table of all models, see
[41: Runtime comparison](41-runtime-comparison.md).

> ### 📏 How these numbers were measured (please read)
> - **Real measurement, not a guess.** Each model was built with its *actual*
>   experiment config. Then it was timed on a real batch of the EMF data. The
>   script that produced these numbers is right here:
>   [`benchmark_runtime.py`](benchmark_runtime.py). You can re-run it on your own
>   machine or GPU.
> - **Hardware:** this was run on a **laptop CPU** (Mac, PyTorch 2.11, 4
>   threads). Your training probably uses a **GPU/MPS** (the configs say
>   `mps: True`). That makes the *heavy* models far faster. The **relative
>   ordering and the component breakdown stay the same**. Only the absolute
>   seconds shrink.
> - **Same setup for everyone:** input window 336 steps, then predict 96 steps,
>   1 channel, batch size 512. The data makes **56 training batches, 8
>   validation, 16 test** per epoch.
> - **What the time numbers mean:**
>   - *forward* = making a prediction (one batch).
>   - *forward+backward* = predict **and** learn from it (one training step).
>   - *per epoch* = one full pass over the training data plus a validation pass.
>   - *per seed (100 epochs)* = a full training run for one random seed.
>   - *full run (5 seeds)* = the whole experiment (the configs use 5 seeds).
> - **These are upper bounds.** Training is capped at 100 epochs. But **early
>   stopping** (patience 20; the Transformer uses 10) usually stops much sooner.
>   So real runs are often shorter. Data loading and logging overhead are not
>   included.
> - The component breakdown is for the **forward pass**. It is shown as both
>   milliseconds and a percentage of that pass.

---

## MLP baseline: `RF_EMF_MLP`

The smallest model. ~14k weights. ([explanation](10-mlp.md))

| Component | What it is | Time | Share |
|-----------|-----------|------|-------|
| `fc1` | first weighted sum layer | ~0.1 ms | 44% |
| `act` | Tanh bend | ~0.1 ms | 33% |
| `fc2` | second weighted sum layer | ~0.1 ms | 24% |

- **Forward:** 0.3 ms/batch · **Train step:** 0.9 ms/batch
- **Per epoch:** ~0.05 s · **Per seed (100 ep):** ~6 s · **Full run (5 seeds):** **~25 s**
- *Why so fast:* it is three tiny operations. Everything here is below the
  measurement noise floor. It is basically instant.

---

## CNN baseline: `RF_EMF_CNN`

~14k weights. ([explanation](11-cnn.md))

| Component | What it is | Time | Share |
|-----------|-----------|------|-------|
| `conv1` | the (full-width) convolution filters | ~1.6 ms | **85%** |
| `act` | Tanh bend | ~0.2 ms | 8% |
| `fc` | final weighted sum layer | ~0.1 ms | 7% |

- **Forward:** 1.9 ms/batch · **Train step:** 3.4 ms/batch
- **Per epoch:** ~0.2 s · **Per seed (100 ep):** ~20 s · **Full run (5 seeds):** **~1.7 min**
- *Why:* almost all the (tiny) cost is the convolution. Still basically instant.

---

## LSTM baseline: `RF_EMF_LSTM`

~51k weights. ([explanation](12-lstm.md))

| Component | What it is | Time | Share |
|-----------|-----------|------|-------|
| `lstm` | the memory-notebook recurrence | ~7.0 ms | **99%** |
| `fc` | final weighted sum layer | ~0.1 ms | 1% |

- **Forward:** 7.1 ms/batch · **Train step:** **73 ms/batch** (≈10× the forward!)
- **Per epoch:** ~4.2 s · **Per seed (100 ep):** ~7 min · **Full run (5 seeds):** **~35 min**
- *Why the huge jump from forward to train:* an LSTM learns by "unrolling" its
  memory backwards over every step in its sequence. Because of the quirk
  documented in [12-lstm.md](12-lstm.md), its "sequence" is the **whole batch of
  512**. So the backward (learning) pass unrolls over 512 steps. That is what
  makes the training step ~10× slower than just predicting.

---

## Transformer baseline: `RF_EMF_Transformer` 🔴 (slowest of all)

~373k weights. ([explanation](13-transformer.md))

| Component | What it is | Time | Share |
|-----------|-----------|------|-------|
| `conv1` | first tail convolution | ~378 ms | **41%** |
| `conv2` | second tail convolution | ~318 ms | **35%** |
| `enc` | the attention blocks | ~214 ms | 23% |
| `act`, `fc1`, `fc2` | bends plus small linear layers | ~3 ms | <1% |

- **Forward:** 913 ms/batch · **Train step:** 1140 ms/batch
- **Per epoch:** ~71 s · **Per seed (100 ep):** ~2 h · **Full run (5 seeds):** **~9.9 h**
- *Why so slow:* this is surprising. The **two tail convolution layers eat
  ~76%** of the time. That is *more than the attention itself*. Those `Conv1d`
  layers are wide (their in/out size equals the 336-long window), so they are
  expensive. This quirky design (convolutions bolted on after the transformer
  blocks) is the single biggest reason this baseline is the slowest model in the
  whole project. (Early stopping uses patience 10 here, so real runs likely end
  well before 100 epochs.)

---

## DLinear: `DLinear`

~65k weights. ([explanation](21-dlinear.md))

| Component | What it is | Time | Share |
|-----------|-----------|------|-------|
| `decomposition` | the moving-average split (trend vs bumps) | ~2.9 ms | **73%** |
| `Linear_Seasonal` | predicts the "bumps" part | ~0.6 ms | 15% |
| `Linear_Trend` | predicts the smooth-trend part | ~0.5 ms | 13% |

- **Forward / train step:** ~3 ms/batch (both near the noise floor)
- **Per epoch:** ~0.2 s · **Per seed (100 ep):** ~18 s · **Full run (5 seeds):** **~1.5 min**
- *Why so fast:* it is basically two small weighted sum layers. Even the split
  into trend and bumps (a moving average) costs only a few milliseconds. It is
  one of the fastest models. That is exactly why it is such a strong "simple"
  benchmark.

---

## TSMixer: `TSMixer`

~54k weights. ([explanation](22-tsmixer.md))

| Component | What it is | Time | Share |
|-----------|-----------|------|-------|
| `backbone` | the time-mix / channel-mix blocks | ~9.7 ms | **83%** |
| `revin` | normalize in / un-normalize out | ~1.7 ms | 15% |
| `head` | final weighted sum layer | ~0.3 ms | 2% |

- **Forward:** 11.8 ms/batch · **Train step:** 35.7 ms/batch
- **Per epoch:** ~2.1 s · **Per seed (100 ep):** ~3.5 min · **Full run (5 seeds):** **~17 min**
- *Why:* the repeated "stir time, then stir channels" mixing blocks do almost
  all the work. Still light overall.

---

## PatchTST: `PatchTST`

~72k weights. ([explanation](23-patchtst.md))

| Component | What it is | Time | Share |
|-----------|-----------|------|-------|
| `backbone` | the transformer (attention over patches) | ~37.6 ms | **77%** |
| `revin` | normalize in / un-normalize out | ~8.1 ms | 17% |
| `pos_enc` | project patches plus add position tags | ~1.7 ms | 4% |
| `patcher` | cut the window into patches | ~1.4 ms | 3% |

- **Forward:** 48.8 ms/batch · **Train step:** 104 ms/batch
- **Per epoch:** ~6.2 s · **Per seed (100 ep):** ~10 min · **Full run (5 seeds):** **~52 min**
- *Why:* the attention transformer over the patches is the main cost. Cutting
  the window into patches first is what keeps this *much* cheaper than the toy
  Transformer baseline above.

---

## EMForecaster: `EMForecaster` 🔴 (heaviest model, the star)

**~2.4 million weights, by far the largest.** ([explanation](30-emforecaster.md))

| Component | What it is | Time | Share |
|-----------|-----------|------|-------|
| `patch_model` | the TSMixer patch-encoder (summarizes each patch) | ~323 ms | **92%** |
| `patch_act` | the GELU bend on the patch features | ~16 ms | 5% |
| `base_model` | the final big linear forecasting head | ~8 ms | 2% |
| `revin` | normalize in / un-normalize out | ~2.7 ms | 1% |
| `patcher`, `patch_norm` | cut into patches / (off here) | <0.5 ms | ~0% |

- **Forward:** 351 ms/batch · **Train step:** 933 ms/batch
- **Per epoch:** ~55 s · **Per seed (100 ep):** ~1.5 h · **Full run (5 seeds):** **~7.7 h**
- *Why so heavy:* almost all the time (**92%**) is the **TSMixer patch-encoder**.
  With these settings it runs a full TSMixer (hidden size 256, 2 mixing layers)
  across **~84 patches** of every window. That is where the 2.4M weights live.
  The final forecasting head is a large 21,504→96 layer, but it is only ~2% of
  the time. This is the price of the model's accuracy and flexibility. **On a GPU
  this drops a lot** (big matrix mixing is exactly what GPUs speed up). So the
  CPU number here is the worst case.

---

### Big-picture takeaways
- **Three models are basically free** (full run under ~2 minutes): MLP, CNN,
  DLinear. They are the cheap yardsticks.
- **Two are light or medium:** TSMixer (~17 min) and PatchTST (~52 min). LSTM
  sits around ~35 min, but it is slow mostly because of its backward pass.
- **Two are heavy:** EMForecaster (~7.7 h) and the Transformer baseline
  (~9.9 h) on CPU. The Transformer is slow because of a wasteful design (tail
  convolutions). EMForecaster is slow because it is genuinely big and powerful.
  It also speeds up the most on a GPU.

See [41: Runtime comparison](41-runtime-comparison.md) for everything in one table.
