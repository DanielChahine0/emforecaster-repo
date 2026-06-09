# How the Models Work, Explained Simply

Welcome! This folder explains **every forecasting model** in the EMForecaster
project in **very simple English**. No math. No jargon. You do not need any
machine-learning background. If you can picture guessing tomorrow's weather
from the last few days, you can understand everything here.

Each model has the same job. It looks at a window of the recent past. Then it
predicts a window of the future. In this project the thing being predicted is
**EMF exposure**. That is how strong the radio-wave signal from wireless
networks is over time. The models start very simple. The simplest ones just
take a kind of weighted average to guess. They build up to the clever star of
the paper, **EMForecaster**.

> Every explanation is based on the **actual code** in this repository. It is
> not based on the textbook ideal. So where this codebase's version is simpler,
> stranger, or even buggy, the file says so plainly. (See "Honest notes" at the
> bottom.)

---

## 📖 Suggested reading order

**1. Start with the big picture (read these first):**

| File | What it covers |
|------|----------------|
| [01: The Big Picture](01-the-big-picture.md) | What EMF forecasting is, and the past-window to future-window setup that every model shares |
| [02: Key Ideas Explained](02-key-ideas-explained.md) | A plain-English word list: normalization (RevIN), patching, trend and seasonal decomposition, positional tags, and attention |
| [03: Uncertainty Quantification](03-uncertainty-quantification.md) | How the project draws an honest "trust band" around each forecast (conformal prediction) and scores it |

**2. The simple baselines.** These are the easy models that the good ones are
supposed to beat:

| File | One-line idea |
|------|---------------|
| [10: MLP](10-mlp.md) | The tiniest baseline: a two-step weighted guesser with one bend |
| [11: CNN](11-cnn.md) | A simple CNN whose filters span the *whole* input window |
| [12: LSTM](12-lstm.md) | A memory-notebook reader (with an honest quirk: it walks the batch, not time) |
| [13: Transformer](13-transformer.md) | An attention-based baseline with some odd extra convolution layers |

**3. The real forecasters.** These are serious, modern models:

| File | One-line idea |
|------|---------------|
| [20: Linear](20-linear.md) | The simplest possible forecaster: one weighted blend of the past |
| [21: DLinear](21-dlinear.md) | Splits the past into a smooth trend plus leftover bumps, predicts each, then adds them |
| [22: TSMixer](22-tsmixer.md) | An all-MLP model that takes turns mixing across time, then across channels |
| [23: PatchTST](23-patchtst.md) | A chunk-based transformer: cut the timeline into patches, then let them compare notes |
| [24: TimesNet](24-timesnet.md) | Finds repeating cycles, folds the timeline into a grid, and scans it like an image |
| [25: ModernTCN](25-modern-tcn.md) | A modern convolutional model with very wide "pattern stamps" |
| [26: RecurrentModel](26-recurrent-model.md) | A flexible RNN/GRU/LSTM reader that walks the past values properly |

**4. The star of the show:**

| File | One-line idea |
|------|---------------|
| [30: EMForecaster](30-emforecaster.md) | The main model, built from parts: **patch, then summarize, then remember order, then forecast**. This is the model the paper is named after. |

**5. How long each model takes to run.** These times were actually measured,
and each one has a per-component breakdown:

| File | What it covers |
|------|----------------|
| [40: Runtime breakdown](40-runtime-breakdown.md) | Timing for each model, plus which internal part eats the time |
| [41: Runtime comparison](41-runtime-comparison.md) | Every model's runtime in one side-by-side table |

> These numbers were produced by [`benchmark_runtime.py`](benchmark_runtime.py)
> on a laptop CPU. Re-run it on your own GPU to get your real times. The order
> of the models and the component breakdown hold no matter what hardware you
> use.

---

## 🧭 If you only read three files

1. [01: The Big Picture](01-the-big-picture.md) (what is going on)
2. [21: DLinear](21-dlinear.md) (the "aha" idea that simple often wins)
3. [30: EMForecaster](30-emforecaster.md) (the model this whole project is about)

---

## 🔍 Honest notes (real quirks found in the code)

While writing these, a few models turned out to behave differently from how
they are usually described. The explanations are written to match reality:

- **Linear**. Its built-in normalization option *was* broken. It asked for a
  value that was never created, so it would crash on startup. ✅ **Fixed on
  2026-06-09.** See [20: Linear](20-linear.md) and `../KNOWN_ISSUES.md`.
- **LSTM baseline**. Because of how the input is shaped, it runs its memory
  over the *batch of examples* instead of over *time within a window*. So it
  does not really learn the time pattern an LSTM is meant for. See
  [12: LSTM](12-lstm.md).
- **ModernTCN**. The copy here is wired for *classification*. The project
  reuses that path for forecasting. A couple of its forecasting parts (a RevIN
  normalizer and a dedicated forecast head) are built but never used. See
  [25: ModernTCN](25-modern-tcn.md).

These notes are about the *baselines and side models*. The project's star
model, **EMForecaster**, and its main competitors (DLinear, TSMixer, PatchTST,
TimesNet) all behave as intended.
