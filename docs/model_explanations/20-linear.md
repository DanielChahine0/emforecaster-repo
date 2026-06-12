# Linear (Linear)

**In one sentence:** This is the simplest forecaster there is. It turns the whole past window into the whole future window using one fixed "recipe" of weights, and nothing more.

## The simple idea

Imagine you want to guess tomorrow's temperatures. You have the last 96 hourly readings written down. Here is the simplest trick you could try. For each future hour you want, just mix all 96 past numbers together using fixed amounts. "Take a little bit of this one. Take a lot of that one. Take a tiny bit of this other one. Add them all up. That is my guess."

Every future number gets its own mixing recipe. The recipes never change once the model has learned them. There are no clever tricks. There is no memory. The model never looks at small pieces. It just blends the past with fixed weights to make the future.

That is exactly what this model is. In this project the past is a window of recent EMF exposure readings (how strong the radio-wave signal is over time). The future is the next stretch of those readings.

## How it actually works

Looking at the real code, the whole model is just a few lines:

1. It takes the past window of numbers. (For example, 96 past readings going in.)
2. It "squeezes" the input. This just removes an empty extra dimension so the numbers line up neatly as a flat list.
3. It cleans up the numbers first with a normalization step. (Normalize means rescale the numbers so they sit in a tidy, centered range before mixing.) By default it uses "layer" normalization. You can also choose "batch1d". You can also turn it off so the numbers pass straight through.
4. It applies one **linear layer**. (This is a single step that multiplies every input number by a learned weight and adds them all up.) This produces the future window directly, say 24 future readings coming out.
5. Those output numbers are the forecast. There is no second step.

So it really takes, for example, 96 past numbers and produces 24 future numbers, all in one shot.

**A note on the "norm_mode" option.** The model can *normalize* the input first. (Normalize means rescale the numbers so they sit in a tidy, centered range before mixing.) There are three choices. "layer" (the default) and "batch1d" both do real normalization. Any other value turns it off, so the numbers pass straight through. (Heads up if you read older code or the git history. This option used to be broken. It would crash the model the moment you built it. It was fixed on 2026-06-09. See `KNOWN_ISSUES.md` at the repo root for the details.)

## Why you'd use it / its trade-offs

**Strengths:** It is tiny, fast, and very hard to get wrong. It often does surprisingly well. That is because many real signals trend smoothly, and a plain weighted blend captures that.

**Weaknesses:** It can't react to *patterns*. It can only react to fixed positions. It has no memory. It can't tell a rising trend apart from a falling one if both pass through the same positions. It can't separate slow drift from repeating waves the way [DLinear](21-dlinear.md) can. It is a bare-bones model on purpose. Think of it as a yardstick, not a serious contender.

## In this project

Think of this as the **floor**, the simplest sanity-check model there is. If a fancy model can't beat a plain weighted blend, something is wrong. It reads a window of past EMF readings and produces a window of future EMF readings. One thing to know. This model is *available* in the framework, but it is **not** one of the models actually run in the project's main EMF experiments. Those use the four baselines plus DLinear, TSMixer, PatchTST, and the star [EMForecaster](30-emforecaster.md).

## See also

- [DLinear](21-dlinear.md), the smarter linear cousin that splits trend from wiggle first
- [The big picture](01-the-big-picture.md), what forecasting means here
- [Key ideas explained](02-key-ideas-explained.md), what normalization actually does
