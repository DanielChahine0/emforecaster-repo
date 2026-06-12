# The Big Picture

**In one sentence:** This project teaches different "guessing machines" to look at recent measurements of wireless signal exposure and predict what the next stretch of measurements will be.

## What is a "time series"?

A time series is just a list of numbers. The numbers are measured one after another, over time. Think of the temperature outside, written down once every hour. Or the number of steps you walk, counted every minute. The order matters. Each number comes right after the one before it.

The most important idea here is simple. If you know what just happened, you can often make a good guess about what happens next. The weather tomorrow usually looks a lot like today. A song's next note usually follows the melody so far. That kind of guessing is called **forecasting** (predicting future values from past ones).

## What does this project forecast?

This project forecasts **EMF exposure**. EMF stands for electromagnetic field. It is the invisible radio energy that phones, cell towers, and Wi-Fi give off so your devices can talk to each other. Special sensors measure how strong this energy is at one spot. They do this again and again, hour after hour. That gives a time series of exposure values. The values are measured in volts per meter, which is a way to say "how strong the signal is here".

Why care? More cell towers and faster networks (like 5G) keep getting switched on. People want to know that exposure stays within safe limits. If we can predict exposure ahead of time, network operators can plan better and keep things safe. So the goal is simple. Given the recent exposure readings, guess the upcoming ones.

In the code, the data comes from real monitoring stations. They sit in places like Turin, Novara, and other Italian and Turkish sites (see `dataloading.py`). It is mostly **one signal over time**, which is the exposure at one place. That is why this is called univariate forecasting. It means just one thing is being tracked, not many at once.

## The setup every model shares

Every model in this folder plays the exact same game. Picture sliding a window along the line of numbers. It is like dragging a magnifying glass across a graph.

1. The model is shown a chunk of recent past readings. The length of that chunk is the **sequence length** (how many past numbers it gets to see).
2. The model must guess the next chunk of readings. The length of that guess is the **prediction length** (how many future numbers it must produce).

So a typical job is this. Take 96 past numbers, then produce 24 future numbers. That's it. Look back, then look ahead.

How does the code build these chunks? In `dataloading.py`, the long recording is first split into three parts. There is a big part for **training** (practice). There is a smaller part for **validation** (a check up). And there is a part for **testing** (the final exam). Then the code slides the window across each part. Each window is cut in two. The first piece becomes the "past" the model sees. The leftover piece becomes the "correct future answer" the model is graded against. The numbers are also gently rescaled so they sit around zero. This just makes learning smoother. It is called normalization.

## How a model learns to guess better

The models start out guessing badly. Training fixes that. It is like practising free throws until you stop missing.

In `exp.py`, training works like this. The model is shown one batch of examples (a handful of past window and future window pairs at a time). It makes its guesses. Then a **loss** is measured. The loss is a single number that says how wrong the guess was. Next, the model's internal settings are nudged a tiny bit. They move in the direction that would have made the guess better. This repeats over and over, for many passes through the data. Each full pass is called an epoch. Once in a while the model is checked on the validation part, which it never trained on. The best version is saved.

## All these models are different "guessing machines"

That's the whole point of this folder. Every model is a different design trying to win the same game. Some are very simple (almost a fancy weighted average). Some are clever and modern. The star of the project's paper is **EMForecaster**. It is the serious contender. The simpler ones are mostly there as honest yardsticks to beat.

The project also does something extra. It doesn't just guess a single number. It can also say **how sure it is**, by drawing a band around its guess. That's the uncertainty part. It is explained in its own file.

## Where to go next

- New to the building blocks? Read [Key ideas explained](02-key-ideas-explained.md). It covers shared tricks like normalization, patching, and attention in plain words.
- Curious how the models say "how sure am I"? See [Uncertainty quantification](03-uncertainty-quantification.md).
- Want the simplest models first (good warm ups and yardsticks)? Start with [MLP](10-mlp.md), then [CNN](11-cnn.md), [LSTM](12-lstm.md), and [Transformer](13-transformer.md).
- Ready for the real forecasting models? Try [Linear](20-linear.md), [DLinear](21-dlinear.md), [TSMixer](22-tsmixer.md), [PatchTST](23-patchtst.md), [TimesNet](24-timesnet.md), [ModernTCN](25-modern-tcn.md), and [RecurrentModel](26-recurrent-model.md).
- For the main attraction, go to [EMForecaster](30-emforecaster.md).
