# Simple Weighted Sum Guesser (RF_EMF_MLP)

**In one sentence:** This is the smallest model in the project. It just mixes all the recent past numbers together, adds one small bend, and mixes them again to guess the future.

## The simple idea

Imagine you want to guess tomorrow's temperature. Here is a very simple way. Take the last 20 days. Give each day a weight, so some days count more and some count less. Add them all up. That sum is your guess. That is basically what this model does.

The "MLP" name stands for *multi-layer perceptron*. That is a fancy term for a small stack of mixing steps. But do not be scared by the name. Here the model is tiny. It has just two mixing steps with one gentle squeeze in between.

Think of it as a student who has not learned any clever tricks yet. They look at all the past numbers at once. Then they make a quick best guess. Sometimes that guess is surprisingly good. It is the score everyone else has to beat.

## How it actually works

The model takes a window of past values. It then produces a window of future values. In this project the default is simple. Take 20 past numbers, then produce 20 future numbers.

1. **Mix the past into a hidden set of numbers.** The first step is called a *Linear* layer. It just multiplies things by weights and adds them up. It takes all 20 past numbers and blends them into 32 new "summary" numbers. Each summary number is its own blend of the whole past window.
2. **Add one gentle bend.** Next it passes those 32 numbers through a *Tanh* function. Tanh is just a soft squashing curve. It gently pushes every number to sit between -1 and +1. Big numbers get flattened. Small numbers stay roughly the same. This one small bend is the only part of the model that is not a straight line. Without it, the model could only draw perfectly straight relationships.
3. **Mix again into the future.** The second Linear step takes those 32 squashed numbers. It blends them into the 20 future numbers, which is the forecast.

That is the entire model. Two mixing steps and one squeeze. The squeeze at the very end is removed (the code "squeezes" away an empty dimension). So you are left with a clean list of future numbers.

## Why you'd use it and its trade-offs

**Strengths:**
- It is very simple and fast. It has very few numbers to learn.
- It is hard to get wrong, and it is a great honesty check. If a big fancy model cannot beat this little one, something is off.
- The middle bend lets it catch *some* curved patterns. A pure straight-line model cannot do that.

**Weaknesses:**
- It treats the past window as one big bag of numbers. It has no real sense of *time order*. It does not know that "yesterday" comes right before "today."
- It cannot spot repeating shapes or trends on purpose, the way the bigger models can.
- With only one bend, it cannot learn very twisty patterns.

## In this project

This is a **baseline**. That means it is a model that is simple on purpose, and its job is to be beaten. The serious models (like [EMForecaster](30-emforecaster.md) and [PatchTST](23-patchtst.md)) should do better than this. If they cannot, that tells the researchers their fancy ideas are not actually helping.

It reads a window of recent EMF (electromagnetic field) exposure readings. Then it outputs a window of predicted future readings. Its code id is `RF_EMF_MLP`, and it lives next to its sibling baselines in `rf_emf_baselines.py`.

## See also

- [CNN baseline](11-cnn.md). This is another simple baseline that, in this code, also ends up being a fancy weighted sum.
- [Linear](20-linear.md). This one is even simpler. It is a single straight-line mix, with no bend at all.
- [The big picture](01-the-big-picture.md). This explains what forecasting means in this project.
