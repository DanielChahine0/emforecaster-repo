# The CNN Baseline (RF_EMF_CNN)

**In one sentence:** This model looks at the whole recent past at once through a few different "lenses." It gets one number from each lens. Then it turns those few numbers into the forecast.

## The simple idea

Imagine you have a line on a graph. It shows how strong a radio signal has been for the last little while. (EMF is the everyday radio energy in the air.) You want to guess what comes next.

A normal CNN works like sliding a tiny magnifying glass along the line. (CNN stands for "convolutional neural network." It is a model that slides a small pattern finder along the data.) The glass is small. So it only sees a few points at a time. It slides from start to end, looking for little shapes, like a small bump or a dip.

Here is the honest twist in this code. The magnifying glass is **as wide as the whole window**. It does not slide at all. It sits over every past point at once. It squeezes them all into a single number. So it does not find small local shapes. Instead, each lens is really just a **weighted sum** of the whole past window. A weighted sum means it multiplies each past value by its own importance number. Then it adds them all up.

The model has several of these lenses side by side (32 by default). Each one weights the past in its own way. So you end up with a handful of summary numbers. They describe the recent past from different angles.

## How it actually works

Here are the real steps the code takes, in plain words:

1. **Start with the past window.** The input is a batch of recent EMF readings. For example, 20 past numbers per example.
2. **Add a "channel" slot.** The code reshapes the data so it has one channel (one signal track). This is just bookkeeping for the next step.
3. **Apply the wide lenses.** Each of the 32 lenses covers the full window. Each lens is exactly as wide as the input. So it produces just **one number** per lens. This way, 20 past numbers become 32 summary numbers.
4. **Squash with Tanh.** Tanh keeps the values in a tidy range. (Tanh is a function that gently bends any number to sit between -1 and +1.) It also lets the model bend, not just add.
5. **Flatten.** The 32 numbers are lined up in a row.
6. **Final mapping.** One plain layer turns those 32 numbers into the forecast, for example 20 future numbers. (This is a "Linear" layer, which is another weighted sum plus a shift.)

So the whole journey is: 20 past numbers turn into 32 summaries, and then into 20 future numbers.

## Why you'd use it / its trade-offs

**Strengths:** It is tiny, fast, and easy to train. It is a clean, simple yardstick to measure fancier models against.

**Weaknesses:** The lenses never slide. So this model does **not** spot small repeating shapes the way a real CNN does. It is closer to a few different weighted averages of the past, bent a bit, then combined. That is not very expressive. So it usually will not beat the serious models.

## In this project

This is one of four simple **baselines**. (The others are MLP, LSTM, and Transformer.) Its job is to be a basic comparison point, not the winner. It reads a window of recent EMF values. Then it produces a window of future EMF values. The real star of the project is [EMForecaster](30-emforecaster.md). The CNN is there so the team can show their main model does better than easy options.

## See also

- [The MLP baseline](10-mlp.md). This is an even simpler sibling, just plain weighted sums.
- [Key ideas explained](02-key-ideas-explained.md). This covers shared building blocks like Tanh and weighted sums.
- [The big picture](01-the-big-picture.md). This explains what forecasting means in this project.
