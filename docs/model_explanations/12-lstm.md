# LSTM (RF_EMF_LSTM)

**In one sentence:** This is a small "reader with a memory notebook." It looks at a window of recent EMF (electromagnetic field exposure) values. Then it writes down its guess for the next stretch of values.

## The simple idea

Imagine someone reading a story out loud. As they read, they keep a little notebook. They jot down what seems important. They also cross out things that stop mattering. By the end, the notebook holds a summary of everything they think is worth remembering.

An LSTM (Long Short-Term Memory) works like that notebook keeper. It is a part of a neural network. It is good at handling things that come in order, one step after another. As it reads, it decides what to keep and what to let go. The clever part is the "forget" and "remember" choices. They let it hold onto important hints without getting cluttered.

In this project, that "story" is a list of recent EMF readings. The "guess" is what the readings will look like a little later on.

## How it actually works

Here is what the code really does, step by step:

1. **It takes the recent window.** The model receives a batch of short clips. The data comes in shaped like (32 examples, 1, 20 numbers). That means 32 separate windows. Each one is a window of past EMF numbers (for example, the last 20 readings).

2. **The memory keeper reads step by step, but down the wrong line.** The LSTM still does its real walk step by step. It jots in its notebook at each step. But here is the quirky part. Because of how the code sets it up, it does not walk along the 20 values *inside* one window. Instead it walks across the 32 *different examples* in the batch by accident. It treats them as if they were 32 moments in time. At each "step" it reads one whole 20-value window. So it really does take 32 steps with its memory notebook. But those steps are across unrelated examples. They are not across time inside a single window.

3. **The memory keeper makes notes.** At each step the LSTM updates a fixed-size bundle of notes. In the code this bundle has a chosen size, like 32 slots. Think of it as a running summary.

4. **A final layer turns notes into a forecast.** A plain linear layer reads the summary. It is a weighted sum, which means it mixes the notes together and gives each one its own importance. It then produces the future window of numbers, for example the next 20 values.

5. **It tidies the shape and returns the answer.** The output is squeezed into a clean list of predicted future values.

So the full trip is simple. 20 past numbers go in. A summary of 32 notes sits in the middle. 20 future numbers come out.

## Why you'd use it / its trade-offs

**Strengths:** LSTMs are a classic, trusted tool for sequences. They are easy to set up. They give a reasonable baseline so you can compare fancier models against them.

**Weaknesses here:** The LSTM *does* use its step-by-step memory. But it points that memory in the wrong direction. It walks across the different examples in the batch. It should walk across the moments inside a single window instead. So it never really learns the time pattern *within* one window of EMF readings. And that is the whole point of using an LSTM for forecasting. Each example's window is still handled as one fixed block of inputs. So it is honest to call this a simple baseline. It is not a showcase of what LSTMs can do.

## In this project

This is one of the four simple baselines. These are the easy models that the better ones should beat. Its `model_id` is `RF_EMF_LSTM`. It reads a window of past EMF exposure values. Then it produces a window of future EMF values. It is meant as a yardstick, not the star of the paper.

## See also

- [The recurrent model](26-recurrent-model.md). This is a fuller version that uses the step-by-step memory idea properly.
- [The big picture](01-the-big-picture.md). This explains what forecasting means in this project.
- [Key ideas explained](02-key-ideas-explained.md). These are the shared building blocks across the models.
