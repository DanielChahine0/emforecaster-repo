# TSMixer (TSMixer)

**In one sentence:** TSMixer forecasts the future by stirring its input in two directions over and over. First it stirs along time. Then it stirs across the different signals. It uses only simple little number mixers, with no fancy attention.

## The simple idea

Imagine you are making a sauce. You stir it one way to blend the timeline of flavors. Then you stir it another way to blend the ingredients together. Then you do it all again. After a few rounds of stirring back and forth, everything is well mixed. All the flavors get to talk to each other.

TSMixer does the same thing with numbers. It is given a window of recent past values. (Here, that is mostly EMF exposure over time.) It wants to guess the values that come next. To do that, it keeps "stirring" the data. First it blends across **time**, so a moment can borrow from earlier and later moments. Then it blends across **channels**, which are the different signals being tracked. Stir, stir, repeat. At the end it reads off the future.

Here is the clever part. There is no attention, no convolution, and none of the heavy machinery other models use. It is built almost entirely from "mixers." A mixer is a tiny stage. It takes a row of numbers and makes a new row of numbers by adding them up in learned amounts. That is it. The whole model is just a stack of these.

## How it actually works

Here is what really happens to your window of past numbers as it passes through, step by step:

1. **Steady the numbers first (RevIN).** Before anything else, the input is centered and scaled. It subtracts the average and divides by the spread. So each signal sits around zero with a normal size. This stops a loud day from drowning out a quiet day. (See [Key ideas explained](02-key-ideas-explained.md) for more on RevIN.)
2. **Enter a "ResBlock" (a stirring round).** The model has a few of these stacked in a row.
3. **Time mixing.** Inside the block, a tiny mixer looks along the time direction. It takes the whole timeline of one signal. It blends it down to a middle sized batch of numbers. Then it bends it gently. (This is GELU, a soft on and off curve. It lets useful signals through and softens the rest.) Then it blends it back to the original length. This lets each moment in time pull in information from every other moment.
4. **Add the input back (residual).** The result is added on top of what went in. So the block only has to learn a small *adjustment*. It does not have to rebuild everything. This keeps training stable.
5. **Channel mixing.** A second tiny mixer now looks across the different signals at each moment. It blends them together. (Again it uses the GELU bend and a blend back step.) Again it adds its result back on top.
6. **Repeat.** Steps 3 to 5 are one ResBlock. The numbers flow through several of them, one after another. They get more and more thoroughly mixed.
7. **Read off the future.** A final simple layer (the "head") takes the stirred timeline and maps it straight to the future length you asked for. So if you fed in 512 past steps and want 96 future steps, this stage turns the time axis from 512 into 96.
8. **Undo the steadying (optional).** If turned on, RevIN puts the average and spread back. So the prediction is on the same scale as the real data.

Tiny shape sketch: it takes a window like 512 past steps and produces something like 96 future steps, for each signal.

## Why you'd use it / its trade-offs

**Strengths:** It is simple, small, and fast. It is made of only basic mixers, but it often forecasts surprisingly well. Sometimes it does better than far more complicated models. The channel mixing step is a real advantage when several signals influence each other.

**Weaknesses:** The time mixer is sized to the input length. So it expects a fixed window. It is not flexible about length. It has no special sense of "shape over time" the way convolution or attention models do. It just learns fixed blending weights. With only one signal, the channel mixing step has very little to do.

## In this project

TSMixer is a **serious contender**, not a throwaway baseline. It reads a window of recent EMF values and produces a window of future ones. It uses RevIN at both ends.

It has a second job too. A switch called `return_head` can be turned off. When off, TSMixer skips the final future reading step. It just hands back its well stirred timeline. In that mode it works as the **patch-encoder inside [EMForecaster](30-emforecaster.md)**, the project's star model. It does the mixing while EMForecaster handles the rest.

## See also

- [Key ideas explained](02-key-ideas-explained.md): RevIN, residual connections, and other shared building blocks.
- [EMForecaster](30-emforecaster.md): the star model that can use TSMixer inside it.
- [DLinear](21-dlinear.md): another light and almost linear forecaster to compare against.
