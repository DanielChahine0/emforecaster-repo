# DLinear (DLinear)

**In one sentence:** DLinear splits the recent past into a smooth "big-picture" line and the small bumps left over. It predicts the future of each part with one tiny weighted sum layer. Then it adds the two predictions back together.

## The simple idea

Imagine you are looking at a graph of how much radio signal a phone tower gives off over time. (This project forecasts EMF exposure, which means how strong the invisible radio waves are.) The line wiggles a lot. That makes it hard to read.

So you trace it twice. First you draw a smooth, slow line through the middle. This line ignores most of the jitter. That smooth line is the **trend**. The trend is the slow overall drift, up or down. Next you note how far the real line jumps above or below that smooth line at each moment. Those gaps are the **leftover bumps**. They are also called the "seasonal" part or the remainder part.

Two simple pictures are much easier to continue into the future than one messy picture. So DLinear continues each one on its own. Then it stacks them back on top of each other.

## How it actually works

1. **Take in the past.** The model receives a window of recent values for each signal it watches. The code's built in default window is 512 numbers long. But this project sets it to 336 past time steps instead.
2. **Find the smooth line.** It slides a small averaging window across the past. This is called a "moving average". At each point it just averages the nearby values. In this project that window is only 3 steps wide, so the smoothing is very light. It just barely takes the rough edge off the line. (The code's generic default is a wider window of 25, which would smooth more strongly.) The edges are padded by repeating the first and last value, so nothing gets cut off. The result is the **trend**.
3. **Find the bumps.** It subtracts that smooth line from the real values. Whatever is left is the up and down jitter. That leftover is the **seasonal** part, or the remainder.
4. **Predict each part.** Each part goes through its own tiny layer. That layer turns the 336 past numbers into the future numbers we want. (This project asks for 96 future steps.) This layer is just a weighted sum. There are no curves and no fancy tricks. There are two such layers: one for the trend and one for the bumps.
5. **Add them back.** The two predicted pieces are added together to make the final forecast.

A nice detail: the layers start out as a plain average. Every weight begins at 1 divided by the input length. So before any training, the model simply guesses the average of the past. Training then nudges these weights to do better.

## Why you'd use it / its trade-offs

**Strengths:** It is tiny, fast, and easy to understand. Separating the slow drift from the fast wiggles is the whole trick. Each piece becomes an easy, almost straight thing to extend. DLinear is nearly all simple weighted sums, with no deep layers. Even so, it is famously hard to beat on many forecasting tasks. That is its surprise: simple done right often wins.

**Weaknesses:** Because it is mostly linear, it cannot capture twisty, complicated relationships. It learns a fixed pattern for each future step. So very irregular or sudden changes can fool it.

**Shared vs. individual:** There is a switch called `individual`. When it is off (the default here), all signals share the same two layers. When it is on, each signal gets its own pair of layers. That is more flexible, but there is more to learn. In this project there is mostly one signal (EMF), so the difference is small.

## In this project

DLinear is one of the **serious, strong contenders**. It is not just a throwaway baseline. It reads a window of 336 past EMF values and produces a window of 96 future EMF values, shaped as (batch, channels, future length). It is exactly the kind of "simple but tough to beat" model the star model is measured against. In fact, the very same split into trend plus bumps is reused **inside** the project's headline model, [EMForecaster](30-emforecaster.md).

## See also

- [Key ideas explained](02-key-ideas-explained.md) covers the decomposition (trend plus remainder) idea in more depth.
- [Linear](20-linear.md) is the even simpler cousin. It is a single weighted sum with no splitting.
- [EMForecaster](30-emforecaster.md) is the star model that borrows this same decomposition.
