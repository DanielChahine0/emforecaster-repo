# TimesNet (TimesNet)

**In one sentence:** TimesNet looks for the strongest repeating rhythms in the past data (like a daily or weekly cycle). It folds the flat timeline into a grid so each repeat lines up neatly. Then it studies that grid like a picture to guess what comes next.

## The simple idea

Imagine you have a long paper ribbon. It has one number written every hour, all in a single row. It is hard to see the pattern when it is one long strip.

So you do something clever. You cut the ribbon into equal pieces. Each piece is one day long. Then you stack them on top of each other. Now Monday sits above Tuesday, which sits above Wednesday. Reading **down** a column tells you "what happens at 9am each day." Reading **across** a row tells you "what happens over one full day." Suddenly the daily rhythm jumps out at you.

That stacking trick is the heart of TimesNet. It turns a flat line of numbers into a little grid (like a small image). Then it looks at the grid the way you would look at a photo to spot patterns.

## How it actually works

1. **Get the past values ready.** The model takes a window of recent EMF readings. First it smooths the numbers. This is an optional step called RevIN. It subtracts the average and divides by the spread, so everything is centered and on a fair scale. Then a small layer turns each time step into a richer set of features. It also adds a "position stamp" so the model knows the order of the numbers.

2. **Stretch the timeline to cover the future too.** A single straight line layer extends the past window. Now it is long enough to also hold the future stretch it must predict. So if it saw a window of past readings, it now has a longer canvas. The canvas covers the past plus the future part to fill in.

3. **Find the strongest rhythms (the FFT step).** TimesNet runs a frequency analysis. The math tool for this is called the FFT. It measures how strongly each repeating cycle shows up in the data. Then it picks the top few strongest cycles. For example, "repeats every 24 steps" and "repeats every 7 steps." How many it picks is set by a number called `top_k`.

4. **Fold the line into a grid.** For each strong cycle, it cuts the timeline into pieces. Each piece is the length of that cycle. Then it stacks them. One side of the grid is "position within one cycle." The other side is "which cycle number." Sometimes the timeline does not divide evenly. In that case it pads the end with zeros, so the grid is a clean rectangle.

5. **Look at the grid like a picture.** It slides several small square scanners over the grid at the same time. They come in different sizes. This is the Inception block. It uses a handful of filters of sizes 1, 3, 5, and so on. They run side by side and are then averaged together. Small scanners catch fine detail. Bigger ones catch wider patterns. The grid has two directions. So this captures both "what happens inside one cycle" and "how cycles change from one to the next."

6. **Unfold and combine.** Each grid is flattened back into a line. The cycles that the frequency step found strongest get the most say in the final blend. This is a softmax weighting, which means the strongest rhythms count more. The result is added back onto the input. This is a residual connection. It just means "keep the original and add the new findings on top." This whole find, fold, scan, and combine routine is one TimesBlock. The model stacks a few of them in a row.

7. **Make the forecast.** A final straight line layer maps the processed features into the predicted future EMF values. Then RevIN is undone, so the numbers return to their real scale.

## Why you'd use it / its trade-offs

Strengths: TimesNet is genuinely smart about **periodic** data. Say EMF exposure has clear daily or weekly rhythms. Then the fold into a grid trick lets ordinary image style scanners find those patterns easily. It can catch several different cycles at once.

Weaknesses: It is one of the heavier and more complicated models here. It does more work. It runs frequency analysis plus 2D scanning on every block. So it is slower and harder to tune than the simple models. Say the data has no strong repeating cycle. Then the grid trick gives little benefit, and a plain linear model may match it for far less effort.

## In this project

TimesNet is a **serious, well known competitor** model. It is not a throwaway baseline. It reads a window of recent EMF readings and produces the next stretch of EMF values. In this codebase it is set up for forecasting. The code also has a classification mode, but the project uses forecasting. It shares the optional RevIN smoothing trick with other strong models here. It stands as a strong reference point that the project's star model, [EMForecaster](30-emforecaster.md), aims to match or beat.

## See also

- [Key ideas explained](02-key-ideas-explained.md), which covers RevIN, positional encoding, and other shared building blocks
- [ModernTCN](25-modern-tcn.md), another forecaster built on convolution and a useful contrast
- [EMForecaster](30-emforecaster.md), the project's headline model
