# EMForecaster (EMForecaster)

**In one sentence:** EMForecaster takes a stretch of recent past values. It chops that stretch into small overlapping snapshots. It summarizes each one. It remembers their order. Then it looks at all the summaries together to predict the future. Best of all, you can swap its inner parts in and out.

This is the star model of the project. It is named after the project itself, so it gets a little extra room here. It forecasts one signal, which is just the EMF exposure value over time. The code's own note says it is "univariate only." That means it watches and predicts a single number, not many at once.

## The simple idea (the analogy)

Imagine you want to guess how busy a coffee shop will be later today. You do not stare at every single second of the morning. Instead you take a few overlapping photos. One photo shows the 8am rush. One shows the mid morning lull. One shows the noon crowd. The photos overlap a bit, so nothing important falls between them. You jot a short summary under each photo. You keep them in order, with morning first and noon last. Then you look at the whole row of summaries together and make your guess.

That is exactly what EMForecaster does. The "photos" are called patches. They are small overlapping chunks of the recent history. The summaries are made by a small helper model. There is one more nice trick too. Before it looks, it ignores how high or low the numbers are overall. It just studies the *shape* of the ups and downs. So if the whole signal drifts up or down over time, the model still works fine. It puts the real level back at the end.

## How it actually works (the real steps in the forward pass)

1. **Level the input (RevIN).** First it measures two things about the recent window. It finds the average and the spread (how bumpy the window is). It subtracts the average and divides by the spread. Now everything sits centered around zero, no matter how big the original numbers were. It quietly remembers the average and spread so it can restore them later. (RevIN means "reversible instance normalization." That is a fancy name for "flatten it now, un-flatten it at the end.")

2. **Cut into overlapping patches (Patcher).** It slides a fixed size window across the history. Each step moves forward by a smaller amount, so the windows overlap. For example: take 24 numbers, slide forward 12, take the next 24, and so on. The very end is padded by repeating the last value, so nothing is lost. The result is a stack of small chunks. Each one is `patch_dim` long.

3. **Summarize each patch (the patch encoder).** A small "patch model" turns each chunk into a richer summary. The summary is a longer list of features, `patch_embed_dim` long. This inner model is *configurable*. It is either a tiny [DLinear](21-dlinear.md) or a [TSMixer](22-tsmixer.md). There are two ways it can run:
   - **Independent patching:** each patch is summarized on its own. The *same* little model is applied over and over. It is like writing each photo's caption separately.
   - **Dependent patching:** all the patches are fed in together. So the summarizer can let them influence each other. It is like captioning the whole row of photos at once.

4. **Tidy up and tag the order.** It can pass the summaries through an activation step. That is a simple "bend the numbers" step, like GeLU. It can also pass them through a normalization step, which re-centers them. Then it adds a learnable positional tag. That is a small set of numbers the model trains for itself. The tag means it never forgets which summary came first and which came last. (Order matters in time series. This stops the model from treating the past as a shuffled bag.)

5. **Reason over everything and forecast (the backbone head).** All the patch summaries are laid out in one long row. The "backbone" looks at that whole row and produces the final forecast (the future window, `pred_len` long). The backbone is configurable too. It is either a [DLinear](21-dlinear.md) or a plain Linear layer.

6. **Restore the real level (RevIN denormalize).** Finally it puts back the average and spread it set aside in step 1. So the forecast comes out in real EMF units instead of the flattened, centered version.

So from start to finish: it might take 336 past numbers and produce 96 future numbers.

## Why you'd use it / its trade-offs

**Strengths.** It is *modular*. That is the whole selling point. You mix and match. You can pick DLinear or TSMixer for the patch summaries. You can pick DLinear or Linear for the final reasoning. You can pick independent or dependent patching. You can use the order tag or leave it out. So one design covers many models. Patching also lets it look at "chunks" of shape rather than single points. This often captures patterns more cheaply than reading every point one at a time. And the level-removing trick (RevIN) makes it steady even when signals wander up and down.

**Weaknesses.** It is univariate. It forecasts one signal only. So it cannot blend several related signals together on its own. It has more moving parts than a plain Linear model. So it takes a bit more tuning and runs a bit slower. And many of its inner pieces are still made of simple linear layers. So it is powerful but not magic. Its strength comes from the clever way it patches and restores, more than from one heavy component.

## In this project

EMForecaster is the headline model. It is the one the paper and the whole repo are built around. The simpler baselines ([MLP](10-mlp.md), [CNN](11-cnn.md), [LSTM](12-lstm.md)) and the standalone models ([Linear](20-linear.md), [DLinear](21-dlinear.md), [TSMixer](22-tsmixer.md), [PatchTST](23-patchtst.md)) are the field it is trying to beat. It reads a window of recent EMF readings and produces a window of future readings. It also pairs naturally with the conformal "how sure are we?" step. So each forecast can come with an honest range. See [Uncertainty Quantification](03-uncertainty-quantification.md).

## See also

- [Key ideas explained](02-key-ideas-explained.md). RevIN, patching, and decomposition in one place.
- [DLinear](21-dlinear.md) and [TSMixer](22-tsmixer.md). The two choices for its inner pieces.
- [Uncertainty Quantification](03-uncertainty-quantification.md). Turning a single forecast into a trustworthy range.
