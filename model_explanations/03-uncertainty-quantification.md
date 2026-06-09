# How Sure Are We? (Conformal Prediction)

**In one sentence:** Instead of giving one single guess for the future, this part of the project draws an honest *band* around the guess. The band has a low value and a high value. It is built to actually contain the truth a chosen fraction of the time, like 90%.

## Why a single number is risky

Imagine a friend who forecasts tomorrow's temperature. If they say "exactly 21 degrees," they will almost always be a little bit wrong. A much more useful answer is "somewhere between 18 and 24 degrees." Now you can plan. You also get a feeling for how confident they are. A tight band ("20 to 22") means they are sure. A wide band ("10 to 32") means they are really not.

All the models in this project forecast one signal over time. That signal is EMF exposure, which is the strength of radio and wireless signals around us. Each model takes a window of recent past values and produces a window of future values. But a single predicted line is a guess, and guesses miss. So this project wraps every forecast in a band. There is a *lower* line and an *upper* line. That band is what tells you how much to trust the forecast.

## Conformal prediction, in plain words

How wide should the band be? Conformal prediction has a clever, honest trick. It does **not** trust the model to judge its own confidence. Instead it *measures* how wrong the model usually is. It uses real examples where it already knows the right answers.

Here is the recipe, and it matches the code in `coverage.py`:

1. **Set aside a batch of past examples** (called the *calibration set*). This is data the model did not train on, but where we know the true future.
2. **Measure the mistakes.** For each example, the code compares the model's forecast to the truth. It records how far off it was (plain absolute distance, the `nonconformity` function). It does this *separately for each step ahead*. That is because predicting 1 step into the future is easier than predicting 24 steps ahead.
3. **Find the cutoff.** For each future step, it lines up all those mistake sizes from smallest to largest. Then it picks the value below which, say, 90% of mistakes fall (`get_critical_scores`). This cutoff is the model's usual worst error for that step. We want the band to cover *every* step at once. So the code also makes a stricter "Bonferroni-corrected" cutoff. It shrinks the allowed miss rate by spreading it across all the steps. This gives slightly wider but safer bands.
4. **Build the band.** For a new forecast, it subtracts the cutoff to get the lower line. It adds the cutoff to get the upper line (`get_intervals`). Steps that are harder to predict get fatter bands automatically.
5. **Check honesty.** The `coverage` function then asks, for fresh test data: did the true value actually land inside the band? It reports two numbers. The first is *independent coverage* (how often each single step was covered). The second is *joint coverage* (how often the *whole* future window was covered all at once, which is stricter).

The beautiful part: this works for *any* model. You do not change the model at all. You just watch how it errs and wrap it in a fair band.

## The trade-off score: narrow vs. covered

A band can cheat in two opposite ways:

- **Too narrow.** It looks confident but keeps missing the truth. Useless.
- **Too wide.** "The value is between 0 and a billion." Always right, tells you nothing. Also useless.

A good model is **narrow *and* still covers the truth often.** To score that balance fairly, `tradeoff.py` computes a single **Trade-off Score (TOS)** for each model:

1. It first combines the two coverage numbers into one **Weighted Average Coverage (WAC)**. It leans more on the stricter joint coverage (weight `beta = 2/3`) than on the per step coverage. The code also halves the result. So WAC is not on the same 0 to 100 scale as the inputs. But that just rescales every model equally and does not change the ranking. This is the "did the band catch the truth?" half.
2. It looks at the **Mean Interval Width (MIW)**, which is how fat the band is on average. To compare models fairly, it converts each model's width into "how it compares to the others" (a *z-score*: above or below the group average). Then it squashes it through an S-shaped curve (a *sigmoid*). So a band that is narrower than average scores closer to 1, and a wider one scores closer to 0.
3. Finally it adds the coverage half and the narrowness half together in equal measure (each counts 50%). A higher TOS means a model gives bands that are both tight and trustworthy. The results are saved to a CSV so models can be ranked.

So coverage alone never wins, and tininess alone never wins. The model that earns the best score is the one that says "between X and Y" as tightly as possible *while still being right as often as promised.*

## See also

- [The big picture](01-the-big-picture.md). What forecasting means and what EMF is.
- [Key ideas explained](02-key-ideas-explained.md). The shared building blocks the models use.
- [EMForecaster](30-emforecaster.md). The project's star model, the one these bands are wrapped around.
