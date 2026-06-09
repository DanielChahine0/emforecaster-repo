# ModernTCN (ModernTCN)

**In one sentence:** ModernTCN forecasts the future by sliding lots of "pattern stamps" of different widths over your recent data. Some of these stamps are very wide, so they can catch slow patterns that take a long time to play out.

## The simple idea

Imagine you have a long strip of paper. On it is the past values of EMF exposure (the radio-wave signal we want to forecast), drawn as a wiggly line. Now imagine a set of rubber stamps. Each stamp knows one little shape. It might be a bump, a dip, or a slow rise. You slide each stamp along the strip. You mark every spot where its shape shows up.

Some stamps are narrow. They only notice quick wiggles. Other stamps are very wide. They can notice slow patterns that take a long time to play out. ModernTCN's big trick is using these very wide stamps. A wide stamp can "see" far back all at once. So the model understands long stretches of the signal, not just tiny pieces.

After stamping, the model has another step. This step lets the different stamp results "talk to each other" and combine. Then it stacks this whole stamping and combining process several times. Each time it builds up a richer understanding. Finally it uses what it learned to draw the next chunk of the line. That chunk is the forecast.

## How it actually works

Here is what the code in this project *really* does to a window of past values, step by step. One thing to know first. This copy of ModernTCN was set up for *classification*. The project reuses that exact path for forecasting. So a few parts described in the paper are present in the file but are never actually run here. We point those out as we go.

1. **Cut the line into patches.** A first sliding layer (called the "stem") chops the past window into chunks ("patches"). It turns each chunk into a small group of numbers that describe it. So instead of one long line, you now have a row of patch descriptions. (There is no normalization step before this. A RevIN normalizer object is created in the code, but nothing ever calls it. So it just sits there unused. No centering or rescaling actually happens.)

2. **Run the stamping blocks.** Each block does three things in order:
   - A **wide per-channel stamp** ("depthwise convolution" with a large kernel) slides across the patches. "Depthwise" means each feature gets its own stamp, applied on its own. It does not mix features yet. The stamp is wide (set by `large_size`), which is how the model sees long patterns. There is also a small companion stamp (`small_size`) added in to catch fine detail.
   - A **mixing step** ("pointwise convolution", a stamp that is only 1 patch wide) blends the features together. It *can* grow them to a bigger size and shrink them back. How much it grows is set by `ffn_ratio`. With the default setting of 1 it does not grow at all, because the bigger size equals the original size.
   - It does a second mixing step the same way, but along the other arrangement of the data. So information flows both across features and across the signal's own channels.
   - Each block also **adds its input back** to its output (a "residual connection"). This is like keeping your original notes next to your edits, so nothing useful gets lost.

3. **Stack the blocks into stages.** `num_enc_layers` says how many blocks to stack. `d_model` says how wide the internal number groups are (bigger means more capacity). You can make these into lists to build several stages.

4. **Produce the output through the classification head.** Everything the blocks learned is flattened into one long row of numbers. That row is passed through a single plain layer (a "classification head" called `head_class`). This turns it into one number per output slot. The project sets the number of output slots equal to the forecast length. So this row of numbers *is* the forecast. There is no separate forecasting head. There is no step that adds an average or spread back. The numbers come out raw, with no undo-normalization. In plain words, it takes a window of past values and produces a window of future values.

## Why you'd use it / its trade-offs

**Strengths.** It uses convolutions (sliding stamps), not "attention". Attention is the heavy mix-everything-with-everything trick that transformers use. Convolutions are cheap and fast. The clever part is the *wide* stamps. Usually convolutions only see tiny windows. But ModernTCN borrows ideas that made transformers strong. It gets a long reach while staying efficient.

**Weaknesses.** It has many knobs (`large_size`, `small_size`, `ffn_ratio`, `num_enc_layers`, `d_model`). So tuning takes effort. It is more complicated than the simple baselines.

**A real quirk in this code, stated honestly.** The version here was wired up for *classification*, not forecasting. The forward pass runs the classification path. The project sets the number of "classes" equal to the forecast length. So in practice it produces one number per future step through the classification head, rather than the paper's dedicated forecasting head. Two paper features are present in the file but never used on this path. The RevIN normalizer is created but never called. A dedicated forecasting head (`Flatten_Head`) is built but never run. It works, but it is not the textbook forecasting setup. Keep that in mind when reading results.

## In this project

ModernTCN is one of the stronger, more modern competitor models, not a simple baseline. It reads the recent window of EMF exposure (and any extra channels). It outputs the predicted future window. It is here to be compared against the project's star model, [EMForecaster](30-emforecaster.md).

## See also

- [Key ideas explained](02-key-ideas-explained.md), which covers RevIN, patching, and convolutions in plain words.
- [CNN baseline](11-cnn.md), the simpler convolution model to compare against.
- [PatchTST](23-patchtst.md), which also patches the signal, but uses attention instead of stamps.
