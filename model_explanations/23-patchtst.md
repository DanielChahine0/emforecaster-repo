# PatchTST (PatchTST)

**In one sentence:** PatchTST chops the recent past into short chunks that overlap. Then it lets the chunks "compare notes" with each other to figure out what comes next.

## The simple idea

Imagine you are reading sheet music to guess the next few notes. You do not stare at one single note at a time. That would be slow and confusing. Instead your eyes take in a few notes at once. That is a little phrase, a shape you can recognize. Then your brain compares the phrases. "This phrase sounds like that earlier one, so the next bit is probably this."

PatchTST does the same thing with a line on a graph. The "signal" here is EMF exposure. That is the amount of radio energy in the air, measured over time. PatchTST does not look at every single past number on its own. Instead it groups them into short chunks. These chunks are called "patches." Then it lets every chunk look at every other chunk to find shapes that repeat. From all that, it draws the future part of the line.

## How it actually works

Here is the real path the data takes inside the code. We will go step by step.

1. **Level the data first (RevIN).** Before anything else, the model subtracts the average. Then it divides by the spread. Now the numbers are centered and a similar size. This is called normalization. It is explained in [02-key-ideas-explained.md](02-key-ideas-explained.md#revin). At the very end the model puts the original scale back.

2. **Cut the timeline into chunks (patching).** The `Patcher` slides a window across the past values. It grabs a chunk each time it moves. By default each chunk is 16 numbers wide. The window slides over by a small step, so the chunks overlap. If the window runs off the end, the code pads it with the last value repeated. The result is a handful of short chunks instead of one long line. (See "patching" in [02-key-ideas-explained.md](02-key-ideas-explained.md#patching).)

3. **Turn each chunk into a richer description, and tag its position.** A simple linear layer takes each chunk. (This layer is just a weighted mix.) It blows up each 16-number chunk into a longer list of features (size `d_model`). Then the model adds a "position tag." This way it remembers which chunk came first, second, third, and so on. The tag is a learned set of numbers, one per chunk slot. (See "positional encoding" in [02-key-ideas-explained.md](02-key-ideas-explained.md#positional-encoding).)

4. **Let the chunks compare notes (the Transformer).** This is the heart of the model. It stacks several encoder blocks. In each block, "attention" lets every chunk look at every other chunk. Each chunk then decides which other chunks matter for it. After attention, a small network with two layers reshapes each chunk's features. Both steps add the input back in (a shortcut) and then normalize. (See "attention" in [02-key-ideas-explained.md](02-key-ideas-explained.md#attention).)

5. **Each channel is handled on its own.** EMF data may have more than one channel. That means more than one signal. PatchTST glues the batch and channel together. So attention works on one channel at a time. It never mixes channels inside the attention step. This is the "channel independent" design.

6. **Draw the future (the head).** Finally, the chunks for each channel are flattened into one long list. A linear layer maps that list to the future values. So it takes a window of past numbers and produces a fixed window of future numbers. There is also an optional "mlp" head. That is a slightly deeper version with an extra layer and a wiggle function called GELU. But the default is the plain linear head.

7. **Put the scale back (RevIN denorm).** Step 1 is undone. Now the forecast lands in real EMF units.

## Why you'd use it / its trade-offs

Strengths: working with chunks instead of single points means there are far fewer items for attention to juggle. So it is much faster and uses less memory. Chunks also capture little shapes (a rise, a dip) better than lone points do. It is a strong forecaster that many people respect.

Weaknesses: it is heavier and slower to train than the plain linear models. Each channel is processed alone. So it cannot directly learn how one channel affects another inside attention. And like all these models, it always predicts the same future window. That window has a fixed length.

## In this project

PatchTST is a serious contender here, not a throwaway baseline. It is far more capable than the toy [transformer baseline](13-transformer.md), which is only meant to be beaten. PatchTST reads a window of recent EMF values and produces the predicted next window. In this project it is one of the real benchmarks. The star model, [EMForecaster](30-emforecaster.md), is measured against it.

## See also

- [02-key-ideas-explained.md](02-key-ideas-explained.md): patching, attention, RevIN, and positional encoding in plain words.
- [13-transformer.md](13-transformer.md): the much simpler transformer baseline, for contrast.
- [30-emforecaster.md](30-emforecaster.md): the paper's main model that PatchTST is compared with.
