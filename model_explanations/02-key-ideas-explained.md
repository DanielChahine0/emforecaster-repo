# Key Ideas Explained (the shared building blocks)

Many of the models in this project are built from the same few small parts. We do not want to explain each part over and over. So this page explains them once. The other model pages link back here when they use one of these ideas.

Quick reminder: every model here does *forecasting*. It looks at a window of recent past values. For example, it might look at the last 96 readings of EMF exposure. Then it tries to guess the next stretch of future values. Keep that picture in mind as you read.

---

## 1. RevIN, leveling the playing field

**The analogy:** Imagine you want to learn the *shape* of a hill. But every hill you see is at a different height. Some are stretched taller than others. This makes them hard to compare. So before you study a hill, you slide it down to sea level. Then you squash it to a standard height. Now all the hills are easy to compare. After you finish, you put the hill back to its real height and size.

**What the code does** (`emforecaster/layers/patchtst/revin.py`):

- In **"norm"** mode, it looks at the input window and finds two things. First, the average value (the overall level). Second, the spread (how bumpy or flat it is). It then subtracts the average, so everything is centered around zero. It also divides by the spread, so the size is standard. Now the model only has to learn the *shape*, not the height.
- It remembers that average and spread.
- In **"denorm"** mode, the model has already made its prediction. Now RevIN does the reverse. It multiplies the spread back in and adds the average back on. This puts the forecast back into real EMF units.

**The optional "affine" part:** there is a learnable scale and a learnable shift. These are two small knobs the model can tune while it trains. When they are turned on, the values are first centered and squashed. Then the model can stretch and nudge them a little in its own learned way. At the end it undoes that nudge. This part is on by default here.

Why it matters: real signals drift. The general level of EMF exposure today might be different from last week. RevIN removes that drift, so the model is not confused by it.

---

## 2. Patching, reading in chunks and not letter by letter

**The analogy:** Reading a book one single letter at a time is slow. You also lose the meaning. It is much easier to read whole short words at a glance. Patching does the same thing to the timeline. Instead of feeding the model one number at a time, it chops the past window into short chunks (called *patches*). It treats each chunk as one item.

**What the code does** (`emforecaster/layers/patcher.py`, the `Patcher` class):

- **Patch size** is how many time steps go in each chunk. The default is 16 numbers per chunk.
- **Stride** is how far the window slides before it grabs the next chunk. The default is 8. The stride (8) is smaller than the patch size (16). So the chunks **overlap**. The second half of one chunk is the first half of the next. Overlap helps the model. It means the model does not miss anything that falls on a chunk boundary.
- It pads the very end of the signal by repeating the last value a few times. This way the final chunk is full and nothing is cut off.
- The result is a stack of chunks. A long timeline of single numbers becomes a shorter list of short chunks.

There is also a `VerticalPatcher` in the same file. It is used when several signals (channels) need to be glued together inside each chunk. The basic idea is the same: group nearby numbers into chunks.

Why it matters: chunks are shorter to process. Each chunk also carries a little local pattern. That is easier to reason about than lone numbers.

---

## 3. Series decomposition, the slow part plus the bumpy part

**The analogy:** Think of a noisy line on a graph as a calm river current with little ripples on top. If you blur your eyes, you see only the slow current (the *trend*). The ripples are what is left over after you remove the current.

**What the code does** (`emforecaster/layers/dlinear/series_decomp.py`):

- It runs a **moving average** over the signal. A moving average slides a small window along the line. It replaces each point with the average of its neighbors. This smooths out the ripples and leaves the slow, drifting trend. (The window pads the two ends by repeating the first and last values. This way the smooth line is the same length as the original.)
- It then computes the leftover by simple subtraction. **The remainder is the original signal minus the smoothed trend.** This leftover holds the ripples and the repeating ups and downs (often called the seasonal part).

So one wiggly line becomes two cleaner lines: a smooth trend and a bumpy remainder. Some models forecast each part on its own. Then they add the two forecasts back together. This is often easier than forecasting the messy combined line.

---

## 4. Positional encoding, tagging each chunk with "where am I?"

**The analogy:** Imagine you cut a comic strip into separate panels and shuffle them on a table. You lose the story order. So you write a small number on the back of each panel. The number says which one comes first, second, third. Now you can look at them all at once and the order is not lost.

**What the code does** (`emforecaster/layers/patchtst/pos_enc.py`):

- First it takes each chunk (a short patch of raw numbers). It passes the chunk through a small layer. This turns it into a longer list of features the model can work with. (This step just re-describes the chunk in a richer form.)
- Then it **adds a position marker** to each chunk. There is one learnable marker per slot in time (first chunk, second chunk, and so on). These markers are learned during training. They start from tiny random values. Adding them tells the model where in time each chunk sits.

Why it matters: the attention step (next section) looks at all chunks together. On its own, it has no sense of order. The position markers put the time order back in.

---

## 5. Attention, letting each chunk look around and decide what matters

**The analogy:** When you answer a question about a paragraph, you re-read it. You pay extra attention to the few sentences that actually matter. You skim the rest. Attention lets each chunk do the same. It looks at every other chunk and decides which ones are most relevant to it.

**What the code does** (`emforecaster/layers/patchtst/enc_block.py`, the `_MultiheadAttention` and `EncoderBlock` classes):

- The `_MultiheadAttention` class is a thin wrapper around PyTorch's built-in attention. It is used in **self-attention** mode. Every chunk acts as the question, the index, and the content all at once. (In the code, the same input is passed three times as query, key, and value.) In plain words, each chunk compares itself to all the chunks. Then it builds a new version of itself by mixing in the ones it found most relevant.
- **"Multi-head"** means it does several of these look-around passes at the same time. Each pass is free to focus on a different kind of relationship. Then it combines them.
- An `EncoderBlock` wraps this up into a repeatable unit:
  1. Run attention so chunks share information.
  2. Add the result back to the original (a "shortcut" so nothing useful is lost). Then normalize it (rescale the numbers so they stay in a steady range and training is stable).
  3. Pass it through a small two layer feed forward network (with a GELU activation, a smooth on/off gate). This lets each chunk refine its own features.
  4. Add and normalize again.

Stacking several of these blocks lets the model build up a richer and richer understanding of how the chunks relate. It does this before it makes a forecast.

---

## See also

- [The big picture](01-the-big-picture.md): what the project does and what forecasting means.
- [PatchTST](23-patchtst.md): the model that uses patching, positional encoding, and attention all together.
- [DLinear](21-dlinear.md): the model built directly on series decomposition.
- [EMForecaster](30-emforecaster.md): the project's main model.
