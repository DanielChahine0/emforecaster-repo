# Transformer Baseline (RF_EMF_Transformer)

**In one sentence:** This is a simple "everyone looks at everyone" model. It lets every past moment compare itself to all the other moments. Then it turns that into a guess about the future.

## The simple idea

Imagine a group of friends trying to guess tomorrow's weather. Each friend remembers one day from the recent past. Before anyone guesses, every friend turns to all the others and asks: "What did you see? How much should I care about your day compared to my own?" Each friend then mixes in a little of what the most useful friends said. After a couple of rounds of this group chat, they combine everything into a single prediction.

That "turn to everyone and weigh what matters" step is called **attention**. (Each item looks at all the other items and decides how important each one is.) It is the heart of this model.

In this project the model is doing forecasting. It takes a window of recent past values of one signal (EMF exposure over time). Then it predicts a window of future values.

## How it actually works

Here is what the code really does, step by step:

1. **Spread out each moment.** The input is a row of recent numbers (for example 336 past readings). Each single number is passed through a small layer (`fc1`). This blows it up from 1 value into a few values (called `d_model`, the model's working width). Think of it as giving each day a richer "description" instead of a bare number, so the friends have more to talk about. In this project that width is only **2** (set in the config), so each moment gets a 2 number description.

2. **Run a few rounds of group chat.** The numbers go through a couple of **EncoderBlocks** (here, 2). Inside each block:
   - **Attention:** every moment looks at every other moment and mixes in what matters most.
   - **Add and normalize:** the original input is added back in, so nothing important gets lost. Then the values are rescaled to a calm, steady range (`LayerNorm`, which keeps the numbers from getting too big or too small).
   - **A tiny think step:** a small two layer mini network (`feedforward`) reshapes each moment on its own, with a smooth squashing function (`Tanh`) in the middle.
   - **Add and normalize again.** (Here is a small quirk worth noting. When it adds back in the second time, the code adds the attention result rather than the value that was just normalized. This is a slightly unusual wiring, but it is minor.)

3. **Then, oddly, two convolution steps.** After the group chat, the data is shaped like (batch, moments, feature width), where the feature width is the 2 number description from step 1. The two `Conv1d` layers (with `Tanh` between them) do **not** slide over time, and they do **not** blend neighboring moments. Instead they treat each moment position as a separate input channel. They slide over the *feature width* (the `d_model` numbers describing each moment). So they mix the features inside each description, not the moments next to each other. Here is the load bearing detail. The feature width is just **2**, and the first convolution uses a window of size 2. So that window covers the *entire* feature width at once and crushes those 2 numbers down to a single number. The second convolution (window of size 1) leaves it at a single number. This is the unusual part. Real transformers do not normally bolt extra convolutions onto the end. This baseline mixes the two ideas.

4. **Make the forecast.** Because the two convolutions have already squeezed the feature width down to 1, the data is now just one number per moment. The code drops that leftover size 1 dimension. This leaves a plain row of past length values (batch, moments). Only *now* can the final layer (`fc2`) map that past length row straight to the future length you asked for. So `fc2` does not act on the convolution output directly. The convolutions must first shrink the feature width to 1. Then `fc2` turns "the past window" into "the future window" in one shot.

A warning for anyone reading the code: this whole chain only lines up because the feature width is exactly 2. The model's built in default is a width of 32. With that default the network does not even run. The first convolution's size 2 window would leave 31 numbers, but `fc2` expects the past length. It only works at the width of 2 set in this project's config.

## Why you'd use it / its trade-offs

Strengths: attention lets the model link far apart moments directly, not just neighbors, which can be powerful. Weaknesses: this version is small and a bit patched together. The extra convolution layers at the end are non standard (and, as noted, only work at one specific width). The whole thing is meant as a plain yardstick, not a polished design.

## In this project

This is a **simple baseline** (`model_id: RF_EMF_Transformer`). It is something the fancier models are supposed to beat. It reads a window of past EMF values and produces a window of predicted future values. It is built from a basic homemade EncoderBlock plus those two extra convolutions. So treat it as a reference point, not a serious contender.

The serious, properly built transformer in this project is PatchTST. If you want to see attention done the careful way, look there.

## See also

- [Key ideas explained](02-key-ideas-explained.md). What attention really means.
- [PatchTST](23-patchtst.md). The serious transformer here.
- [The big picture](01-the-big-picture.md). What forecasting means in this project.
