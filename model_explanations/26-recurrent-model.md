# The Flexible Reader (RecurrentModel)

**In one sentence:** This is one piece of code that can become a few different "reader" networks (RNN, GRU, or LSTM). Each one walks along your past data from left to right. It keeps a running memory as it goes. Then it guesses the future from what it remembers.

## The simple idea

Imagine reading a story one word at a time. As you read, you keep a little summary in your head. Each new word updates that summary. By the end, your head holds the "gist" of the whole story. You use it to guess how the story continues.

A recurrent network does the same thing with numbers. It reads the recent past of the EMF signal one step at a time. (The EMF signal is how strong the radio waves were over time.) It updates an internal memory at each step. At the end, it uses that memory to predict the next stretch of values.

The neat trick here is that this single file can wear different hats. You tell it which kind of reader to be with a setting called `backbone_id`.

## How it actually works

1. **Pick a reader (the `backbone_id` setting).**
   - **RNN** = the plainest reader. It keeps one running memory and overwrites it each step. It is simple, but it tends to forget things that happened long ago.
   - **LSTM** = a smarter reader with little "gates" (tiny valves). The gates decide what to keep, what to throw away, and what to pass on. So it remembers important things for longer.
   - **GRU** = a lighter, simpler version of the LSTM with fewer gates. It does almost the same job with less machinery.
   (There is also a "Mamba" option in the code. But that is a different kind of model, not a classic reader.)

2. **(Optional) Tidy the numbers first (RevIN).** If turned on, it shifts and scales the input so it sits around zero with a steady size. This makes learning easier. It can undo this on the output so the forecast comes back in real units. See [02-key-ideas-explained.md](02-key-ideas-explained.md).

3. **(Optional) Chop into patches.** If `patching` is on, the past window is cut into overlapping chunks. (Think of slicing a long ribbon into short strips that overlap.) Each strip becomes one "step" the reader reads, instead of single numbers. See [02-key-ideas-explained.md](02-key-ideas-explained.md).

4. **Read it, maybe both ways.** The reader walks through the steps building memory. If **bidirectional** is on, it also reads the sequence backwards. Whether the forward and backward summaries actually get combined depends on the next setting. The "overall gist" path does merge information from both directions. But the default "takeaway at the end" path only keeps one direction's final memory. So bidirectional helps, but it does not always glue the two summaries together.

5. **Decide what to hand off.** Two choices in the code:
   - **last_state**: trust the final summary (the memory after the last step). This asks, "what's my takeaway at the end?"
   - **avg_state**: average the memory across every step. This asks, "what's the overall gist?"

6. **Make the forecast.** That summary goes through a simple output layer (a "head"). The head turns it into the predicted future values. It takes the past window and produces a fixed number of future numbers.

## Why you'd use it / its trade-offs

Strengths: reading one step at a time is a natural fit for time. LSTM and GRU memory gates help it hold onto patterns. It is flexible. One knob switches between three classic readers.

Weaknesses: it reads one step at a time, so it is slower to train. It can still struggle to remember very old, very long patterns. Newer designs in this project often beat it.

## In this project

This is the grown-up, flexible cousin of the tiny [LSTM baseline](12-lstm.md). It reads a window of past EMF values and outputs the predicted future window. With patching, RevIN, and the bidirectional and last-versus-average options, it can be a serious contender. But it is mainly here as a strong, classic point of comparison rather than the headline model.

## See also
- [12-lstm.md](12-lstm.md). The small, fixed LSTM baseline this builds on.
- [02-key-ideas-explained.md](02-key-ideas-explained.md). RevIN and patching, explained simply.
- [01-the-big-picture.md](01-the-big-picture.md). What forecasting means here.
