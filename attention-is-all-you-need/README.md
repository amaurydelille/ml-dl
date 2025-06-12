# Attention Is All You Need
### Implementation of the Transformer architecture described in the original paper:
https://arxiv.org/pdf/1706.03762.

I decided to train it to solve English-Vietnamese translation task.

I choosed to implement this transformer using Numpy only without any helper
lib such as PyTorch or Tensorflow because I think it's a good exercise for 
me (and everyone who wants to truely understand Machine Learning) to not be
helped by any abstraction (NumPy already contains a lot of abstraction but is still necessary
for optimized computations). Of course I don't encourage anybody nor think it's 
a good idea to implement Transformers or any model without PyTorch in a professional context.
I have tried to make my code as clean and commented as possible, but it still might
seems a bit messy.

My hand-written notes can be found as _attention-is-all-you-need-notes.pdf_ in 
the repo. Don't hesitate to open issues if you find any error in my calculations.
