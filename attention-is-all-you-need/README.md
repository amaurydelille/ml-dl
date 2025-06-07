# Attention Is All You Need
### Implementation of the Transformer architecture described in the original paper:
https://arxiv.org/pdf/1706.03762.

I decided to train it to solve English-Vietnamese translation task.

I choosed to implement this transformer using Numpy only without any "fancy"
lib such as PyTorch or Tensorflow because I think it's a good exercise for 
me (and everyone who wants to truely understand Machine Learning) to not be
helped by any abstraction (NumPy already contains a lot of abstraction but is still necessary
for optimized computations). Of course I don't encourage anybody nor think it's 
a good idea to implement Transformers without PyTorch in a professional context.

My hand-written notes can be found as "attention-is-all-you-need-notes.pdf" in 
the repo. Don't hesitate to open issues if you find any error in my calculations.