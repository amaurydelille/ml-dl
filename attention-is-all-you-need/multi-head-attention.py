import torch
from torch import nn

# implementation of multi-head attention, as described in the paper
# * queries Q = matrix representing the questions asked by the current token about
# the other tokens in the sequence
# * keys K = matrix representing the answers to the questions asked by the other tokens
# about the current token
# * values V = matrix representing the actual information being passed around

# each of these matrices has dimensions (batch_size, sequence_length, d_model)

def compute_attention(Q, Wq, K, Wk, V, Wv, dk, mask=None):
    """
    Compute the attention for a single head
    """
    Q = Q @ Wq
    K = K @ Wk
    V = V @ Wv

    numerator = Q @ K.T
    denominator = torch.sqrt(torch.tensor(dk))

    return torch.matmul(torch.softmax(numerator/denominator), V)


def compute_attention_head():
    pass