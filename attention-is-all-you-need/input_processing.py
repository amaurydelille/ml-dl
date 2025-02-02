# implementation of input processing for the transformer model
# x_i = embedding(x_i) + positional_encoding(i)

import torch
from torch import nn

def tokenizer(input_text: str):
    """
    Tokenize the input text.
    Very basic tokenizer, because that's not really the focus of this project
    """
    return input_text.split()

def generate_embeddings(tokens: list[str], verbose=False):
    """
    Generate embeddings for the tokens
    """
    num_embeddings = len(tokens)
    embedding_dim = 512

    token_to_index = { token: idx for idx, token in enumerate(tokens) }
    token_indices = [token_to_index[token] for token in tokens]

    if verbose:
        print(f"TOKEN INDICES\n{token_indices}")
        print(f"TOKEN TO INDEX\n{token_to_index}")

    token_indices_tensor = torch.tensor(token_indices)
    embeddings = nn.Embedding(num_embeddings, embedding_dim)

    return embeddings(token_indices_tensor)

def generate_positional_encodings(tensors: torch.Tensor, verbose=False):
    """
    Generate positional encodings for the tokens
    """
    seq_length, d_model = tensors.shape

    # unsqueeze function adds a new dimension at the specified index
    # arrange function creates a range of numbers
    position = torch.arrange(seq_length).unsqueeze(1)
    div_term = torch.exp(torch.arrange(0, d_model, 2) * -(torch.log(10000.0) / d_model))

    tensors[:, 0::2] = torch.sin(position * div_term)
    tensors[:, 1::2] = torch.cos(position * div_term)

    return tensors

def run(text: str):
    tokens = tokenizer(text)
    embeddings = generate_embeddings(tokens)
    positional_encodings = generate_positional_encodings(embeddings)
    return positional_encodings

if __name__ == "__main__":
    input_text = "Hello, how are you?"
    tokens = tokenizer(input_text)
    embeddings = generate_embeddings(tokens, verbose=True)
    print(embeddings)
    generate_positional_encodings(embeddings, verbose=True)