import torch
from torch import nn
import math
import torch.nn.functional as F
class PositionalEncoding(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, X: torch.Tensor, d_model: int) -> torch.Tensor:
        seq_len = X.size(1)
        pe = torch.zeros(seq_len, d_model, device=X.device)
        position = torch.arange(0, seq_len, dtype=torch.float, device=X.device).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float().to(X.device) * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        return X + pe.unsqueeze(0)

class Patch:
    def __init__(self, X: torch.Tensor, patch_len: int) -> None:
        self.X = X
        self.patch_len = patch_len

    def forward(self) -> None:
        n_patches = self.X.shape[1] // self.patch_len

        X = self.X.reshape(self.X.shape[0], n_patches, self.patch_len)
        return X

class ResidualBlock(nn.Module):
    def __init__(self, input_dim: int, output_dim: int) -> None: # output_dim should be model_dim
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_layer = nn.Linear(in_features=input_dim, out_features=512)
        self.fc = nn.Linear(in_features=512, out_features=output_dim)

        if input_dim != output_dim:
            self.proj = nn.Linear(input_dim, output_dim)
        else:
            self.proj = nn.Identity()

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return self.fc(self.hidden_layer(X)) + self.proj(X)

class SelfAttention(nn.Module):
    def __init__(self, d_model: int, d_k: int) -> None:
        super().__init__()
        self.d_model = d_model
        self.d_k = d_k

        self.Wq = nn.Linear(d_model, d_k)
        self.Wk = nn.Linear(d_model, d_k)
        self.Wv = nn.Linear(d_model, d_k)

    def forward(self, X: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        Q = self.Wq(X)
        K = self.Wk(X)
        V = self.Wv(X)

        attention_scores = (Q @ K.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            attention_scores = attention_scores.masked_fill(mask == 0, float('-inf'))
        attention_weights = F.softmax(attention_scores, dim=-1)
        return attention_weights @ V

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, n_heads: int, d_model: int, d_k: int) -> None:
        super().__init__()
        self.d_model = d_model
        self.d_k = d_k
        self.n_heads = n_heads
        self.self_attention = [SelfAttention(d_model=self.d_model, d_k=self.d_k) for _ in range(n_heads)]
        
    def forward(self, X: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        attention_outputs = [self.self_attention[i].forward(X, mask) for i in range(self.n_heads)]
        return torch.concatenate(attention_outputs, dim=-1)

class FeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int) -> None:
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.fc = nn.Linear(in_features=self.d_model, out_features=self.d_ff)
        self.fc2 = nn.Linear(in_features=self.d_ff, out_features=self.d_model)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.relu(self.fc(X)))


class TimesFM(nn.Module):
    def __init__(self, d_model: int, patch_len: int) -> None:
        super().__init__()
        self.d_model = d_model
        self.d_k = d_model // 4
        self.patch_len = patch_len
        self.patch_projection = nn.Linear(in_features=patch_len, out_features=d_model)
        self.self_attention = MultiHeadSelfAttention(n_heads=4, d_model=self.d_model, d_k=self.d_k)
        self.feed_forward = FeedForward(d_model=self.d_model, d_ff=self.d_model * 4)
    
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        x_patched = Patch(X, self.patch_len).forward()
        x_projected = self.patch_projection(x_patched)
        x_patched_encoded = PositionalEncoding().forward(x_projected, self.d_model)
        x_residual_1 = ResidualBlock(input_dim=self.d_model, output_dim=self.d_model).forward(x_patched_encoded)
        x_attention = self.self_attention(x_residual_1, mask=None)
        x_ffn = self.feed_forward(x_attention)
        x_residual_2 = ResidualBlock(input_dim=self.d_model, output_dim=self.d_model).forward(x_ffn)
        return x_residual_2