import torch
from torch import nn

class SelfAttention(nn.Module):
    def __init__(self, embeddings_size, heads):
        super(SelfAttention).__init__()
        self.embeddings_size = embeddings_size
        self.heads = heads
        self.heads_dimension = embeddings_size // heads

        assert (self.heads_dimension * heads == embeddings_size), "Embeddings size needs to be divisible by heads"

        self.values = nn.Linear(self.heads_dimension, self.heads_dimension, bias=False)
        self.keys = nn.Linear(self.heads_dimension, self.heads_dimension, bias=False)
        self.queries = nn.Linear(self.heads_dimension, self.heads_dimension, bias=False)
        self.fc_out = nn.Linear(heads * self.heads_dimension, embeddings_size)   


    def forward(self, values, keys, query, mask=None):
        n = query.shape[0]
        value_len, key_len, query_len = values.shape[1], keys.shape[1], query.shape[1]

        # split embeddings into heads pieces 
        values = values.reshape(n, value_len, self.heads, self.heads_dimension)
        keys = keys.reshape(n, key_len, self.heads, self.heads_dimension)
        queries = query.reshape(n, query_len, self.heads, self.heads_dimension)

        values = self.values(values)
        keys = self.keys(keys)
        queries = self.queries(queries)

        # einstein summation convention.
        output = torch.einsum("nqhd,nkhd->nhqk", [queries, keys])

        if mask:
            output = output.masked_fill(mask == 0, float("-1e20"))

        attention_scores = torch.softmax(output / (self.embeddings_size ** (1/2)), dim=3)
        output = torch.einsum("nhql, nlhd->nqhd", [attention_scores, values])
        output = output.reshape(n, query_len, self.heads * self.heads_dimension)

        return self.fc_out(output)
    
class TransformerBlock(nn.Module):
    def __init__(self, embeddings_size, heads, dropout, forward_expansion):
        super(TransformerBlock).__init__()
        self.attention = SelfAttention(embeddings_size, heads)
        self.layer_norm_1 = nn.LayerNorm(embeddings_size)
        self.layer_norm_2 = nn.LayerNorm(embeddings_size)

        self.feed_forward = nn.Sequential(
            nn.Linear(embeddings_size, forward_expansion * embeddings_size),
            nn.ReLU(),
            nn.Linear(forward_expansion * embeddings_size, embeddings_size),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, value, key, query, mask):
        attention = self.attention(value, key, query, mask)
        x = self.dropout(self.layer_norm_1(attention + query))
        forward = self.feed_forward(x)
        output = self.dropout(self.layer_norm_2(forward + x))
        return output

class Encoder(nn.Module):
    def __init__(
        self,
        vocab_size,
        embeddings_size,
        n_layers, 
        heads,
        device,
        forward_expansion, 
        dropout,
        max_length       
    ):
        super(Encoder, self).__init__()
        self.embeddings_size = embeddings_size
        self.device = device
        self.word_embedding = nn.Embedding(vocab_size, embeddings_size)
        self.position_embedding = nn.Embedding(max_length, embeddings_size)

        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    embeddings_size,
                    heads,
                    dropout=dropout,
                    forward_expansion=forward_expansion,
                )
            for _ in range(n_layers)]
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        n, seq_length = x.shape
        positions = torch.arange(0, seq_length).expand(n, seq_length).to(self.device)
        output = self.dropout(self.word_embedding(x) + self.position_embedding(positions))

        for layer in self.layers:
            output = layer(output, output, output, mask)

        return output
    
class DecoderBlock(nn.Module):
    def __init__(self, embeddings_size, heads, forward_expansion, dropout, device):
        super(DecoderBlock, self).__init__()
        self.attention = SelfAttention(embeddings_size, heads)
        self.layer_norm = nn.LayerNorm(embeddings_size)
        self.transformer_block = TransformerBlock(
            embeddings_size, heads, dropout, forward_expansion
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, value, key, source_mask, target_mask):
        attention = self.attention(x, x, x, target_mask)
        query = self.dropout(self.layer_norm(attention + x))
        out = self.transformer_block(value, key, query, source_mask)
        return out
    
class Decoder(nn.Module):
    def __init__(
        self,
        target_vocab_size,
        embeddings_size,
        n_layers,
        heads,
        forward_expansion,
        dropout,
        device,
        max_length
    ):
        super(Decoder, self).__init__()
        self.device = device
        self.word_embedding = nn.Embedding(target_vocab_size, embeddings_size)
        self.position_embedding = nn.Embedding(max_length, embeddings_size)

        self.layers = nn.ModuleList(
            [
                DecoderBlock(embeddings_size, heads, forward_expansion, dropout, device)
                for _ in range(n_layers)
            ]
        )
        self.fc_out = nn.Linear(embeddings_size, target_vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, encoder_out, source_mask, target_mask):
        n, seq_length = x.shape
        positions = torch.arange(0, seq_length).expand(n, seq_length).to(self.device)
        x = self.dropout(self.word_embedding(x) + self.position_embedding(positions))

        for layer in self.layers:
            x = layer(x, encoder_out, encoder_out, source_mask, target_mask)

        output = self.fc_out(x)
        return output
    
class Transformer(nn.Module):
    def __init__(
        self,
        source_vocab_size,
        target_vocab_size,
        source_pad_idx,
        target_pad_idx,
        embeddings_size=256,
        n_layers=6,
        forward_expansion=4,
        heads=8,
        dropout=0,
        device=torch.device("cpu"),
        max_length=100,
    ):
        super(Transformer, self).__init__()

        self.encoder = Encoder(
            source_vocab_size,
            embeddings_size,
            n_layers,
            heads,
            device,
            forward_expansion,
            dropout,
            max_length,
        )

        self.decoder = Decoder(
            target_vocab_size,
            embeddings_size,
            n_layers,
            heads,
            forward_expansion,
            dropout,
            device,
            max_length,
        )

        self.source_pad_idx = source_pad_idx
        self.target_pad_idx = target_pad_idx
        self.device = device

    def make_source_mask(self, src):
        src_mask = (src != self.source_pad_idx).unsqueeze(1).unsqueeze(2)
        return src_mask.to(self.device)
    
    def make_target_mask(self, trg):
        N, trg_len = trg.shape
        trg_mask = torch.tril(torch.ones((trg_len, trg_len))).expand(
            N, 1, trg_len, trg_len
        )
        return trg_mask.to(self.device)
    
    def forward(self, source, target):
        source_mask = self.make_source_mask(source)
        target_mask = self.make_target_mask(target)
        enc_src = self.encoder(source, source_mask)
        output = self.decoder(target, enc_src, source_mask, target_mask)
        return output