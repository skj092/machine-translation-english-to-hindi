import torch
import math
import torch.nn as nn


class WordEmbedding(nn.Module):
    def __init__(self, n_vocab: int, d_model: int) -> None:
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(n_vocab, d_model)

    def forward(self, xb: torch.tensor):
        # (bs, seq_len) -> (bs, seq_len, d_model)
        return self.embedding(xb) * math.sqrt(self.d_model)


class PositionalEncoding(nn.Module):
    def __init__(self, seq_len: int, d_model: int, dropout: float) -> None:
        super().__init__()
        self.seq_len = seq_len
        self.d_model = d_model
        self.dropout = nn.Dropout(dropout)

        # Create a matrix of shape (seq_len, d_model)
        pe = torch.zeros(seq_len, d_model)

        # create a vector of shape (seq_len, 1)
        position = torch.arange(0, seq_len).unsqueeze(1)
        div_factor = torch.exp(torch.arange(0, d_model, 2)
                               * (-math.log(10000.0) / d_model))

        # apply sin on the even position and cosine on the odd
        pe[:, 0::2] = torch.sin(position * div_factor)
        pe[:, 1::2] = torch.cos(position * div_factor)

        # (seq_len, d_model) -> (1, seq_len, d_model)
        pe = pe.unsqueeze(0)

        self.register_buffer('pe', pe)

    def forward(self, x):
        # (bs, seq_len, d_model) -> (bs, seq_len, d_model)
        val = x + (self.pe[:, :x.shape[1], :]).requires_grad_(False)
        return self.dropout(val)


class LayerNorm(nn.Module):
    def __init__(self, features: int, eps: float = 1e-10) -> None:
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(features))
        self.bias = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = torch.mean(x, dim=-1, keepdim=True)
        std = torch.std(x, dim=-1, keepdim=True)
        return self.alpha * ((x - mean) / (std + self.eps)) + self.bias


class FeedFowrard(nn.Module):
    def __init__(self, d_model: int, d_ff: int) -> None:
        super().__init__()
        self.l1 = nn.Linear(d_model, d_ff)
        self.l2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        return self.l2(torch.relu(self.l1(x)))


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, h: int, seq_len: int, dropout: float) -> None:
        super().__init__()
        self.d_model = d_model
        self.h = h

        assert self.d_model // self.h, "d_model is not divisible by h"

        self.d_k = self.d_model // self.h

        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def attention(query, key, value, mask, dropout: nn.Dropout):
        d_k = query.shape[-1]
        score = (query @ key.transpose(-1, -2)) / math.sqrt(d_k)
        if mask is not None:
            score = score.masked_fill_(mask == 1, float("-inf"))
        score = torch.softmax(score, dim=-1)  # (bs, h, seq_len, seq_len)
        if dropout is not None:
            score = dropout(score)
        return score @ value, score

    def forward(self, q, k, v, mask):
        # (bs, seq_len, d_model) -> (bs, seq_len, d_model)
        query = self.w_q(q)
        key = self.w_k(k)
        value = self.w_v(v)

        # (bs, seq_len, d_model) -> (bs, seq_len, h, d_k) -> (bs, h, seq_len, d_k)

        query = query.view(
            query.shape[0], query.shape[1], self.h, self.d_k).transpose(1, 2)
        key = key.view(key.shape[0], key.shape[1],
                       self.h, self.d_k).transpose(1, 2)
        value = value.view(
            value.shape[0], value.shape[1], self.h, self.d_k).transpose(1, 2)

        x, self.attention_score = MultiHeadAttention.attention(
            query, key, value, mask, self.dropout)

        # (bs, h, seq_len, d_k) -> (bs, seq_len, h, d_k) -> (bs, seq_len, h*d_k)
        x = x.transpose(1, 2).contiguous().view(
            x.shape[0], -1, self.h * self.d_k)

        out = self.w_o(x)
        return out


class ResidualBlock(nn.Module):
    def __init__(self, features: int, dropout: float) -> None:
        super().__init__()
        self.norm = LayerNorm(features)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))


class EncoderBlock(nn.Module):
    def __init__(self, features: int, feedforward: FeedFowrard, attention_block: MultiHeadAttention, dropout: float) -> None:
        super().__init__()
        self.forward_block = feedforward
        self.attention_block = attention_block
        self.dropout = nn.Dropout(dropout)
        self.residual_block = nn.ModuleList(
            [ResidualBlock(features, dropout) for _ in range(2)])

    def forward(self, x, src_mask):
        out = self.residual_block[0](
            x, lambda x: self.attention_block(x, x, x, src_mask))

        out = self.residual_block[1](x, self.forward_block)
        return out


class Encoder(nn.Module):
    def __init__(self, features: int, layers: any) -> None:
        super().__init__()
        self.layers = layers
        self.norm = LayerNorm(features)

    def forward(self, x, src_mask):
        for layer in self.layers:
            x = layer(x, src_mask)
        return self.norm(x)


class DecoderBlock(nn.Module):
    def __init__(self, features: int,  feedforward: FeedFowrard, attention_block: MultiHeadAttention, cross_attention: MultiHeadAttention, dropout: float) -> None:
        super().__init__()
        self.forward_block = feedforward
        self.attention_block = attention_block
        self.cross_attention = cross_attention
        self.dropout = nn.Dropout(dropout)
        self.residual_block = nn.ModuleList(
            [ResidualBlock(features, dropout) for _ in range(3)])

    def forward(self, x, encoder_out, src_mask, tgt_mask):
        out = self.residual_block[0](
            x, lambda x: self.attention_block(x, x, x, tgt_mask))
        out = self.residual_block[1](
            x, lambda x: self.cross_attention(x, encoder_out, encoder_out, src_mask))
        out = self.residual_block[2](x, self.forward_block)
        return out


class Decoder(nn.Module):
    def __init__(self, features: int, layers: any) -> None:
        super().__init__()
        self.layers = layers
        self.norm = LayerNorm(features)

    def forward(self, x, encoder_out, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, encoder_out, src_mask, tgt_mask)
        return self.norm(x)


class ProjectionLayer(nn.Module):
    def __init__(self, d_model: int, tgt_vocab_sz: int) -> None:
        super().__init__()
        self.proj = nn.Linear(d_model, tgt_vocab_sz)

    def forward(self, x):
        return self.proj(x)


class Transformer(nn.Module):
    def __init__(self, src_word_embedding: WordEmbedding, src_positional_encoding: PositionalEncoding, encoder: Encoder, tgt_word_embedding: WordEmbedding, tgt_pe: PositionalEncoding, decoder: Decoder, projection: ProjectionLayer) -> None:
        super().__init__()
        self.src_word_embedding = src_word_embedding
        self.src_positional_encoding = src_positional_encoding
        self.tgt_word_embedding = tgt_word_embedding
        self.tgt_pe = tgt_pe
        self.encoder = encoder
        self.decoder = decoder
        self.projection = projection

    def encoder_layer(self, x: torch.tensor, src_mask: torch.tensor):
        x = self.src_word_embedding(x)
        x = self.src_positional_encoding(x)
        x = self.encoder(x, src_mask)
        return x

    def decoder_layer(self, x, encoder_out, src_mask, tgt_mask):
        x = self.tgt_word_embedding(x)
        x = self.tgt_pe(x)
        x = self.decoder(x, encoder_out, src_mask, tgt_mask)
        return x

    def project(self, x):
        return self.projection(x)


def get_model(src_vocab_sz: int, src_seq_len: int, tgt_vocab_sz: int, tgt_seq_len: int, d_model: int = 512, d_ff: int = 2048, h: int = 8, N: int = 6, dropout: float = 0.1):

    # Get the WordEmbedding
    src_word_embedding = WordEmbedding(src_vocab_sz, d_model)
    tgt_word_embedding = WordEmbedding(tgt_vocab_sz, d_model)

    # PositionalEncoding
    src_pe = PositionalEncoding(src_seq_len, d_model, dropout)
    tgt_pe = PositionalEncoding(tgt_seq_len, d_model, dropout)

    # Encoders
    encoders = []
    for _ in range(N):
        forward_block = FeedFowrard(d_model, d_ff)
        attention_block = MultiHeadAttention(d_model, h, src_seq_len, dropout)
        encoder_block = EncoderBlock(
            d_model, forward_block, attention_block, dropout)
        encoders.append(encoder_block)

    # Decoders
    decoders = []
    for _ in range(N):
        forward_block = FeedFowrard(d_model, d_ff)
        attention_block = MultiHeadAttention(d_model, h, tgt_seq_len, dropout)
        cross_attention = MultiHeadAttention(d_model, h, tgt_seq_len, dropout)
        decoder_block = DecoderBlock(d_model,
                                     forward_block, attention_block, cross_attention, dropout)
        decoders.append(decoder_block)

    encoder = Encoder(d_model, nn.ModuleList(encoders))
    decoder = Decoder(d_model, nn.ModuleList(decoders))

    project = ProjectionLayer(d_model, tgt_vocab_sz)

    transformer = Transformer(
        src_word_embedding, src_pe, encoder, tgt_word_embedding, tgt_pe, decoder, project)
    return transformer


if __name__ == "__main__":
    from config import get_config
    from train import get_ds

    config = get_config()
    train_dl, valid_dl, _, _ = get_ds(config)
    batch = next(iter(train_dl))
    encoder_input = batch['encoder_input']
    encoder_mask = batch['encoder_mask']
    decoder_input = batch['decoder_input']
    decoder_mask = batch['decoder_mask']
    model = get_model(30000, 350, 30000, 350)
    encoder_out = model.encoder_layer(encoder_input, encoder_mask)
    print(f"encoder out shape {encoder_out.shape}")
    print(f"decoder input shape {decoder_input.shape}")
    decoder_out = model.decoder_layer(
        decoder_input, encoder_out, encoder_mask, decoder_mask)
    print(f"decoder out shape {type(decoder_out)}")
    proj = model.project(decoder_out)
    print(f"projection shape {proj.shape}")
