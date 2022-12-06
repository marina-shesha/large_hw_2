import torch.nn as nn
from torch import Tensor
import torch
import math

class PositionalEncoding(nn.Module):

    def __init__(self, embed_dim, max_len: int = 5000):
        """
        Inputs
            embed_dim - Hidden dimensionality of the input.
            max_len - Maximum length of a sequence to expect.
        """
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2) * (-math.log(10000.0) / embed_dim))
        pe = torch.zeros(1, max_len, embed_dim)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        # here should be a tensor of size (1, max_len, embed_dim), dummy dimension is needed for proper addition

        # register_buffer => Tensor which is not a parameter, but should be part of the modules state.
        # Used for tensors that need to be on the same device as the module.
        # persistent=False tells PyTorch to not add the buffer to the state dict (e.g. when we save the model)
        self.register_buffer('pe', pe, persistent=False)

    def forward(self, x):
        x = x + self.pe[:, :x.shape[1]]
        return x


class TranslationModel(nn.Module):
    def __init__(
        self,
        num_encoder_layers: int,
        num_decoder_layers: int,
        emb_size: int,
        dim_feedforward: int,
        n_head: int,
        src_vocab_size: int,
        tgt_vocab_size: int,
        dropout_prob: float,
        max_len: int,
        src_pad,
        tgt_pad,
    ):
        """
        Creates a standard Transformer encoder-decoder model.
        :param num_encoder_layers: Number of encoder layers
        :param num_decoder_layers: Number of decoder layers
        :param emb_size: Size of intermediate vector representations for each token
        :param dim_feedforward: Size of intermediate representations in FFN layers
        :param n_head: Number of attention heads for each layer
        :param src_vocab_size: Number of tokens in the source language vocabulary
        :param tgt_vocab_size: Number of tokens in the target language vocabulary
        :param dropout_prob: Dropout probability throughout the model
        """
        super().__init__()
        self.emb_size = emb_size
        self.transformer = nn.Transformer(
            emb_size,
            n_head,
            num_encoder_layers,
            num_decoder_layers,
            dim_feedforward,
            dropout_prob,
            batch_first=True,
            norm_first=True,
        )
        self.linear = nn.Linear(emb_size, tgt_vocab_size)
        self.positional_encoding = PositionalEncoding(emb_size, max_len)
        self.src_embedding = nn.Embedding(src_vocab_size, emb_size, padding_idx=src_pad)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, emb_size, padding_idx=tgt_pad)

    def forward(
        self,
        src_tokens: Tensor,
        tgt_tokens: Tensor,
        tgt_mask: Tensor,
        src_padding_mask: Tensor,
        tgt_padding_mask: Tensor,
    ):
        """
        Given tokens from a batch of source and target sentences, predict logits for next tokens in target sentences.
        """
        src_emb = self.src_embedding(src_tokens)
        src_emb = self.positional_encoding(src_emb)
        tgt_emb = self.tgt_embedding(tgt_tokens)
        tgt_emb = self.positional_encoding(tgt_emb)
        out = self.transformer(src_emb, tgt_emb, tgt_mask=tgt_mask,
                               src_key_padding_mask=src_padding_mask, tgt_key_padding_mask=tgt_padding_mask)
        out = self.linear(out)
        return out

    def encode(self, src_tokens: Tensor, src_padding_mask: Tensor):
        return self.transformer.encoder(self.positional_encoding(
                            self.src_embedding(src_tokens)),  src_key_padding_mask=src_padding_mask)

    def decode(self, tgt_tokens: Tensor, memory: Tensor, tgt_mask: Tensor):
        out = self.transformer.decoder(self.positional_encoding(
                          self.tgt_embedding(tgt_tokens)), memory=memory, tgt_mask=tgt_mask)
        out = self.linear(out[:, -1])
        return out
