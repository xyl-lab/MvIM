import torch.nn as nn
import torch
from einops import repeat
from typing import Optional, Tuple
import torch.nn.functional as F

from pypots.nn.modules.saits.embedding import SaitsEmbedding
# from pypots.nn.modules.transformer import (
#     TransformerEncoderLayer,
#     ScaledDotProductAttention,
# )
from .TransformerLayers import TransformerEncoderLayer
from .AttentionLayers import ScaledDotProductAttention, AttentionLayer, EmbeddedAttention, MultiHeadAttention


class Dense(nn.Module):
    """A simple fully-connected layer."""

    def __init__(self, input_size, output_size, dropout=0.0, bias=True):
        super(Dense, self).__init__()
        self.layer = nn.Sequential(
            nn.Linear(input_size, output_size, bias=bias),
            nn.ReLU(),
            nn.Dropout(dropout) if dropout > 0.0 else nn.Identity(),
        )

    def forward(self, x):
        return self.layer(x)


class MLP(nn.Module):
    """
    Simple Multi-layer Perceptron encoder with optional linear readout.
    """

    def __init__(
        self, input_size, hidden_size, output_size=None, n_layers=1, dropout=0.0
    ):
        super(MLP, self).__init__()

        layers = [
            Dense(
                input_size=input_size if i == 0 else hidden_size,
                output_size=hidden_size,
                dropout=dropout,
            )
            for i in range(n_layers)
        ]
        self.mlp = nn.Sequential(*layers)

        if output_size is not None:
            self.readout = nn.Linear(hidden_size, output_size)
        else:
            self.register_parameter("readout", None)

    def forward(self, x, u=None):
        """"""
        out = self.mlp(x)
        if self.readout is not None:
            return self.readout(out)
        return out


class ProjectedAttentionLayer(nn.Module):
    """
    Temporal projected attention layer.
    A low-rank factorization is achieved in the temporal attention matrix.
    """

    def __init__(
        self,
        seq_len,
        dim_proj,
        d_model,
        n_heads,
        d_ff=None,
        dropout=0.1,
    ):
        super().__init__()
        d_ff = d_ff or 4 * d_model
        # self.out_attn = AttentionLayer(d_model, n_heads, mask=None)
        # self.in_attn = AttentionLayer(d_model, n_heads, mask=None)
        # self.projector = nn.Parameter(torch.randn(seq_len, dim_proj, d_model))
        self.d_k = d_model // n_heads
        self.slf_attn = MultiHeadAttention(
            d_model, n_heads, self.d_k, self.d_k)
        # self.projector = nn.Parameter(torch.randn(dim_proj, d_model))

        self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.MLP = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.GELU(), nn.Linear(d_ff, d_model)
        )
        self.seq_len = seq_len

    def forward(self, x, attn_mask: Optional[torch.Tensor] = None):
        # x: [b s n d]
        batch = x.shape[0]
        # projector = repeat(
        #     self.projector,
        #     "seq_len dim_proj d_model -> repeat seq_len dim_proj d_model",
        #     repeat=batch,
        # )  # [b, s, c, d]
        # projector = repeat(self.projector, 'dim_proj d_model -> repeat seq_len dim_proj d_model',
        #                       repeat=batch, seq_len=self.seq_len)  # [b, s, c, d]
        # proj
        # message_out = self.out_attn(
        #     projector, x, x, attn_mask
        # )  # [b, s, c, d] <-> [b s n d] -> [b s c d]
        # # hat
        # message_in = self.in_attn(
        #     x, projector, message_out, attn_mask
        # )  # [b s n d] <-> [b, s, c, d] -> [b s n d]
        message_in = self.slf_attn(x, x, x, attn_mask)
        # print(message_in)
        # assert 1 == 0
        # LayerNorm1
        message = x + self.dropout(message_in)
        message = self.norm1(message)
        message = message + self.dropout(self.MLP(message))
        message = self.norm2(message)

        return message


class EmbeddedAttentionLayer(nn.Module):
    def __init__(
        self,
        model_dim,
        node_embedding_dim,
        feed_forward_dim=2048,
        dropout=0,
    ):
        super().__init__()

        self.attn = EmbeddedAttention(model_dim, node_embedding_dim)
        self.feed_forward = nn.Sequential(
            nn.Linear(model_dim, feed_forward_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feed_forward_dim, model_dim),
        )
        self.ln1 = nn.LayerNorm(model_dim)
        self.ln2 = nn.LayerNorm(model_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, emb, attn_mask: Optional[torch.Tensor] = None, dim=-2):
        x = x.transpose(dim, -2)
        # x: (batch_size, ..., length, model_dim)
        # emb: (..., length, model_dim)
        residual = x
        # (batch_size, ..., length, model_dim)
        out = self.attn(x, emb, attn_mask)
        out = self.dropout1(out)
        out = self.ln1(residual + out)

        residual = out
        out = self.feed_forward(out)  # (batch_size, ..., length, model_dim)
        out = self.dropout2(out)
        out = self.ln2(residual + out)

        out = out.transpose(dim, -2)
        return out


class BackboneSAITS(nn.Module):
    def __init__(
        self,
        n_steps: int,
        n_features: int,
        n_layers: int,
        d_model: int,
        n_heads: int,
        d_k: int,
        d_v: int,
        d_ffn: int,
        dropout: float,
        attn_dropout: float,
    ):
        super().__init__()

        # concatenate the feature vector and missing mask, hence double the number of features
        actual_n_features = n_features * 2

        # for the 1st block
        self.embedding_1 = SaitsEmbedding(
            actual_n_features,
            d_model,
            with_pos=True,
            n_max_steps=n_steps,
            dropout=dropout,
        )
        self.layer_stack_for_first_block = nn.ModuleList(
            [
                TransformerEncoderLayer(
                    ScaledDotProductAttention(d_k**0.5, attn_dropout),
                    d_model,
                    n_heads,
                    d_k,
                    d_v,
                    d_ffn,
                    dropout,
                )
                for _ in range(n_layers)
            ]
        )
        self.reduce_dim_z = nn.Linear(d_model, n_features)

        # for the 2nd block
        self.embedding_2 = SaitsEmbedding(
            actual_n_features,
            d_model,
            with_pos=True,
            n_max_steps=n_steps,
            dropout=dropout,
        )
        self.layer_stack_for_second_block = nn.ModuleList(
            [
                TransformerEncoderLayer(
                    ScaledDotProductAttention(d_k**0.5, attn_dropout),
                    d_model,
                    n_heads,
                    d_k,
                    d_v,
                    d_ffn,
                    dropout,
                )
                for _ in range(n_layers)
            ]
        )

        self.reduce_dim_beta = nn.Linear(d_model, n_features)
        self.reduce_dim_gamma = nn.Linear(n_features, n_features)

        # for delta decay factor
        self.weight_combine = nn.Linear(n_features + n_steps, n_features)

    def forward(
        self, X, missing_mask, attn_mask: Optional = None
    ) -> Tuple[torch.Tensor, ...]:

        # first DMSA block
        enc_output = self.embedding_1(
            X, missing_mask
        )  # namely, term e in the math equation
        first_DMSA_attn_weights = None
        for encoder_layer in self.layer_stack_for_first_block:
            enc_output, first_DMSA_attn_weights = encoder_layer(
                enc_output, attn_mask)
        # Linear
        X_tilde_1 = self.reduce_dim_z(enc_output)
        # Replace R1
        X_prime = missing_mask * X + (1 - missing_mask) * X_tilde_1

        # second DMSA block
        enc_output = self.embedding_2(
            X_prime, missing_mask
        )  # namely term alpha in math algo
        second_DMSA_attn_weights = None
        for encoder_layer in self.layer_stack_for_second_block:
            enc_output, second_DMSA_attn_weights = encoder_layer(
                enc_output, attn_mask)
        X_tilde_2 = self.reduce_dim_gamma(
            F.relu(self.reduce_dim_beta(enc_output)))

        # attention-weighted combine
        copy_second_DMSA_weights = second_DMSA_attn_weights.clone()
        copy_second_DMSA_weights = copy_second_DMSA_weights.squeeze(
            dim=1
        )  # namely term A_hat in Eq.
        if len(copy_second_DMSA_weights.shape) == 4:
            # if having more than 1 head, then average attention weights from all heads
            copy_second_DMSA_weights = torch.transpose(
                copy_second_DMSA_weights, 1, 3)
            copy_second_DMSA_weights = copy_second_DMSA_weights.mean(dim=3)
            copy_second_DMSA_weights = torch.transpose(
                copy_second_DMSA_weights, 1, 2)

        # namely term eta
        combining_weights = torch.sigmoid(
            self.weight_combine(
                torch.cat([missing_mask, copy_second_DMSA_weights], dim=2)
            )
        )
        # combine X_tilde_1 and X_tilde_2
        X_tilde_3 = (1 - combining_weights) * X_tilde_2 + \
            combining_weights * X_tilde_1

        return (
            X_tilde_1,
            X_tilde_2,
            X_tilde_3,
            first_DMSA_attn_weights,
            second_DMSA_attn_weights,
            combining_weights,
        )
