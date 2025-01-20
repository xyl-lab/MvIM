from typing import Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from abc import abstractmethod
from einops import repeat


class AttentionOperator(nn.Module):
    """
    The abstract class for all attention layers.
    """

    def __init__(self):
        super().__init__()

    @abstractmethod
    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError


class ScaledDotProductAttention(AttentionOperator):
    """Scaled dot-product attention.

    Parameters
    ----------
    temperature:
        The temperature for scaling.

    attn_dropout:
        The dropout rate for the attention map.

    """

    def __init__(self, temperature: float, attn_dropout: float = 0.1):
        super().__init__()
        assert temperature > 0, "temperature should be positive"
        assert attn_dropout >= 0, "dropout rate should be non-negative"
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout) if attn_dropout > 0 else None

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward processing of the scaled dot-product attention.

        Parameters
        ----------
        q:
            Query tensor.

        k:
            Key tensor.

        v:
            Value tensor.

        attn_mask:
            Masking tensor for the attention map. The shape should be [batch_size, n_heads, n_steps, n_steps].
            0 in attn_mask means values at the according position in the attention map will be masked out.

        Returns
        -------
        output:
            The result of Value multiplied with the scaled dot-product attention map.

        attn:
            The scaled dot-product attention map.

        """
        # q, k, v all have 4 dimensions [batch_size, n_steps, n_heads, d_tensor]
        # d_tensor could be d_q, d_k, d_v

        # transpose for attention dot product: [batch_size, n_heads, n_steps, d_k or d_v]
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        # dot product q with k.T to obtain similarity
        # print(q.shape, k.shape)
        attn = torch.matmul(q / self.temperature, k.transpose(2, 3))
        # embedded attention
        # q = torch.softmax(q, dim=-1)
        # k = torch.softmax(k, dim=-1)
        # k = k.transpose(2, 3)
        # print(q.shape, k.shape)
        # q = repeat(q, "n s1 s2 -> b n s1 s2", b=batch_size)
        # k = repeat(k, "n s2 s1 -> b n s2 s1", b=batch_size)
        # attn = q @ k

        # apply masking on the attention map, this is optional
        # print(attn.shape)
        # print(attn_mask.shape)
        if attn_mask is not None:
            attn = attn.masked_fill(attn_mask == 0, -1e9)

        # compute attention score [0, 1], then apply dropout
        attn = F.softmax(attn, dim=-1)
        if self.dropout is not None:
            attn = self.dropout(attn)

        # multiply the score with v
        output = torch.matmul(attn, v)
        return output, attn


class MultiHeadAttention(nn.Module):
    """Transformer multi-head attention module.

    Parameters
    ----------
    attn_opt:
        The attention operator, e.g. the self-attention proposed in Transformer.

    d_model:
        The dimension of the input tensor.

    n_heads:
        The number of heads in multi-head attention.

    d_k:
        The dimension of the key and query tensor.

    d_v:
        The dimension of the value tensor.

    """

    def __init__(
        self,
        # attn_opt: AttentionOperator,
        d_model: int,
        n_heads: int,
        d_k: int,
        d_v: int,
    ):
        super().__init__()

        self.n_heads = n_heads
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_model, n_heads * d_k, bias=False)
        self.w_ks = nn.Linear(d_model, n_heads * d_k, bias=False)
        self.w_vs = nn.Linear(d_model, n_heads * d_v, bias=False)

        # self.attention_operator = attn_opt
        self.head_dim = d_k
        self.fc = nn.Linear(n_heads * d_v, d_model, bias=False)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attn_mask: Optional[torch.Tensor],
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward processing of the multi-head attention module.

        Parameters
        ----------
        q:
            Query tensor.

        k:
            Key tensor.

        v:
            Value tensor.

        attn_mask:
            Masking tensor for the attention map. The shape should be [batch_size, n_heads, n_steps, n_steps].
            0 in attn_mask means values at the according position in the attention map will be masked out.

        Returns
        -------
        v:
            The output of the multi-head attention layer.

        attn_weights:
            The attention map.

        """
        # the shapes of q, k, v are the same [batch_size, n_steps, d_model]

        batch_size, q_len = q.size(0), q.size(1)
        k_len = k.size(1)
        v_len = v.size(1)

        # now separate the last dimension of q, k, v into different heads -> [batch_size, n_steps, n_heads, d_k or d_v]
        q = self.w_qs(q)
        k = self.w_ks(k)
        v = self.w_vs(v)
        q = torch.cat(torch.split(q, self.head_dim, dim=-1), dim=0)
        k = torch.cat(torch.split(k, self.head_dim, dim=-1), dim=0)
        v = torch.cat(torch.split(v, self.head_dim, dim=-1), dim=0)

        k = k.transpose(
            -1, -2
        )
        attn_score = (
            q @ k
        ) / self.head_dim**0.5  # (num_heads * batch_size, ..., tgt_length, src_length)
        if attn_mask is not None:
            attn_score = attn_score.masked_fill(attn_mask == 0, -1e9)
        attn_score = torch.softmax(attn_score, dim=-1)
        out = attn_score @ v
        out = torch.cat(
            torch.split(out, batch_size, dim=0), dim=-1
        )
        out = self.fc(out)

        return out
        # q = self.w_qs(q).view(batch_size, q_len, self.n_heads, self.d_k)
        # k = self.w_ks(k).view(batch_size, k_len, self.n_heads, self.d_k)
        # v = self.w_vs(v).view(batch_size, v_len, self.n_heads, self.d_v)
        # for generalization, we don't do transposing here but leave it for the attention operator if necessary

        # if attn_mask is not None:
        #     # broadcasting on the head axis
        #     attn_mask = attn_mask.unsqueeze(1)

        # v, attn_weights = self.attention_operator(q, k, v, attn_mask, **kwargs)

        # # transpose back -> [batch_size, n_steps, n_heads, d_v]
        # # then merge the last two dimensions to combine all the heads -> [batch_size, n_steps, n_heads*d_v]
        # v = v.transpose(1, 2).contiguous().view(batch_size, q_len, -1)
        # v = self.fc(v)

        # return v, attn_weights


class EmbeddedAttention(nn.Module):
    """
    Spatial embedded attention layer.
    The node embedding serves as the query and key matrices for attentive aggregation on graphs.
    """

    def __init__(self, model_dim, node_embedding_dim):
        super().__init__()

        self.model_dim = model_dim
        self.FC_Q_K = nn.Linear(node_embedding_dim, model_dim)
        self.FC_V = nn.Linear(model_dim, model_dim)
        self.out_proj = nn.Linear(model_dim, model_dim)

    def forward(self, value, emb, attn_mask: Optional[torch.Tensor] = None):
        # V (batch_size, ..., seq_length, model_dim)
        # emb (..., length, model_dim)
        batch_size = value.shape[0]
        query = self.FC_Q_K(emb)
        key = self.FC_Q_K(emb)
        value = self.FC_V(value)

        # Q, K (..., length, model_dim)
        # V (batch_size, ..., length, model_dim)
        key = key.transpose(-1, -2)  # (..., model_dim, src_length)
        # attn_score = query @ key  # (..., tgt_length, src_length)
        # attn_score = torch.softmax(attn_score, dim=-1)
        # attn_score = repeat(attn_score, 'n s1 s2 -> b n s1 s2', b=batch_size)

        # re-normalization
        query = torch.softmax(query, dim=-1)
        key = torch.softmax(key, dim=-1)
        query = repeat(query, "n s1 s2 -> b n s1 s2", b=batch_size)
        key = repeat(key, "n s2 s1 -> b n s2 s1", b=batch_size)

        # out = attn_score @ value  # (batch_size, ..., tgt_length, model_dim)
        attn_score = query @ key
        # print(attn_score.shape)
        if attn_mask is not None:
            attn_mask = attn_mask.unsqueeze(1)
            # print(attn_mask.shape)
            attn_score = attn_score.masked_fill(attn_mask == 0, -1e9)
        out = attn_score @ value
        # out = key @ value  # (batch_size, ..., tgt_length, model_dim)
        # out = query @ out  # (batch_size, ..., tgt_length, model_dim)

        return out


class AttentionLayer(nn.Module):
    """Perform attention across the -2 dim (the -1 dim is `model_dim`).

    Make sure the tensor is permuted to correct shape before attention.

    E.g.
    - Input shape (batch_size, in_steps, num_nodes, model_dim).
    - Then the attention will be performed across the nodes.

    Also, it supports different src and tgt length.

    But must `src length == K length == V length`.

    """

    def __init__(self, model_dim, num_heads=8, mask=False):
        super().__init__()

        self.model_dim = model_dim
        self.num_heads = num_heads
        self.mask = mask
        self.head_dim = model_dim // num_heads

        self.FC_Q = nn.Linear(model_dim, model_dim)
        self.FC_K = nn.Linear(model_dim, model_dim)
        self.FC_V = nn.Linear(model_dim, model_dim)

        self.out_proj = nn.Linear(model_dim, model_dim)

    def forward(self, query, key, value, attn_mask: Optional[torch.Tensor] = None):
        # Q    (batch_size, ..., tgt_length, model_dim)
        # K, V (batch_size, ..., src_length, model_dim)
        batch_size = query.shape[0]
        tgt_length = query.shape[-2]
        src_length = key.shape[-2]

        query = self.FC_Q(query)
        key = self.FC_K(key)
        value = self.FC_V(value)

        # Qhead, Khead, Vhead (num_heads * batch_size, ..., length, head_dim)
        query = torch.cat(torch.split(query, self.head_dim, dim=-1), dim=0)
        key = torch.cat(torch.split(key, self.head_dim, dim=-1), dim=0)
        value = torch.cat(torch.split(value, self.head_dim, dim=-1), dim=0)

        key = key.transpose(
            -1, -2
        )  # (num_heads * batch_size, ..., head_dim, src_length)

        attn_score = (
            query @ key
        ) / self.head_dim**0.5  # (num_heads * batch_size, ..., tgt_length, src_length)

        if self.mask:
            mask = torch.ones(
                tgt_length, src_length, dtype=torch.bool, device=query.device
            ).tril()  # lower triangular part of the matrix
            attn_score.masked_fill_(~mask, -torch.inf)  # fill in-place
        # print(attn_score.shape)
        # print(value.shape)
        # print(attn_mask.shape)
        if attn_mask is not None:
            attn_score = attn_score.masked_fill(attn_mask == 0, -1e9)
        attn_score = torch.softmax(attn_score, dim=-1)
        # (num_heads * batch_size, ..., tgt_length, head_dim)
        out = attn_score @ value
        out = torch.cat(
            torch.split(out, batch_size, dim=0), dim=-1
        )  # (batch_size, ..., tgt_length, head_dim * num_heads = model_dim)

        out = self.out_proj(out)

        return out
