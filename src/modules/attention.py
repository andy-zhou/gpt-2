# Implemented from scratch, mostly following the API of nn.MultiheadAttention
# Goal is that this can load weights from HF
import torch
import torch.nn as nn
import torch.nn.functional as F

from .linear import TransposedLinear


class MultiheadAttention(nn.Module):
    # The attention mask is called "bias" in the gpt2 model
    # https://github.com/huggingface/transformers/issues/1419#issuecomment-538505604
    bias: torch.Tensor

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        context_len: int,  # not in pytorch, but needed if we want to initialize the mask as a buffer
        dropout=0.0,
        kdim: int | None = None,
        vdim: int | None = None,
        use_flash_attention: bool = False,
        # The below are different from Pytorch's default values:
        # batch_first = True
        # bias = False
    ):
        super().__init__()

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.kdim = kdim if kdim is not None else embed_dim // self.num_heads
        self.vdim = vdim if vdim is not None else embed_dim // self.num_heads

        self.c_attn = TransposedLinear(
            self.embed_dim,
            self.kdim * self.num_heads * 2 + self.vdim * self.num_heads,
            std=0.02,
        )
        self.c_proj = TransposedLinear(
            self.vdim * self.num_heads, self.embed_dim, std=0.02
        )
        self.dropout = nn.Dropout(dropout)

        self.register_buffer(
            "bias",
            torch.tril(torch.ones(1).expand(context_len, context_len)),
            persistent=False,  # Don't try to load this from HF weights
        )

        self.use_flash_attention = use_flash_attention

    def forward(self, x: torch.Tensor):
        B, T, _ = x.shape

        query, key, value = self.c_attn(x).split(
            [
                self.kdim * self.num_heads,
                self.kdim * self.num_heads,
                self.vdim * self.num_heads,
            ],
            dim=-1,
        )

        assert isinstance(key, torch.Tensor)
        assert isinstance(query, torch.Tensor)
        assert isinstance(value, torch.Tensor)

        query = query.view(B, T, self.num_heads, self.kdim)
        key = key.view(B, T, self.num_heads, self.kdim)
        value = value.view(B, T, self.num_heads, self.vdim)
        attn_mask = self.bias[:T, :T] == 1

        if self.use_flash_attention:
            query = query.transpose(1, 2)
            key = key.transpose(1, 2)
            value = value.transpose(1, 2)
            c = F.scaled_dot_product_attention(
                query,
                key,
                value,
                attn_mask,
            )
            c = c.transpose(1, 2).contiguous()
        else:
            attn_weights = torch.einsum("bthk,bihk->bhti", query, key) * (
                self.kdim**-0.5
            )
            attn_weights = attn_weights.masked_fill(
                attn_mask.logical_not(), float("-inf")
            )
            attn_weights = F.softmax(attn_weights, dim=-1)
            c = torch.einsum("bhti,bihv->bthv", attn_weights, value).contiguous()

        c = c.view(B, T, self.num_heads * self.vdim)
        return self.dropout(self.c_proj(c))
