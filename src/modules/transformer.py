import torch
import torch.nn as nn

from .attention import MultiheadAttention
from .embedding import Embedding
from .mlp import MLP


class TransformerBlock(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        context_len: int,
        dropout=0.0,
        use_flash_attention: bool = False,
    ):
        super().__init__()

        self.ln_1 = nn.LayerNorm(embed_dim)
        self.attn = MultiheadAttention(
            embed_dim,
            num_heads,
            context_len,
            dropout=dropout,
            use_flash_attention=use_flash_attention,
        )
        self.ln_2 = nn.LayerNorm(embed_dim)
        self.mlp = MLP(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor):
        x = x + self.attn(self.ln_1(x))
        x = x + self.dropout(self.mlp(self.ln_2(x)))
        return x


class TransformerModel(nn.Module):
    position_idx: torch.Tensor

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        context_len: int,
        layers: int,
        num_heads=12,
        dropout=0.0,
        use_flash_attention: bool = False,
    ):
        super().__init__()

        self.wte = Embedding(vocab_size, embed_dim, std=0.02)
        # Positional embeddings have a different std for some reason
        self.wpe = Embedding(context_len, embed_dim, std=0.01)
        # Is a buffer here is idiomatic? I like that it responds to model.to(device)
        self.register_buffer(
            "position_idx", torch.arange(context_len), persistent=False
        )

        self.h = nn.Sequential(
            *[
                TransformerBlock(
                    embed_dim,
                    num_heads,
                    context_len,
                    dropout=dropout,
                    use_flash_attention=use_flash_attention,
                )
                for _ in range(layers)
            ]
        )
        self.scale_residual_weights()

        self.ln_f = nn.LayerNorm(embed_dim)

    @torch.no_grad
    def scale_residual_weights(self):
        """
        Scale the attention weights to account for residual stream std inflation
        from TransformerBlocks all contributing to the std of the residual stream
        """
        scaling_factor = len(self.h) ** -0.5
        for layer in self.h:
            assert isinstance(layer, TransformerBlock)
            layer.attn.c_proj.weight *= scaling_factor

    def forward(self, x: torch.Tensor):
        _, T = x.shape
        x = self.wte(x) + self.wpe(self.position_idx[:T])
        x = self.h(x)
        x = self.ln_f(x)
        return x
