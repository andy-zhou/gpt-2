import torch
import torch.nn as nn

from .linear import TransposedLinear


class MLP(nn.Module):
    def __init__(self, embed_dim: int):
        super().__init__()

        self.embed_dim = embed_dim
        self.hidden_dim = 4 * embed_dim  # This was hardcoded in the original model
        self.c_fc = TransposedLinear(self.embed_dim, self.hidden_dim)
        self.c_proj = TransposedLinear(self.hidden_dim, self.embed_dim)

        # The original implementation seems to use approximate version
        # https://github.com/openai/gpt-2/blob/9b63575ef42771a015060c964af2c3da4cf7c8ab/src/model.py#L25
        self.activation = nn.GELU(approximate="tanh")

    def forward(self, x: torch.Tensor):
        return self.c_proj(self.activation(self.c_fc(x)))
