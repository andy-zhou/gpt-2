# A collection of special linear layers
import torch
import torch.nn as nn
from torch.nn import functional as F, init

from .embedding import Embedding


class TransposedLinear(nn.Module):
    """
    A linear layer modified such that the HF weights can be loaded correctly.
    nn.Linear stores its weights as (fan_out, fan_in) while the gpt-2 weights expects (fan_in, fan_out)
    due to GPT and GPT-2 using Conv1d.

    nn.Linear: https://github.com/pytorch/pytorch/issues/2159#issuecomment-390068272

    gpt-2: https://github.com/huggingface/transformers/blob/main/src/transformers/pytorch_utils.py#L87
    """

    def __init__(self, fan_in: int, fan_out: int, std=1.0, bias=True):
        super().__init__()
        self.std = std
        self.weight = nn.Parameter(torch.empty((fan_in, fan_out)))
        if bias:
            self.bias = nn.Parameter(torch.empty((fan_out,)))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self):
        init.normal_(self.weight, std=self.std)

    def forward(self, x: torch.Tensor):
        x = x @ self.weight
        if self.bias is not None:
            x = x + self.bias
        return x


class TiedLinear(nn.Module):
    """
    A linear layer with weights tied to an embedding layer. See https://arxiv.org/pdf/1608.05859.
    """

    def __init__(self, tied_embedding_layer: Embedding):
        super().__init__()
        self.weight = tied_embedding_layer.weight

    def forward(self, x: torch.Tensor):
        # I did x @ self.weight.T before, but it ran very slowly. I wonder if torch.compile can find this?
        return F.linear(x, self.weight)
