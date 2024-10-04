import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass

from .transformer import TransformerModel
from .linear import TiedLinear


@dataclass
class GPT2Output:
    """
    Emulates the GPT2 output class
    """

    loss: torch.Tensor | None
    logits: torch.Tensor


class GPT2(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        context_len: int,
        layers: int,
        dropout=0.0,
    ):
        super().__init__()

        self.transformer = TransformerModel(
            vocab_size, embed_dim, context_len, layers, dropout=dropout
        )
        self.lm_head = TiedLinear(self.transformer.wte)

    def forward(
        self, context: torch.LongTensor, labels: torch.LongTensor | None = None
    ):
        logits: torch.FloatTensor = self.lm_head(self.transformer(context))
        loss = (
            F.cross_entropy(logits.transpose(1, -1), labels.view(1, -1))
            if labels is not None
            else None
        )
        return GPT2Output(loss=loss, logits=logits)
