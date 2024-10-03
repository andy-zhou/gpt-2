import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass

from .transformer import TransformerModel


# Emulates the GPT2 output class
@dataclass
class GPT2Output:
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
        # lm_head uses nn.Linear instead of Conv1d
        # See https://github.com/huggingface/transformers/blob/d7950bff82b18c823193d17d72188c5e46d06c83/src/transformers/models/gpt2/modeling_gpt2.py#L1192
        self.lm_head = nn.Linear(embed_dim, vocab_size, bias=False)

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
