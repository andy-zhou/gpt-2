import torch
import torch.nn as nn
from torch.nn import functional as F, init


class Embedding(nn.Module):
    """
    An embedding layer that lets us use GPT-2's initialization
    """

    def __init__(self, num_embeddings: int, embedding_dim: int, std: float = 1.0):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(num_embeddings, embedding_dim))
        self.std = std
        self._init_parameters()

    def _init_parameters(self):
        init.normal_(self.weight, std=self.std)

    def forward(self, x: torch.LongTensor):
        return F.embedding(x, self.weight)
