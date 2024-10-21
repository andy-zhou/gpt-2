from statistics import mean
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from ..modules import GPT2Output


EVAL_GENERATOR_SEED = 42
EVAL_DEVICE = "cuda"


@torch.no_grad()
def eval_gpt2(
    model: nn.Module,
    dataset: Dataset,
    batch_size: int = 4,
) -> float:
    model.eval()

    g = torch.Generator().manual_seed(EVAL_GENERATOR_SEED)
    dataloader = DataLoader(dataset, batch_size=batch_size, generator=g)
    losses = []
    for context, labels in dataloader:
        assert isinstance(context, torch.LongTensor)
        assert isinstance(labels, torch.LongTensor)
        context, labels = context.to(EVAL_DEVICE), labels.to(EVAL_DEVICE)
        output = model(context, labels)
        assert isinstance(output, GPT2Output)
        assert output.loss is not None
        losses.append(output.loss.item())
    return mean(losses)
