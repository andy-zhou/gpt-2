import torch
import torch.nn as nn


# A linear layer modified such that the HF weights can be loaded correctly.
# nn.Linear stores its weights as (fan_out, fan_in) while the gpt-2 weights expects (fan_in, fan_out)
# due to GPT and GPT-2 using Conv1d.
# nn.Linear: https://github.com/pytorch/pytorch/issues/2159#issuecomment-390068272
# gpt-2: https://github.com/huggingface/transformers/blob/main/src/transformers/pytorch_utils.py#L87
class LinearTransposed(nn.Module):
    def __init__(self, fan_in: int, fan_out: int, bias=True):
        super().__init__()
        self.weight = nn.Parameter(torch.randn((fan_in, fan_out)) * fan_in**-0.5)
        self.bias = (
            nn.Parameter(torch.randn((fan_out,)) * fan_in**-0.5) if bias else None
        )

    def forward(self, x: torch.Tensor):
        x = x @ self.weight
        if self.bias is not None:
            x = x + self.bias
        return x
