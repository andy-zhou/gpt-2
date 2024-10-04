from collections.abc import Callable
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GPT2Tokenizer
from tqdm import tqdm


@torch.no_grad()
def generate_completion(
    prompt: str,
    tokenizer: GPT2Tokenizer,
    model: nn.Module,
    num_completions: int = 3,
    completion_len=40,
    generator: torch.Generator | None = None,
    loading_bar_prefix: str | None = None,
    device: str = "cpu",
) -> list[str]:
    model.eval()

    context = torch.tensor(
        tokenizer.encode(prompt), dtype=torch.long, device=device
    ).expand(num_completions, -1)
    for _ in tqdm(range(completion_len), desc=f"{loading_bar_prefix} ({device})"):
        logits = model(context).logits
        assert isinstance(logits, torch.Tensor)
        probs = F.softmax(logits[:, -1, :], dim=-1)
        next_token = torch.multinomial(probs, num_samples=1, generator=generator)
        context = torch.cat([context, next_token], dim=-1)

    return [tokenizer.decode(toks) for toks in context.tolist()]
