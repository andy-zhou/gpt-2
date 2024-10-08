import torch
import math
from torch.utils.data import Dataset, DataLoader
from tqdm.autonotebook import trange, tqdm

from ..modules import GPT2


def pad_num(num: int, padding_multiple: int = 2):
    digits = 1 if num == 0 else int(math.log10(num)) + 1
    padding_required = ((digits // padding_multiple) + 1) * padding_multiple
    return f"{num:>{padding_required}d}"


def train_gpt2(
    model: GPT2,
    dataset: Dataset,
    batch_size=4,
    num_epochs=1,
    logging_interval=100,
    lr=3e-4,
    device="cpu",
    generator=None,
):
    model.train()
    optim = torch.optim.AdamW(model.parameters(), lr=lr)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
        drop_last=True,
    )

    for epoch in trange(num_epochs, desc="Epoch", unit=" epochs"):
        for minibatch, (context, labels) in enumerate(
            tqdm(dataloader, desc="Minibatch", leave=False)
        ):
            assert isinstance(context, torch.Tensor)
            assert isinstance(labels, torch.Tensor)

            context = context.to(device)
            labels = labels.to(device)

            loss = model(context, labels=labels).loss
            assert isinstance(loss, torch.Tensor)
            optim.zero_grad()
            loss.backward()
            optim.step()

            if minibatch % logging_interval == 0:
                tqdm.write(
                    f"[Epoch {pad_num(epoch)}, Minibatch {pad_num(minibatch)}]: Loss={loss.item():.4f}",
                )
