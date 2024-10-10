from statistics import mean
import torch
import torch.amp
from torch.utils.data import Dataset, DataLoader
from tqdm.autonotebook import trange

from ..modules import GPT2
from ..utils import pad_num
from .eval import eval_gpt2


class StatTracker:
    def __init__(self):
        self.train_losses = []
        self.forward_time = 0.0
        self.backward_time = 0.0
        self.tokens_processed = 0

    def record_loss(self, loss: torch.Tensor):
        self.train_losses.append(loss.item())

    def record_forward_time(self, forward_time: float):
        self.forward_time += forward_time

    def record_backward_time(self, backward_time: float):
        self.backward_time += backward_time

    def record_tokens_processed(self, tokens_processed: int):
        self.tokens_processed += tokens_processed

    def average_train_loss(self):
        return mean(self.train_losses)

    def tokens_per_ms(self):
        return self.tokens_processed / (self.forward_time + self.backward_time)

    def avg_forward_time(self):
        return self.forward_time / len(self.train_losses)

    def avg_backward_time(self):
        return self.backward_time / len(self.train_losses)

    def reset(self):
        self.train_losses = []
        self.forward_time = 0.0
        self.backward_time = 0.0
        self.tokens_processed = 0


def train_gpt2(
    # Training Params
    model: GPT2,
    train_dataset: Dataset,
    eval_dataset: Dataset,
    batch_size: int = 4,
    num_epochs: int = 1,
    lr: float = 3e-4,
    generator: torch.Generator | None = None,
    device: str = "cuda",
    # Optimization Params
    enable_tf32: bool = False,
    enable_bf16_amp: bool = False,
    # Logging Params
    logging_interval: int | None = None,
    log_final_iteration: bool = True,
    label: str | None = None,
) -> GPT2:
    assert device == "cuda", "Only cuda is supported for training"
    if enable_tf32:
        torch.set_float32_matmul_precision('high')
    else:
        torch.set_float32_matmul_precision('highest')

    optim = torch.optim.AdamW(model.parameters(), lr=lr)
    dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
        drop_last=True,
    )

    stat_tracker = StatTracker()
    forward = torch.cuda.Event(enable_timing=True)
    backward = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    total_iterations = num_epochs * len(dataloader)
    progress_bar = trange(total_iterations, unit="batches")
    for epoch in range(num_epochs):
        for minibatch, (context, labels) in enumerate(dataloader):
            current_iteration = epoch * len(dataloader) + minibatch
            progress_bar.set_description(
                (
                    f"{label + ' | ' if label else ''}"
                    f"Epoch {epoch}, Minibatch {minibatch}"
                )
            )
            model.train()
            assert isinstance(context, torch.Tensor)
            assert isinstance(labels, torch.Tensor)
            context, labels = context.to(device), labels.to(device)

            # Forward pass
            forward.record()
            with torch.autocast(device, dtype=torch.bfloat16, enabled=enable_bf16_amp):
                loss = model(context, labels=labels).loss
            assert isinstance(loss, torch.Tensor)

            # Backward pass
            backward.record()
            optim.zero_grad()
            loss.backward()
            optim.step()

            # Synchronize for timing
            end.record()
            torch.cuda.synchronize()

            # Record stats
            stat_tracker.record_loss(loss)
            stat_tracker.record_forward_time(forward.elapsed_time(backward))
            stat_tracker.record_backward_time(backward.elapsed_time(end))
            stat_tracker.record_tokens_processed(context.numel())
            progress_bar.update()

            if (
                logging_interval is not None
                and current_iteration % logging_interval == 0
                or log_final_iteration
                and current_iteration == total_iterations - 1
            ):
                with torch.no_grad():
                    model.eval()
                    progress_bar.write(
                        (
                            f"{label + ' | ' if label else ''}"
                            f"Epoch {pad_num(epoch, padding_multiple=4)} | "
                            f"Minibatch {pad_num(minibatch, padding_multiple=4)} | "
                            f"Avg Train Loss: {stat_tracker.average_train_loss():.3f} | "
                            f"Eval Loss: {eval_gpt2(model=model, dataset=eval_dataset, batch_size=batch_size):.3f} | "
                            f"Tokens/ms: {stat_tracker.tokens_per_ms():.2f} | "
                            f"Avg Forward Time: {stat_tracker.avg_forward_time():.2f} | "
                            f"Avg Backward Time: {stat_tracker.avg_backward_time():.2f}"
                        )
                    )
                stat_tracker.reset()
