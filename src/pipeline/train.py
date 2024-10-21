from statistics import mean
import torch
import torch.amp
import torch.nn as nn
from torch.optim.optimizer import Optimizer
from tqdm.autonotebook import trange

from ..utils import pad_num
from ..data import TokenizedDataset, MicroBatchDataLoader, MiniBatch
from .eval import eval_gpt2


class MinibatchStatTracker:
    def __init__(self, device: str | int):
        self.forward = torch.cuda.Event(enable_timing=True)
        self.backward = torch.cuda.Event(enable_timing=True)
        self.end = torch.cuda.Event(enable_timing=True)
        self.device = device

        self.train_losses = []
        self.forward_time = 0.0
        self.backward_time = 0.0
        self.tokens_processed = 0

    def record_loss(self, loss: torch.Tensor):
        self.train_losses.append(loss.item())

    def record_forward(self):
        self.forward.record(torch.cuda.current_stream(self.device))

    def record_backward(self):
        self.backward.record(torch.cuda.current_stream(self.device))
        torch.cuda.synchronize(self.device)
        self.backward_time += self.forward.elapsed_time(self.backward)

    def record_end(self):
        self.end.record(torch.cuda.current_stream(self.device))
        torch.cuda.synchronize()
        self.backward_time += self.backward.elapsed_time(self.end)

    def record_tokens_processed(self, tokens_processed: int):
        self.tokens_processed += tokens_processed


class StatTracker:
    def __init__(self):
        self.train_losses = []
        self.forward_time = 0.0
        self.backward_time = 0.0
        self.tokens_processed = 0

    def record_minibatch(self, batch_stat_tracker: MinibatchStatTracker):
        self.train_losses.append(mean(batch_stat_tracker.train_losses))
        self.forward_time += batch_stat_tracker.forward_time
        self.backward_time += batch_stat_tracker.backward_time
        self.tokens_processed += batch_stat_tracker.tokens_processed

    def reset(self):
        self.train_losses = []
        self.forward_time = 0.0
        self.backward_time = 0.0
        self.tokens_processed = 0

    @property
    def average_train_loss(self):
        return mean(self.train_losses)

    @property
    def tokens_per_ms(self):
        return self.tokens_processed / (self.forward_time + self.backward_time)

    @property
    def avg_forward_time(self):
        return self.forward_time / len(self.train_losses)

    @property
    def avg_backward_time(self):
        return self.backward_time / len(self.train_losses)


def train_gpt2_with_grad_accumulation(
    model: nn.Module,
    batch: MiniBatch,
    optimizer: Optimizer,
    device: str | int,
    enable_bf16_amp: bool,
    stat_tracker: MinibatchStatTracker | None,
) -> None:
    # Set up batch
    model.train()
    optimizer.zero_grad()

    for context, labels in batch:
        context, labels = context.to(device), labels.to(device)

        # Forward pass
        if stat_tracker:
            stat_tracker.record_forward()
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=enable_bf16_amp):
            loss = model(context, labels=labels).loss / len(batch)
            assert isinstance(loss, torch.Tensor)

        # Backward pass
        if stat_tracker:
            stat_tracker.record_backward()
        loss.backward()

        # Record stats
        if stat_tracker:
            stat_tracker.record_end()
            stat_tracker.record_loss(loss.detach())
            stat_tracker.record_tokens_processed(context.numel())

    # Update weights
    optimizer.step()


def train_gpt2(
    # Training Params
    model: nn.Module,
    train_dataset: TokenizedDataset,
    eval_dataset: TokenizedDataset | None,
    batch_size: int = 4,
    micro_batch_size: int | None = None,
    num_epochs: int = 1,
    lr: float = 3e-4,
    device: str | int = "cuda",
    # Optimization Params
    enable_tf32: bool = False,
    enable_bf16_amp: bool = False,
    # Logging Params
    logging_interval: int | None = None,
    log_final_iteration: bool = True,
    display_progress: bool = True,
    label: str | None = None,
):
    if enable_tf32:
        torch.set_float32_matmul_precision("high")
    else:
        torch.set_float32_matmul_precision("highest")

    should_log = logging_interval is not None or log_final_iteration
    stat_tracker = StatTracker() if should_log else None

    optim = torch.optim.adamw.AdamW(model.parameters(), lr=lr)
    dataloader = MicroBatchDataLoader(
        train_dataset,
        batch_size=batch_size,
        micro_batch_size=micro_batch_size or batch_size,
    )

    progress_bar = (
        trange(num_epochs * len(dataloader), unit="batches")
        if display_progress
        else None
    )
    for epoch in range(num_epochs):
        for minibatch_idx, minibatch in enumerate(dataloader):
            if progress_bar:
                progress_bar.set_description(
                    (
                        f"{label + ' | ' if label else ''}"
                        f"Epoch {epoch}, Minibatch {minibatch_idx}"
                    )
                )

            minibatch_stat_tracker = (
                MinibatchStatTracker(device) if stat_tracker else None
            )
            train_gpt2_with_grad_accumulation(
                model=model,
                batch=minibatch,
                optimizer=optim,
                device=device,
                enable_bf16_amp=enable_bf16_amp,
                stat_tracker=minibatch_stat_tracker,
            )

            if minibatch_stat_tracker and stat_tracker:
                stat_tracker.record_minibatch(minibatch_stat_tracker)
            if progress_bar:
                progress_bar.update()

            # Log stats
            if (
                eval_dataset
                and stat_tracker
                and progress_bar
                and (
                    logging_interval
                    and minibatch_idx % logging_interval == 0
                    or log_final_iteration
                    and minibatch_idx == len(dataloader) - 1
                )
            ):
                with torch.no_grad():
                    model.eval()
                    progress_bar.write(
                        (
                            f"{label + ' | ' if label else ''}"
                            f"Epoch {pad_num(epoch, padding_multiple=4)} | "
                            f"Minibatch {pad_num(minibatch_idx, padding_multiple=4)} | "
                            f"Avg Train Loss: {stat_tracker.average_train_loss():.3f} | "
                            f"Eval Loss: {eval_gpt2(model=model, dataset=eval_dataset, batch_size=micro_batch_size or batch_size):.3f} | "
                            f"Tokens/ms: {stat_tracker.tokens_per_ms:.2f} | "
                            f"Avg Forward Time: {stat_tracker.avg_forward_time:.2f} | "
                            f"Avg Backward Time: {stat_tracker.avg_backward_time:.2f}"
                        )
                    )
                stat_tracker.reset()
