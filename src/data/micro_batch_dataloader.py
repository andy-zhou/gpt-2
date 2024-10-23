from itertools import batched
from typing import Iterator, Protocol

import torch

from ..utils.more_itertools import next_n, NextNIterator


Sample = tuple[torch.Tensor, torch.Tensor]
MicroBatch = tuple[torch.Tensor, torch.Tensor]
MiniBatch = NextNIterator[MicroBatch]


class SampleIterable(Protocol):
    def __len__(self) -> int:
        return 0

    def __iter__(self) -> Iterator[Sample]:
        return iter([])


# A special class used to load data in micro-batches
class MicroBatchDataLoader:
    def __init__(
        self,
        dataset: SampleIterable,
        batch_size: int,
        micro_batch_size: int,
    ):
        if micro_batch_size <= 0:
            raise ValueError("Micro batch size must be a positive integer")
        if micro_batch_size <= 0:
            raise ValueError("Micro batch size must be a positive integer")
        if batch_size % micro_batch_size != 0:
            raise ValueError("Batch size must be divisible by micro batch size")

        self.dataset = dataset
        self.batch_size = batch_size
        self.micro_batch_size = micro_batch_size if micro_batch_size else batch_size
        self.micro_batches_per_batch = self.batch_size // self.micro_batch_size
        self.num_batches = len(dataset) // self.batch_size

    def _micro_batched_dataset_iter(self) -> Iterator[MicroBatch]:
        for microbatch in batched(self.dataset, self.micro_batch_size):
            context, labels = zip(*microbatch)
            if (
                len(context) != self.micro_batch_size
                or len(labels) != self.micro_batch_size
            ):
                return

            yield (torch.stack(context, dim=0), torch.stack(labels, dim=0))

        microbatches = batched(self.dataset, self.micro_batch_size)
        microbatches = (
            (torch.stack(b[0], dim=0), torch.stack(b[1], dim=0)) for b in microbatches
        )
        return microbatches

    def __iter__(self) -> Iterator[MiniBatch]:
        microbatches = self._micro_batched_dataset_iter()
        batch: NextNIterator | None = None

        for _ in range(self.num_batches):
            if batch and not batch.exhausted():
                raise RuntimeError("Previous batch was not exhausted")
            batch = next_n(microbatches, self.micro_batches_per_batch)
            yield batch

    def __len__(self):
        return self.num_batches
