from itertools import batched
from typing import Generator, Literal
from datasets import (
    load_dataset,
    IterableDataset as HuggingFaceIterableDataset,
    Dataset as HuggingFaceDataset,
    DatasetDict as HuggingFaceDatasetDict,
    IterableDatasetDict as HuggingFaceIterableDatasetDict,
)
import torch
from torch.utils.data import IterableDataset

from ..utils import PeekableIterator


class TokenizedDataset(IterableDataset):
    def __init__(
        self,
        context_len: int,
        streaming: bool = False,
        split: Literal["train", "test"] = "train",
        rank: int = 0,
        world_size: int = 1,
        dataset: str
        | HuggingFaceDatasetDict
        | HuggingFaceIterableDatasetDict = "azhou890/tinystories-gpt-2",
    ):
        super().__init__()
        self.context_len = context_len
        self.rank = rank
        self.world_size = world_size
        self.split = split

        if isinstance(dataset, str):
            ds = load_dataset(
                dataset, split=self.split, streaming=streaming
            ).with_format("torch")
        else:
            ds = dataset[self.split]
        assert isinstance(ds, (HuggingFaceIterableDataset, HuggingFaceDataset))
        self.ds = ds
        assert self.ds.info.splits
        num_examples = self.ds.info.splits[self.split].num_examples
        assert isinstance(num_examples, int)
        self.num_examples = num_examples

    def __iter__(self) -> Generator[tuple[torch.Tensor, torch.Tensor], None, None]:
        rows = PeekableIterator(self.ds)
        remaining_iterations = len(self)
        if remaining_iterations <= 0:
            return

        for i, batch in enumerate(batched(rows, self.context_len)):
            if i % self.world_size != self.rank:
                continue
            next_row = rows.peek()
            if next_row is not None:
                tokens = torch.tensor(
                    [row["tokens"] for row in [*batch, next_row]], dtype=torch.long
                )
                context, labels = tuple(tokens.unfold(0, self.context_len, 1))
                yield (context, labels)
            if (remaining_iterations := remaining_iterations - 1) <= 0:
                break  # Stop early if the next full batch doesn't fit

    def __len__(self):
        return (self.num_examples // self.context_len) // self.world_size
