from collections.abc import Iterator
from itertools import islice
from typing import Literal
from datasets import (
    load_dataset,
    IterableDataset as HuggingFaceIterableDataset,
    Dataset as HuggingFaceDataset,
    DatasetDict as HuggingFaceDatasetDict,
)
import torch
from torch.utils.data import IterableDataset

from ..utils.more_itertools import PeekableIterator, skip


class TokenizedDataset(IterableDataset[tuple[torch.Tensor, torch.Tensor]]):
    def __init__(
        self,
        context_len: int,
        # Loading params
        dataset: str | HuggingFaceDatasetDict = "azhou890/tinystories-gpt-2",
        streaming: bool = False,
        split: Literal["train", "test"] = "train",
        token_column_name: str = "tokens",
        # Distributed params
        rank: int = 0,
        world_size: int = 1,
    ):
        super().__init__()
        self.context_len = context_len

        # Dataset params
        self.dataset = dataset
        self.streaming = streaming
        self.split = split
        self.token_column_name = token_column_name

        # Distributed params
        self.rank = rank
        self.world_size = world_size

        # Preload the dataset to get stats and validate structure.
        # This should have a negligible impact on performance since the dataset is lazily loaded in
        # streaming mode and is cached in non-streaming mode.
        loaded_dataset = self._load_dataset()
        assert loaded_dataset.info.splits
        num_tokens = loaded_dataset.info.splits[self.split].num_examples
        assert isinstance(num_tokens, int)
        self.num_tokens = num_tokens
        assert (
            loaded_dataset.column_names
            and self.token_column_name in loaded_dataset.column_names
        )

    def _load_dataset(self):
        if isinstance(self.dataset, str):
            # Threadsafe for local paths. https://github.com/huggingface/datasets/blob/d4422cc24a56dc7132ddc3fd6b285c5edbd60b8c/src/datasets/builder.py#L859
            # NB: If we're using s3 or GCS, every process might download the dataset separately.
            ds = load_dataset(
                self.dataset,
                split=self.split,
                streaming=self.streaming,
                save_infos=True,
            )
        else:
            ds = self.dataset[self.split]
        assert isinstance(ds, (HuggingFaceIterableDataset, HuggingFaceDataset))
        return ds

    def __len__(self):
        # +1 for the label offset
        return self.num_tokens // (self.context_len * self.world_size + 1)

    def __iter__(self) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
        remaining_blocks = len(self)
        tokens: PeekableIterator[int] = PeekableIterator(
            row[self.token_column_name]  # type: ignore -- We know the column exists
            for row in self._load_dataset()
        )
        skip(tokens, self.rank * self.context_len)

        while remaining_blocks > 0:
            sampled_tokens = list(islice(tokens, self.context_len))
            next_token = tokens.peek()
            if (len(sampled_tokens) < self.context_len) or (next_token is None):
                raise RuntimeError("Unexpectedly ran out of tokens")

            sampled_tokens.append(next_token)
            context, labels = tuple(
                torch.tensor(sampled_tokens, dtype=torch.long).unfold(
                    0, self.context_len, 1
                )
            )
            yield (context, labels)
            skip(tokens, (self.world_size - 1) * self.context_len)
            remaining_blocks -= 1
