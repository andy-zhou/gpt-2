import pytest
import torch
from torch.utils.data import DataLoader
from datasets import (
    Dataset,
    DatasetDict,
    IterableDataset,
    IterableDatasetDict,
    DatasetInfo,
)

from ..tokenized_dataset import TokenizedDataset


@pytest.fixture
def dataset_dict():
    ds = Dataset.from_dict(
        {"tokens": list(range(3000))},
        info=DatasetInfo(splits={"train": {"name": "train", "num_examples": 3000}}),
    )
    return DatasetDict({"train": ds})


@pytest.fixture
def distributed_dataset(dataset_dict: DatasetDict):
    world_size = 3
    datasets = [
        TokenizedDataset(
            context_len=101, rank=i, world_size=world_size, dataset=dataset_dict
        )
        for i in range(3)
    ]
    return world_size, datasets


@pytest.fixture
def iter_dataset_dict():
    ds = IterableDataset.from_generator(lambda: ({"tokens": i} for i in range(3000)))
    ds._info = DatasetInfo(splits={"train": {"name": "train", "num_examples": 3000}})
    return IterableDatasetDict({"train": ds})


@pytest.fixture
def iter_distributed_dataset(iter_dataset_dict):
    world_size = 3
    datasets = [
        TokenizedDataset(
            context_len=101, rank=i, world_size=world_size, dataset=iter_dataset_dict
        )
        for i in range(3)
    ]
    return world_size, datasets


def test_loads_tokenized_dataset(dataset_dict):
    ds = TokenizedDataset(context_len=301, dataset=dataset_dict)
    dl = DataLoader(ds, batch_size=1)
    assert len(dl) == 9

    ds = TokenizedDataset(context_len=1000, dataset=dataset_dict)
    dl = DataLoader(ds, batch_size=2)
    assert len(dl) == 2

    dl = DataLoader(ds, batch_size=2, drop_last=True)
    assert len(dl) == 1

    for context, labels in dl:
        assert context.shape == (2, 1000)
        assert labels.shape == (2, 1000)
        assert torch.equal(context, torch.arange(2 * 1000).reshape(2, 1000))
        assert torch.equal(labels, torch.arange(1, 2 * 1000 + 1).reshape(2, 1000))


def test_loads_iter_backed_dataset(iter_dataset_dict):
    ds = TokenizedDataset(context_len=301, dataset=iter_dataset_dict)
    dl = DataLoader(ds, batch_size=1)
    assert len(dl) == 9

    ds = TokenizedDataset(context_len=1000, dataset=iter_dataset_dict)
    dl = DataLoader(ds, batch_size=2)
    assert len(dl) == 2

    dl = DataLoader(ds, batch_size=2, drop_last=True)
    assert len(dl) == 1

    for context, labels in dl:
        assert context.shape == (2, 1000)
        assert labels.shape == (2, 1000)
        assert torch.equal(context, torch.arange(2 * 1000).reshape(2, 1000))
        assert torch.equal(labels, torch.arange(1, 2 * 1000 + 1).reshape(2, 1000))


def test_loads_distributed_dataset(distributed_dataset):
    world_size, datasets = distributed_dataset
    assert len(datasets) == world_size

    for rank, ds in enumerate(datasets):
        dl = DataLoader(ds, batch_size=2, drop_last=True)
        assert len(dl) == 4

        batches = [batch for batch in dl]
        contexts, labels = zip(*batches)
        contexts = torch.cat(contexts, dim=0)
        labels = torch.cat(labels, dim=0)

        assert contexts.shape == (8, 101)
        expected_context = torch.stack([torch.arange(101)] * 8, dim=0)
        expected_context = (
            expected_context
            + 101 * rank  # shift by rank
            + (torch.arange(8) * 101 * world_size)[:, None]  # shift by world_size
        )
        assert torch.equal(contexts, expected_context)

        assert labels.shape == (8, 101)
        expected_labels = torch.stack([torch.arange(101)] * 8, dim=0) + 1
        expected_labels = (
            expected_labels
            + 101 * rank  # shift by rank
            + (torch.arange(8) * 101 * world_size)[:, None]  # shift by world_size
        )
        assert torch.equal(labels, expected_labels)


def test_loads_iter_backed_distributed_dataset(iter_distributed_dataset):
    world_size, datasets = iter_distributed_dataset
    assert len(datasets) == world_size

    for rank, ds in enumerate(datasets):
        dl = DataLoader(ds, batch_size=2, drop_last=True)
        assert len(dl) == 4

        batches = [batch for batch in dl]
        contexts, labels = zip(*batches)
        contexts = torch.cat(contexts, dim=0)
        labels = torch.cat(labels, dim=0)

        assert contexts.shape == (8, 101)
        expected_context = torch.stack([torch.arange(101)] * 8, dim=0)
        expected_context = (
            expected_context
            + 101 * rank  # shift by rank
            + (torch.arange(8) * 101 * world_size)[:, None]  # shift by world_size
        )
        assert torch.equal(contexts, expected_context)

        assert labels.shape == (8, 101)
        expected_labels = torch.stack([torch.arange(101)] * 8, dim=0) + 1
        expected_labels = (
            expected_labels
            + 101 * rank  # shift by rank
            + (torch.arange(8) * 101 * world_size)[:, None]  # shift by world_size
        )
        assert torch.equal(labels, expected_labels)
