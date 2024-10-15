import pytest
import torch
from datasets import (
    Dataset,
    DatasetDict,
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


def test_tokenized_dataset(dataset_dict):
    context_len = 1000
    ds = TokenizedDataset(context_len, dataset=dataset_dict)

    # We expect this to be 2 since we need to offset by the label
    assert len(ds) == 2
    blocks = list(ds)
    assert len(blocks) == 2

    context, labels = zip(*blocks)
    context = torch.stack(context, dim=0)
    labels = torch.stack(labels, dim=0)

    assert context.shape == (len(ds), context_len)
    expected_context = torch.arange(len(ds) * context_len).view(len(ds), context_len)
    assert torch.equal(context, expected_context)

    assert labels.shape == (len(ds), context_len)
    expected_labels = torch.arange(len(ds) * context_len).view(len(ds), context_len) + 1
    assert torch.equal(labels, expected_labels)


def test_tokenized_dataset_can_make_independent_iterators(dataset_dict):
    context_len = 1000
    ds = TokenizedDataset(context_len, dataset=dataset_dict)

    it1 = iter(ds)
    it2 = iter(ds)

    blocks1 = list(it1)
    blocks2 = list(it2)

    assert len(blocks1) == 2
    assert len(blocks2) == 2

    context1, labels1 = zip(*blocks1)
    context1 = torch.stack(context1, dim=0)
    labels1 = torch.stack(labels1, dim=0)

    context2, labels2 = zip(*blocks2)
    context2 = torch.stack(context2, dim=0)
    labels2 = torch.stack(labels2, dim=0)

    assert torch.equal(context1, context2)
    assert torch.equal(labels1, labels2)


def test_loads_distributed_dataset(dataset_dict):
    world_size = 3
    context_len = 100
    datasets = [
        TokenizedDataset(
            context_len, rank=i, world_size=world_size, dataset=dataset_dict
        )
        for i in range(3)
    ]

    for rank, ds in enumerate(datasets):
        # We expect only 9 batches since we need offset by the label
        assert len(ds) == 9
        batches = [batch for batch in ds]
        assert len(batches) == 9

        context, labels = zip(*batches)
        context = torch.stack(context, dim=0)
        labels = torch.stack(labels, dim=0)

        rank_offset = rank * context_len
        world_size_offset = world_size * context_len

        assert context.shape == (len(ds), context_len)
        expected_context = torch.stack(
            [
                torch.arange(context_len) + rank_offset + world_size_offset * i
                for i in range(len(ds))
            ],
            dim=0,
        )
        assert torch.equal(context, expected_context)

        assert labels.shape == (len(ds), context_len)
        expected_labels = (
            torch.stack(
                [
                    torch.arange(context_len) + rank_offset + world_size_offset * i
                    for i in range(len(ds))
                ],
                dim=0,
            )
            + 1
        )
        assert torch.equal(labels, expected_labels)
