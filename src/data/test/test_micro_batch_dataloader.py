import pytest
import torch

from src.data.micro_batch_dataloader import MicroBatchDataLoader


@pytest.fixture
def dataset():
    return [(torch.tensor([i]), torch.tensor([i + 1])) for i in range(1000)]


def test_micro_batch_dataloader(dataset):
    batch_size = 100
    micro_batch_size = 10
    loader = MicroBatchDataLoader(
        dataset=dataset,
        batch_size=batch_size,
        micro_batch_size=micro_batch_size,
    )

    assert len(loader) == 10

    batch_count = 0
    for batch in loader:
        micro_batch_count = 0
        for context, labels in batch:
            assert context.shape == (10, 1)
            assert labels.shape == (10, 1)

            expected_context = (
                torch.arange(10)[:, None]
                + batch_count * batch_size
                + micro_batch_count * micro_batch_size
            )
            expected_labels = expected_context + 1

            assert torch.equal(context, expected_context)
            assert torch.equal(labels, expected_labels)

            micro_batch_count += 1

        assert micro_batch_count == 10
        batch_count += 1

    assert batch_count == 10


def test_micro_batch_dataloader_invalid_micro_batch_size(dataset):
    batch_size = 100
    micro_batch_size = 0
    with pytest.raises(ValueError):
        MicroBatchDataLoader(
            dataset=dataset,
            batch_size=batch_size,
            micro_batch_size=micro_batch_size,
        )

    batch_size = 100
    micro_batch_size = 9
    with pytest.raises(ValueError):
        MicroBatchDataLoader(
            dataset=dataset,
            batch_size=batch_size,
            micro_batch_size=micro_batch_size,
        )

    micro_batch_size = -1
    with pytest.raises(ValueError):
        MicroBatchDataLoader(
            dataset=dataset,
            batch_size=batch_size,
            micro_batch_size=micro_batch_size,
        )

    micro_batch_size = 101
    with pytest.raises(ValueError):
        MicroBatchDataLoader(
            dataset=dataset,
            batch_size=batch_size,
            micro_batch_size=micro_batch_size,
        )


def test_micro_batch_dataloader_handles_no_batches():
    batch_size = 100
    micro_batch_size = 10
    loader = MicroBatchDataLoader(
        dataset=[(torch.tensor(1), torch.tensor(2))],
        batch_size=batch_size,
        micro_batch_size=micro_batch_size,
    )
    assert len(loader) == 0
    it = iter(loader)
    with pytest.raises(StopIteration):
        next(it)


def test_micro_batch_dataloader_throws_when_batch_not_exhausted(dataset):
    batch_size = 100
    micro_batch_size = 10
    loader = MicroBatchDataLoader(
        dataset=dataset,
        batch_size=batch_size,
        micro_batch_size=micro_batch_size,
    )

    it = iter(loader)
    next(it)

    with pytest.raises(RuntimeError):
        next(it)


def test_micro_batch_dataloader_throws_when_underlying_is_exhausted():
    class ExhaustedUnderlying:
        def __iter__(self):
            return self

        def __next__(self):
            raise StopIteration()

        def __len__(self):
            return 100

    batch_size = 100
    micro_batch_size = 10
    loader = MicroBatchDataLoader(
        dataset=ExhaustedUnderlying(),
        batch_size=batch_size,
        micro_batch_size=micro_batch_size,
    )
    assert len(loader) == 1

    it = iter(loader)
    batch = next(it)
    with pytest.raises(RuntimeError):
        next(batch)


def test_micro_batch_dataloader_handles_empty_dataset():
    batch_size = 100
    micro_batch_size = 10
    loader = MicroBatchDataLoader(
        dataset=[],
        batch_size=batch_size,
        micro_batch_size=micro_batch_size,
    )
    assert len(loader) == 0
    it = iter(loader)
    with pytest.raises(StopIteration):
        next(it)


def test_micro_batch_dataloader_throws_on_one_level_list_comprehension(dataset):
    batch_size = 100
    micro_batch_size = 10
    loader = MicroBatchDataLoader(
        dataset=dataset,
        batch_size=batch_size,
        micro_batch_size=micro_batch_size,
    )
    with pytest.raises(RuntimeError):
        [batch for batch in loader]


def test_micro_batch_dataloader_handles_two_level_list_comprehension(dataset):
    batch_size = 100
    micro_batch_size = 10
    loader = MicroBatchDataLoader(
        dataset=dataset,
        batch_size=batch_size,
        micro_batch_size=micro_batch_size,
    )

    data = [microbatch for batch in loader for microbatch in batch]
    assert len(data) == 100

    context, labels = zip(*data)
    context = torch.cat(context, dim=0)
    assert context.shape == (1000, 1)
    assert torch.equal(context, torch.arange(1000)[:, None])
    labels = torch.cat(labels, dim=0)
    assert labels.shape == (1000, 1)
    assert torch.equal(labels, torch.arange(1, 1001)[:, None])


def test_micro_batch_dataloader_can_make_independent_iterators(dataset):
    batch_size = 100
    micro_batch_size = 10
    loader = MicroBatchDataLoader(
        dataset=dataset,
        batch_size=batch_size,
        micro_batch_size=micro_batch_size,
    )

    it1 = iter(loader)
    it2 = iter(loader)
    assert it1 is not it2

    for batch1, batch2 in zip(it1, it2):
        for (context1, labels1), (context2, labels2) in zip(batch1, batch2):
            assert torch.equal(context1, context2)
            assert torch.equal(labels1, labels2)
