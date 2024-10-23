import pytest
from src.utils.more_itertools import next_n, skip


def test_skip_works():
    it = iter(range(10))
    assert next(skip(it, 3)) == 3


def test_skip_works_with_empty_iter():
    it = iter([])
    skipped_it = skip(it, 3)
    with pytest.raises(StopIteration):
        next(skipped_it)


def test_skip_works_with_short_iter():
    it = iter([1])
    skipped_it = skip(it, 3)
    with pytest.raises(StopIteration):
        next(skipped_it)


def test_next_n_works():
    it = iter(range(10))
    n = 3
    next_n_it = next_n(it, n)
    assert len(next_n_it) == n
    assert list(next_n_it) == [0, 1, 2]


def test_next_n_works_with_empty_iter():
    it = iter([])
    n = 3
    next_n_it = next_n(it, n)
    assert len(next_n_it) == n
    with pytest.raises(RuntimeError):
        next(next_n_it)
