from collections.abc import Iterator, Iterable


def skip(it: Iterator, n: int):
    """Skip the first n elements of the given iterable."""
    try:
        for _ in range(n):
            next(it)
    except StopIteration:
        pass
    return it


def next_n(it: Iterator, n: int):
    """Return the next n elements of the given iterable. Similar to islice but includes some additional checks."""
    return NextNIterator(it, n)


class PeekableIterator[Item](Iterator[Item]):
    def __init__(self, it: Iterable[Item]):
        self.it = iter(it)
        self.peeked = None

    def __next__(self):
        if self.peeked is not None:
            result = self.peeked
            self.peeked = None
            return result
        return next(self.it)

    def peek(self):
        if self.peeked is None:
            self.peeked = next(self.it, None)
        return self.peeked


class NextNIterator[Item](Iterator[Item]):
    def __init__(self, underlying_iter: Iterator[Item], n: int):
        self._underlying_iter = underlying_iter
        self.n = n
        self.taken = 0

    def exhausted(self):
        return self.taken >= self.n

    def __next__(self):
        if self.exhausted():
            raise StopIteration()
        self.taken += 1
        try:
            return next(self._underlying_iter)
        except StopIteration:
            raise RuntimeError("Underlying iterator exhausted")

    def __len__(self):
        return self.n
