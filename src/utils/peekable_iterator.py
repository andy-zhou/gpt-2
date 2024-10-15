from collections.abc import Iterable


class PeekableIterator[Item]:
    def __init__(self, it: Iterable[Item]):
        self.it = iter(it)
        self.peeked = None

    def __iter__(self):
        return self

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
