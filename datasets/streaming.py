"""Small, dependency-free helpers for sharded iterable data streams."""

from __future__ import annotations

from collections import deque
from typing import Iterable, Iterator, Sequence, TypeVar

T = TypeVar("T")


def shard_indices(indices: Sequence[int], shard_id: int, num_shards: int) -> list[int]:
    """Return the indices assigned to one logical data-loader shard.

    ``IterableDataset`` workers do not receive an index sampler from
    ``DataLoader``.  Striding the same, deterministically shuffled sequence is
    therefore the simplest way to make episode ownership disjoint across DDP
    ranks and worker processes.
    """

    if num_shards < 1:
        raise ValueError("num_shards must be positive")
    if shard_id < 0 or shard_id >= num_shards:
        raise ValueError("shard_id must be in [0, num_shards)")
    return list(indices[shard_id::num_shards])


def interleave_iterators(iterators: Iterable[Iterator[T]], max_active: int) -> Iterator[T]:
    """Yield one item at a time from a bounded round-robin episode pool.

    ``iterators`` is consumed lazily, so at most ``max_active`` episode
    iterators are live at once.  When one episode is exhausted it is replaced
    by the next one.  This keeps consecutive training batches from being
    dominated by frames from a single long trajectory without opening every
    episode file simultaneously.
    """

    if max_active < 1:
        raise ValueError("max_active must be positive")

    source = iter(iterators)
    active: deque[Iterator[T]] = deque()
    for _ in range(max_active):
        try:
            active.append(iter(next(source)))
        except StopIteration:
            break

    while active:
        current = active.popleft()
        try:
            item = next(current)
        except StopIteration:
            try:
                active.append(iter(next(source)))
            except StopIteration:
                pass
        else:
            yield item
            active.append(current)
