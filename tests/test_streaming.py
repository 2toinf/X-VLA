from types import SimpleNamespace

import torch

from datasets import dataset as dataset_module
from datasets.dataset import InfiniteDataReader
from datasets.streaming import interleave_iterators, shard_indices


def test_shard_indices_are_disjoint_and_cover_the_input():
    indices = list(range(17))
    shards = [shard_indices(indices, shard_id, 4) for shard_id in range(4)]

    assert sorted(item for shard in shards for item in shard) == indices
    assert len({item for shard in shards for item in shard}) == len(indices)


def test_interleave_iterators_uses_a_bounded_round_robin_pool():
    streams = ([f"episode-{episode}:{step}" for step in range(length)] for episode, length in [(0, 3), (1, 2), (2, 2)])

    result = list(interleave_iterators((iter(stream) for stream in streams), max_active=2))

    assert result == [
        "episode-0:0",
        "episode-1:0",
        "episode-0:1",
        "episode-1:1",
        "episode-0:2",
        "episode-2:0",
        "episode-2:1",
    ]


def test_worker_shard_includes_ddp_rank(monkeypatch):
    reader = InfiniteDataReader.__new__(InfiniteDataReader)
    reader.rank = 2
    reader.world_size = 3
    monkeypatch.setattr(dataset_module, "get_worker_info", lambda: SimpleNamespace(id=1, num_workers=4))

    assert reader._worker_shard() == (9, 12)


def test_episode_order_is_shared_before_rank_striding():
    reader_a = InfiniteDataReader.__new__(InfiniteDataReader)
    reader_b = InfiniteDataReader.__new__(InfiniteDataReader)
    for reader in (reader_a, reader_b):
        reader.training = True
        reader.seed = 123
        reader.metas = {"demo": {"datalist": ["a", "b", "c", "d", "e", "f"]}}

    assert reader_a._epoch_indices("demo", 4) == reader_b._epoch_indices("demo", 4)


def test_training_reader_interleaves_distinct_episodes(monkeypatch):
    class FakeHandler:
        def __init__(self, meta, num_views):
            self.meta = meta

        def iter_episode(self, traj_idx, **kwargs):
            del kwargs
            for step in range(2):
                yield {"episode": traj_idx, "abs_trajectory": torch.tensor([[0.0], [float(step + 1)]])}

    reader = InfiniteDataReader.__new__(InfiniteDataReader)
    reader.training = True
    reader.seed = 0
    reader.rank = 0
    reader.world_size = 1
    reader.num_views = 1
    reader.num_actions = 1
    reader.action_mode = "ee6d"
    reader.episode_buffer_size = 2
    reader.image_aug = None
    reader.metas = {"demo": {"datalist": list(range(3)), "dataset_name": "demo"}}
    monkeypatch.setattr(dataset_module, "get_handler_cls", lambda _: FakeHandler)
    monkeypatch.setattr(dataset_module, "DATA_DOMAIN_ID", {"demo": 0})
    monkeypatch.setattr(reader, "_epoch_indices", lambda _name, _epoch: [0, 1, 2])

    stream = reader._iter_one_dataset("demo", shard_id=0, num_shards=1)
    samples = [next(stream) for _ in range(4)]

    assert [sample["episode"] for sample in samples] == [0, 1, 0, 1]
