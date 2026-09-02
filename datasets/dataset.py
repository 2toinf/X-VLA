# ------------------------------------------------------------------------------
# Copyright 2025 2toINF (https://github.com/2toINF)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ------------------------------------------------------------------------------

from __future__ import annotations

import io
import json
import random
import zlib
from typing import Dict, Iterable

import torch
from mmengine import fileio
from torch.utils.data import IterableDataset, get_worker_info
from torchvision import transforms
from torchvision.transforms import InterpolationMode

from .domain_config import DATA_DOMAIN_ID, DATA_WEIGHTS
from .domain_handler.registry import get_handler_cls
from .streaming import interleave_iterators, shard_indices
from .utils import action_slice


class InfiniteDataReader(IterableDataset):
    """
    Output sample:
      {
        'domain_id': LongTensor[],    # domain id
        'language_instruction': str,
        'image_input': FloatTensor[V, C, H, W],
        'image_mask': BoolTensor[V],
        'proprio': FloatTensor[dim_proprio],
        'action': FloatTensor[T, dim_action]
      }
    """

    def __init__(
        self,
        metas_path: str,
        num_actions: int = 10,
        num_views: int = 3,
        training: bool = True,
        action_mode: str = "ee6d",
        lang_aug: str = None,
        rank: int = 0,
        world_size: int = 1,
        seed: int = 0,
        episode_buffer_size: int = 1,
    ):
        if world_size < 1:
            raise ValueError("world_size must be positive")
        if rank < 0 or rank >= world_size:
            raise ValueError("rank must be in [0, world_size)")
        if episode_buffer_size < 1:
            raise ValueError("episode_buffer_size must be positive")

        self.num_views = num_views
        self.training = training
        self.num_actions = num_actions
        self.action_mode = action_mode
        self.rank = int(rank)
        self.world_size = int(world_size)
        self.seed = int(seed)
        self.episode_buffer_size = int(episode_buffer_size)
        self.metas: Dict[str, dict] = {}
        print("use action mode:", action_mode)
        if fileio.isdir(metas_path):
            meta_files = fileio.list_dir_or_file(metas_path, suffix=".json", recursive=True, list_dir=False)
            root = metas_path
        else:
            meta_files, root = [metas_path], ""

        for file in meta_files:
            file_path = fileio.join_path(root, file)
            with io.BytesIO(fileio.get(file_path)) as f:
                meta = json.load(f)
            # General metadata style.
            if "dataset_name" in meta.keys() and "datalist" in meta.keys():
                print(f"== dataset {meta['dataset_name']} with {len(meta['datalist'])} trajs")
                self.metas[meta["dataset_name"]] = meta
            # LeRobot v2.1 metadata style.
            elif "codebase_version" in meta.keys() and meta["codebase_version"] == "v2.1":
                meta["datalist"] = []
                if "root_path" not in meta.keys():
                    meta["root_path"] = "/".join(file_path.split("/")[:-2])
                with io.BytesIO(
                    fileio.get(fileio.join_path("/".join(file_path.split("/")[:-1]), "episodes.jsonl"))
                ) as f:
                    for line in f:
                        meta["datalist"].append(json.loads(line.decode("utf-8")))
                self.metas[meta["root_path"]] = meta
                print(
                    f"== lerobot dataset {meta['robot_type']} with {meta['total_episodes']} trajs "
                    f"at {meta['root_path']}===="
                )
            else:
                raise NotImplementedError(f"unrecognized meta file format: {file}")

        self.image_aug = [
            transforms.Resize((224, 224), interpolation=InterpolationMode.BICUBIC),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.0)
            if training
            else transforms.Lambda(lambda x: x),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225), inplace=True),
        ]
        self.image_aug = transforms.Compose(self.image_aug)

    def _worker_shard(self) -> tuple[int, int]:
        """Return the global shard id for this rank/worker pair."""

        worker = get_worker_info()
        worker_id = worker.id if worker is not None else 0
        workers_per_rank = worker.num_workers if worker is not None else 1
        num_shards = self.world_size * workers_per_rank
        shard_id = self.rank * workers_per_rank + worker_id
        return shard_id, num_shards

    def _epoch_indices(self, dataset_name: str, epoch: int) -> list[int]:
        """Build the same shuffled episode order on every logical shard."""

        indices = list(range(len(self.metas[dataset_name]["datalist"])))
        if self.training:
            # Do not use the process-local global RNG here: every worker must
            # derive the same permutation before taking its disjoint stride.
            dataset_id = zlib.crc32(dataset_name.encode("utf-8"))
            epoch_seed = self.seed + 1_000_003 * epoch + dataset_id
            random.Random(epoch_seed).shuffle(indices)
        return indices

    def _iter_episode(self, handler, robot_type: str, traj_idx: int, lang_aug_map: dict | None) -> Iterable[dict]:
        """Decode and post-process one episode without changing its iterator."""

        for sample in handler.iter_episode(
            traj_idx,
            num_actions=self.num_actions,
            training=self.training,
            image_aug=self.image_aug,
            lang_aug_map=lang_aug_map,
            action_mode=self.action_mode,
        ):
            sample["domain_id"] = torch.tensor(DATA_DOMAIN_ID.get(robot_type, 0))
            idx_for_delta = sample.pop("idx_for_delta", [])
            idx_for_mask_proprio = sample.pop("idx_for_mask_proprio", [])
            sample.update(action_slice(sample.pop("abs_trajectory", None), idx_for_delta, idx_for_mask_proprio))
            yield sample

    def _iter_one_dataset(
        self, dataset_name: str, shard_id: int | None = None, num_shards: int | None = None
    ) -> Iterable[dict]:
        meta = self.metas[dataset_name]
        if "robot_type" in meta.keys():
            robot_type = meta["robot_type"]
        else:
            robot_type = dataset_name
        Handler = get_handler_cls(robot_type)
        handler = Handler(meta=meta, num_views=self.num_views)

        if shard_id is None or num_shards is None:
            shard_id, num_shards = self._worker_shard()

        epoch = 0
        while True:
            traj_indices = self._epoch_indices(dataset_name, epoch)
            if not traj_indices:
                return
            local_indices = shard_indices(traj_indices, shard_id, num_shards)
            if not local_indices:
                # Tiny metadata files can contain fewer episodes than the
                # number of workers. Keep the stream alive by assigning one
                # deterministic episode to this shard during training;
                # evaluation must not duplicate an episode just to feed an
                # otherwise idle worker.
                if not self.training:
                    return
                local_indices = [traj_indices[shard_id % len(traj_indices)]]

            episode_streams = (
                self._iter_episode(handler, robot_type, traj_idx, meta.get("lang_aug_map"))
                for traj_idx in local_indices
            )
            if self.training:
                yield from interleave_iterators(episode_streams, self.episode_buffer_size)
                epoch += 1
            else:
                for stream in episode_streams:
                    yield from stream
                return

    def __iter__(self):
        shard_id, num_shards = self._worker_shard()
        names = list(self.metas.keys())
        if not self.training:
            for n in names:
                yield from self._iter_one_dataset(n, shard_id, num_shards)
        else:
            # names = names * 2 # increase the dataset sampling frequency
            gens = [iter(self._iter_one_dataset(n, shard_id, num_shards)) for n in names]
            ws = [DATA_WEIGHTS.get(n, 1.0) for n in names]
            s = sum(ws)
            ws = [w / s for w in ws]
            while True:
                i = random.choices(range(len(names)), weights=ws, k=1)[0]
                yield next(gens[i])
