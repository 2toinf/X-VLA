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
from typing import Dict, Iterable, List
import io, json, random, numpy as np, torch
from torch.utils.data import IterableDataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from mmengine import fileio
from .utils import action_slice
from .domain_config import DATA_WEIGHTS, DATA_DOMAIN_ID
from .domain_handler.registry import get_handler_cls

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
    def __init__(self, 
                 metas_path: str, 
                 num_actions: int = 10, 
                 num_views: int = 3, 
                 training: bool = True,
                 action_mode: str = "ee6d",
                 lang_aug: str = None,
                 ):
        self.num_views = num_views
        self.training = training
        self.num_actions = num_actions
        self.action_mode = action_mode
        self.metas: Dict[str, dict] = {}
        print("use action mode:", action_mode)
        if fileio.isdir(metas_path):
            meta_files = fileio.list_dir_or_file(metas_path, suffix=".json", recursive=True, list_dir=False)
            root = metas_path
        else: meta_files, root = [metas_path], ""
        
        for file in meta_files:
            file_path = fileio.join_path(root, file)
            with io.BytesIO(fileio.get(file_path)) as f: meta = json.load(f)
            ### General Style
            try:
                if 'dataset_name' in meta.keys() and 'datalist' in meta.keys():
                    print(f"== dataset {meta['dataset_name']} with {len(meta['datalist'])} trajs")
                    self.metas[meta["dataset_name"]] = meta
                ### Lerobot v2.1 style
                elif "codebase_version" in meta.keys() and meta["codebase_version"] == 'v2.1':
                    meta['datalist'] = []
                    if "root_path" not in meta.keys(): meta['root_path'] = "/".join(file_path.split("/")[:-2])
                    with io.BytesIO(fileio.get(fileio.join_path("/".join(file_path.split("/")[:-1]), "episodes.jsonl"))) as f:
                        for line in f: meta['datalist'].append(json.loads(line.decode("utf-8")))
                    self.metas[meta['root_path']] = meta
                    print(f"== lerobot dataset {meta['robot_type']} with {meta['total_episodes']} trajs at {meta['root_path']}====")
                else: raise NotImplementedError(f"unrecognized meta file format: {file}")
            except:
                print(f"!! failed to load meta file: {file}")

        self.image_aug = [
            transforms.Resize((224, 224), interpolation=InterpolationMode.BICUBIC),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.) \
                if training else transforms.Lambda(lambda x: x),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225), inplace=True),
        ]
        self.image_aug = transforms.Compose(self.image_aug)

    def _iter_one_episode(self, dataset_name: str, traj_idx: int) -> Iterable[dict]:
        meta = self.metas[dataset_name]
        robot_type = meta.get('robot_type', dataset_name)
        Handler = get_handler_cls(robot_type)
        handler = Handler(meta=meta, num_views=self.num_views)
        lang_aug_map = meta.get("lang_aug_map")
        domain_id = torch.tensor(DATA_DOMAIN_ID.get(robot_type, 0))

        for sample in handler.iter_episode(
            traj_idx,
            num_actions=self.num_actions,
            training=self.training,
            image_aug=self.image_aug,
            lang_aug_map=lang_aug_map,
            action_mode=self.action_mode
        ):
            sample["domain_id"] = domain_id
            idx_for_delta = sample.pop("idx_for_delta", [])
            idx_for_mask_proprio = sample.pop("idx_for_mask_proprio", [])
            sample.update(action_slice(sample.pop("abs_trajectory", None), idx_for_delta, idx_for_mask_proprio))
            yield sample

    def _dataset_weight(self, dataset_name: str) -> float:
        meta = self.metas[dataset_name]
        robot_type = meta.get("robot_type", meta.get("dataset_name", dataset_name))
        return DATA_WEIGHTS.get(robot_type, DATA_WEIGHTS.get(dataset_name, 1.0))

    def __iter__(self):
        names = list(self.metas.keys())
        if not self.training:
            for n in names:
                for traj_idx in range(len(self.metas[n]["datalist"])):
                    yield from self._iter_one_episode(n, traj_idx)
        else:
            ws = [self._dataset_weight(n) for n in names]
            s = sum(ws); ws = [w / s for w in ws]
            queues: Dict[str, List[int]] = {}
            def _refill(name):
                idxs = list(range(len(self.metas[name]["datalist"])))
                random.shuffle(idxs)
                queues[name] = idxs
            for n in names:
                _refill(n)
            while True:
                i = random.choices(range(len(names)), weights=ws, k=1)[0]
                name = names[i]
                if not queues[name]:
                    _refill(name)
                traj_idx = queues[name].pop()
                yield from self._iter_one_episode(name, traj_idx)
