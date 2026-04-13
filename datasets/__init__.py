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

import torch
from torch.utils.data import DataLoader
from .dataset import InfiniteDataReader

def worker_init_fn(worker_id: int):
    base_seed = torch.initial_seed() % (2**32)
    import random, numpy as np
    np.random.seed(base_seed); random.seed(base_seed); torch.manual_seed(base_seed)


def create_dataloader(batch_size: int, 
                      metas_path: str, 
                      num_actions: int,
                      training: bool,
                      action_mode: str
                      ):
    return DataLoader(
        InfiniteDataReader(metas_path, num_actions=num_actions, training=training, action_mode=action_mode),
        batch_size=batch_size,
        num_workers=4,
        pin_memory=True,
        worker_init_fn=worker_init_fn,
        persistent_workers=True,
        prefetch_factor=4
    )