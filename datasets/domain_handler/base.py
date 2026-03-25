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
import random
from abc import ABC, abstractmethod
from typing import Iterable, Tuple, Optional, Sequence, Any

import numpy as np
import h5py
import torch
from mmengine import fileio
from PIL import Image
from scipy.interpolate import interp1d

class DomainHandler(ABC):
    """
    Minimal domain handler interface.

    Subclasses provide dataset-specific decoding by implementing an iterator
    that yields per-sample dictionaries compatible with the training loop.
    """
    dataset_name: str

    def __init__(self, meta: dict, num_views: int) -> None:
        self.meta = meta
        self.num_views = num_views

    @abstractmethod
    def iter_episode(
        self,
        traj_idx: int,
        *,
        num_actions: int,
        training: bool,
        image_aug,
        action_mode,
        lang_aug_map: dict | None,
        **kwargs
    ) -> Iterable[dict]:
        """Yield samples for a single episode."""
        ...


def _open_h5(path: str) -> h5py.File:
    """Open HDF5 from local FS or remote backend via mmengine.fileio."""
    try:
        return h5py.File(path, "r")
    except OSError:
        return h5py.File(io.BytesIO(fileio.get(path)), "r")


class BaseHDF5Handler(DomainHandler):
    """
    Generic HDF5 handler with resource-safe iteration.

    Subclasses only implement:
      - build_left_right(f) -> (left, right, left_time, right_time, freq, qdur)
          left/right: abs_trajectory [T, C], left_time/right_time: optional time arrays [T],
          freq (Hz), qdur (seconds of future window)
      - index_candidates(T_left, training) -> Iterable[int]

    Optionally override:
      - get_image_datasets(f): sequence of image arrays/datasets
      - read_instruction(f): string instruction
    """

    # --- Optional overrides -------------------------------------------------
    def get_image_datasets(self, f: h5py.File) -> Sequence[Any]:
        keys: Sequence[str] = self.meta["observation_key"]
        return [f[k][()] for k in keys]

    def read_instruction(self, f: h5py.File) -> str:
        key: str = self.meta["language_instruction_key"]
        ds = f[key]
        v = ds[()]
        return v.decode() if getattr(ds, "shape", ()) == () else v[0].decode()

    # --- Required hooks -----------------------------------------------------
    def build_left_right(
        self, f: h5py.File
    ) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], Optional[np.ndarray], float, float]:
        raise NotImplementedError

    def index_candidates(self, T_left: int, training: bool) -> Iterable[int]:
        raise NotImplementedError
    # -----------------------------------------------------------------------

    @staticmethod
    def _pil_from_arr(arr: Any) -> Image.Image:
        from ..utils import decode_image_from_bytes
        return decode_image_from_bytes(arr) if not isinstance(arr, Image.Image) else arr

    def iter_episode(
        self,
        traj_idx: int,
        *,
        num_actions: int,
        training: bool,
        image_aug,
        lang_aug_map: dict | None,
        **kwargs
    ) -> Iterable[dict]:
        """Open once, yield many samples; file is always closed on exit."""
        datapath = self.meta["datalist"][traj_idx]
        if not isinstance(datapath, str):
            datapath = datapath[0]

        with _open_h5(datapath) as f:
            # Images and mask
            images = self.get_image_datasets(f)
            # Language
            ins = self.read_instruction(f)
            # Domain-specific kinematics and timing
            left, right, lt, rt, freq, qdur = self.build_left_right(f)
        
        
        image_mask = torch.zeros(self.num_views, dtype=torch.bool)
        image_mask[:len(images)] = True
        if lt is None: lt = np.arange(left.shape[0], dtype=np.float64) / float(freq)
        if rt is None: rt = np.arange(right.shape[0], dtype=np.float64) / float(freq)

        # Candidate indices (optionally shuffled)
        idxs = list(self.index_candidates(left.shape[0], training))
        if training: random.shuffle(idxs)

        # Interpolators; clamp to endpoints
        L = interp1d(lt, left, axis=0, bounds_error=False, fill_value=(left[0], left[-1]))
        R = interp1d(rt, right, axis=0, bounds_error=False, fill_value=(right[0], right[-1]))
        ref = (lt + rt) / 2.0

        V = min(self.num_views, len(images))
        for idx in idxs:

            # Query future window
            cur = ref[idx]
            q = np.linspace(cur, min(cur + qdur, float(ref.max())), num_actions + 1, dtype=np.float32)
            lseq = torch.tensor(L(q))
            rseq = torch.tensor(R(q))

            # Skip static segments
            if (lseq[1] - lseq[0]).abs().max() < 1e-5 and (rseq[1] - rseq[0]).abs().max() < 1e-5: continue
            
            # Language augmentation
            if training and lang_aug_map and ins in lang_aug_map:
                ins = random.choice(lang_aug_map[ins])
            
            imgs = [image_aug(self._pil_from_arr(images[v][idx])) for v in range(V)]
            while len(imgs) < self.num_views: imgs.append(torch.zeros_like(imgs[0]))
            image_input = torch.stack(imgs, dim=0)

            yield {
                "language_instruction": ins,
                "image_input": image_input,
                "image_mask": image_mask,
                "abs_trajectory": torch.nn.functional.pad(
                    torch.cat([lseq, rseq], -1).float(),
                    (0, 20 - lseq.shape[-1] - rseq.shape[-1]),
                ),
                "idx_for_delta": self.meta.get("idx_for_delta", []),
                "idx_for_mask_proprio": self.meta.get("idx_for_mask_proprio", []),
            }