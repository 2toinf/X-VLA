from __future__ import annotations

from typing import Optional, Tuple, Iterable
import numpy as np
import h5py
import random
from .base import BaseHDF5Handler


class AIRAgilexHQHandler(BaseHDF5Handler):
    """
    AIR-AGILEX-HQ  –  all HDF5 keys read from the meta file.

    Required meta fields:
      action_key            : str   – HDF5 key for eef data, e.g. "observations/eef_6d"
      left_time_key         : str   – HDF5 key for left-arm timestamps
      right_time_key        : str   – HDF5 key for right-arm timestamps
      freq                  : float – recording frequency  (default 30)
      qdur                  : float – query duration in sec (default 2)
      left_dim              : int   – columns for left arm  (default 10)
      grip_scale            : float – grip raw multiplier   (default 1.0)
      tail_margin           : int   – frames to skip at end (default 60)
    """
    dataset_name = "AIR-AGILEX-HQ"

    def build_left_right(
        self, f: h5py.File
    ) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], Optional[np.ndarray], float, float]:
        meta = self.meta
        freq = meta.get("freq", 30.0)
        qdur = meta.get("qdur", 2.0)

        action_key = meta["action_key"]
        left_time_key = meta["left_time_key"]
        right_time_key = meta["right_time_key"]
        left_dim = meta.get("left_dim", 10)

        grip_scale = meta.get("grip_scale", 50.0)

        eef = f[action_key][()]
        left, right = eef[:, :left_dim], eef[:, left_dim:]
        left[:,  -1] *= grip_scale
        right[:, -1] *= grip_scale

        lt = f[left_time_key][()]
        rt = f[right_time_key][()]
        return left, right, lt, rt, freq, qdur

    def index_candidates(self, T_left: int, training: bool) -> Iterable[int]:
        tail_margin = self.meta.get("tail_margin", 60)
        index = list(range(0, max(0, T_left - tail_margin)))
        if training:
            random.shuffle(index)
        return index
