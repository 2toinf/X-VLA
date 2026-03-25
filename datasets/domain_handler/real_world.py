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

from typing import Optional, Tuple, Iterable
import numpy as np
import h5py
import random
from ..utils import quat_to_rotate6d
from .base import BaseHDF5Handler


class AIRAgilexHandler(BaseHDF5Handler):
    """
    AIR-AGILEX (non-HQ).
    HDF5:
      /observations/eef_quaternion [T, 16] =
        L_xyz(3) L_quat(4) L_grip_raw(1) R_xyz(3) R_quat(4) R_grip_raw(1)
    Output: left/right [T,10] = xyz(3)+rot6d(6)+grip(1), grip thresholded.
    """
    dataset_name = "AIR-AGILEX"

    def build_left_right(
        self, f: h5py.File
    ) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], Optional[np.ndarray], float, float]:
        freq, qdur = 30.0, 2.0
        eef = f["observations/eef_quaternion"][()]  # [T,16]
        l_xyz, l_quat, l_grip = eef[:, :3], eef[:, 3:7], (eef[:, 7:8] * 50 < 1.0)
        r_xyz, r_quat, r_grip = eef[:, 8:11], eef[:, 11:15], (eef[:, 15:16] * 50 < 1.0)
        left  = np.concatenate([l_xyz, quat_to_rotate6d(l_quat), l_grip], axis=-1)
        right = np.concatenate([r_xyz, quat_to_rotate6d(r_quat), r_grip], axis=-1)
        return left, right, None, None, freq, qdur

    def index_candidates(self, T_left: int, training: bool) -> Iterable[int]:
        # stride 2 for denser lookahead; leave 30 frames margin
        return range(0, max(0, T_left - 30), 2)



class AIRBotHandler(BaseHDF5Handler):
    """
    AIRBOT.
    HDF5:
      /eef_6d [T,10] -> xyz(3)+rot6d(6)+grip_raw(1)
    Single arm (left), right is zeros. Grip <0.5 => closed.
    """
    dataset_name = "AIRBOT"

    def build_left_right(
        self, f: h5py.File
    ) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], Optional[np.ndarray], float, float]:
        freq, qdur = 10.0, 3.0
        eef = f["eef_6d"][()]  # [T,10]
        left = np.concatenate([eef[:, :9], (eef[:, 9:] < 0.5)], axis=-1)
        right = np.zeros_like(left)
        return left, right, None, None, freq, qdur

    def index_candidates(self, T_left: int, training: bool) -> Iterable[int]:
        return range(0, max(0, T_left - 10))


class WidowxAirHandler(BaseHDF5Handler):
    """
    widowx-air.
    HDF5:
      /abs_action_6d [T,10] -> xyz(3)+rot6d(6)+grip_raw(1)
    Single arm; grip <0.5 => closed.
    """
    dataset_name = "widowx-air"

    def build_left_right(
        self, f: h5py.File
    ) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], Optional[np.ndarray], float, float]:
        freq, qdur = 5.0, 5.0
        a = f["abs_action_6d"][()]  # [T,10]
        left = np.concatenate([a[:, :9], (a[:, 9:] < 0.5)], axis=-1)
        right = np.zeros_like(left)
        return left, right, None, None, freq, qdur

    def index_candidates(self, T_left: int, training: bool) -> Iterable[int]:
        return range(0, max(0, T_left - 15))
