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
import io, numpy as np, pyarrow.parquet as pq, av, cv2
from concurrent.futures import ThreadPoolExecutor
from mmengine import fileio
from PIL import Image
from scipy.spatial.transform import Rotation as R
import h5py
from typing import Sequence, Dict, List
import torch

_video_pool = ThreadPoolExecutor(max_workers=6)

def read_bytes(path: str) -> bytes:
    return fileio.get(path)

def open_h5(path: str) -> h5py.File:
    try: return h5py.File(path, "r")
    except OSError: return h5py.File(io.BytesIO(read_bytes(path)), "r")

def read_video_to_frames(path: str) -> np.ndarray:
    buf = io.BytesIO(read_bytes(path))
    container = av.open(buf, options={'threads': 'auto'})
    container.streams.video[0].thread_type = 'AUTO'
    frames = []
    for packet in container.demux(video=0):
        for f in packet.decode(): frames.append(f.to_ndarray(format="rgb24"))
    container.close()
    return np.stack(frames, axis=0)

def read_videos_parallel(paths: List[str]) -> List[np.ndarray]:
    return list(_video_pool.map(read_video_to_frames, paths))


def read_parquet(path: str) -> dict:
    buf = io.BytesIO(read_bytes(path))
    return pq.read_table(buf).to_pydict()

def decode_image_from_bytes(x) -> Image.Image:
    if isinstance(x, (bytes, bytearray)): x = np.frombuffer(x, dtype=np.uint8)
    rgb = cv2.imdecode(x, cv2.IMREAD_COLOR)
    if rgb is None:
        rgb = np.frombuffer(x, dtype=np.uint8)
        if rgb.size == 2764800: rgb = rgb.reshape(720, 1280, 3)
        elif rgb.size == 921600: rgb = rgb.reshape(480, 640, 3)
    else:
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)

def quat_to_rotate6d(q: np.ndarray, scalar_first = False) -> np.ndarray:
    return R.from_quat(q, scalar_first = scalar_first).as_matrix()[..., :, :2].reshape(q.shape[:-1] + (6,))

def euler_to_rotate6d(q: np.ndarray, pattern: str = "xyz") -> np.ndarray:
    return R.from_euler(pattern, q, degrees=False).as_matrix()[..., :, :2].reshape(q.shape[:-1] + (6,))


def rotate6d_to_xyz(v6: np.ndarray) -> np.ndarray:
    v6 = np.asarray(v6)
    if v6.shape[-1] != 6:
        raise ValueError("Last dimension must be 6 (got %s)" % (v6.shape[-1],))
    a1 = v6[..., 0:5:2]
    a2 = v6[..., 1:6:2]
    b1 = a1 / np.linalg.norm(a1, axis=-1, keepdims=True)
    proj = np.sum(b1 * a2, axis=-1, keepdims=True) * b1
    b2 = a2 - proj
    b2 = b2 / np.linalg.norm(b2, axis=-1, keepdims=True)
    b3 = np.cross(b1, b2)
    rot_mats = np.stack((b1, b2, b3), axis=-1)      # shape (..., 3, 3)
    return R.from_matrix(rot_mats).as_euler('xyz')

def rotate6d_to_quat(v6: np.ndarray, scalar_first = False) -> np.ndarray:
    v6 = np.asarray(v6)
    if v6.shape[-1] != 6:
        raise ValueError("Last dimension must be 6 (got %s)" % (v6.shape[-1],))
    a1 = v6[..., 0:5:2]
    a2 = v6[..., 1:6:2]
    b1 = a1 / np.linalg.norm(a1, axis=-1, keepdims=True)
    proj = np.sum(b1 * a2, axis=-1, keepdims=True) * b1
    b2 = a2 - proj
    b2 = b2 / np.linalg.norm(b2, axis=-1, keepdims=True)
    b3 = np.cross(b1, b2)
    rot_mats = np.stack((b1, b2, b3), axis=-1)      # shape (..., 3, 3)
    return R.from_matrix(rot_mats).as_quat(scalar_first = scalar_first)


def action_slice(abs_traj: torch.Tensor, 
                 idx_for_delta: Sequence[int] = (),
                 idx_for_mask_proprio: Sequence[int] = ()
                ) -> Dict[str, torch.Tensor]:
    if not isinstance(abs_traj, torch.Tensor):
        raise TypeError("abs_traj must be a torch.Tensor")
    if abs_traj.ndim != 2 or abs_traj.size(0) < 2:
        raise ValueError("abs_traj must be [H+1, D] with H>=1")

    proprio = abs_traj[0]         # [D]
    action = abs_traj[1:].clone() # [H, D]

    if idx_for_delta:
        idx = torch.as_tensor(idx_for_delta, dtype=torch.long, device=abs_traj.device)
        action[:, idx] -= proprio[idx]
    if idx_for_mask_proprio:
        idx = torch.as_tensor(idx_for_mask_proprio, dtype=torch.long, device=abs_traj.device)
        proprio[idx] = 0.0
    return {"proprio": proprio, "action": action}