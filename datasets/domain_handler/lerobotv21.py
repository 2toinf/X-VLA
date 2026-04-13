from __future__ import annotations
import numpy as np, torch, random
from mmengine import fileio
from scipy.interpolate import interp1d
from ..utils import read_videos_parallel, read_parquet
from PIL import Image
from .base import DomainHandler

class LeRobotV21Handler(DomainHandler):

    # 默认超参数
    # video.overlook_camera_view  / top_camera_view
    CAMERA_VIEW = ["video.overlook_camera_view", "video.left_camera_view", "video.right_camera_view"]
    ACTION_KEY = ["action.joints", "action.base", "action.gripper"] # 12 + 3 + 4
    idx_for_delta = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
    idx_for_mask_proprio = [12, 13, 14, 15, 16, 17, 18]
    ACTION_DIM = 20
    QDUR = 2.0
    TAIL_MARGIN = 30

    def get_camera_views(self):
        return self.meta.get("camera_views", self.CAMERA_VIEW)

    def get_action_keys(self, data: dict):
        keys = self.meta.get("action_keys", self.ACTION_KEY)
        if all(key in data for key in keys):
            return keys

        raise KeyError(f"Unable to resolve LeRobot action keys from meta: {keys}")

    def postprocess_actions(self, raw_actions: np.ndarray) -> np.ndarray:
        raw_actions[:, 12:14] *= 10.0
        raw_actions[:, 14] = np.unwrap(raw_actions[:, 14], period=360) / 10.0
        raw_actions[:, 15:19] *= 20.0
        return raw_actions

    def iter_episode(self, traj_idx: int, *, num_actions: int, training: bool,
                     image_aug, lang_aug_map: dict | None, **kwargs):
        item = self.meta["datalist"][traj_idx]

        episode_index = item["episode_index"]
        episode_chunk = episode_index // self.meta["chunks_size"]

        data_path = fileio.join_path(self.meta["root_path"], self.meta["data_path"]).format(
            episode_chunk=episode_chunk, episode_index=episode_index
        )

        camera_views = self.get_camera_views()
        video_paths = [
            fileio.join_path(self.meta["root_path"], self.meta["video_path"]).format(
                episode_chunk=episode_chunk, episode_index=episode_index, video_key=vkey
            ) for vkey in camera_views
        ]
        images = read_videos_parallel(video_paths)

        image_mask = torch.zeros(self.num_views, dtype=torch.bool)
        image_mask[: min(self.num_views, len(images))] = True
        data = read_parquet(data_path)

        action_keys = self.get_action_keys(data)
        raw_actions = np.concatenate(
            [np.asarray(data[action_key]) for action_key in action_keys], axis=-1
        ).astype(np.float32)
        raw_actions = self.postprocess_actions(raw_actions)


        freq = float(self.meta.get("fps", 30.0))
        qdur = float(self.meta.get("qdur", self.QDUR))
        t = np.arange(raw_actions.shape[0], dtype=np.float64) / freq

        tail_margin = int(self.meta.get("tail_margin", self.TAIL_MARGIN))
        idxs = list(range(0, max(1, raw_actions.shape[0] - tail_margin)))

        if training:
            random.shuffle(idxs)

        interp_func = interp1d(t, raw_actions, axis=0, bounds_error=False,
                              fill_value=(raw_actions[0], raw_actions[-1]))

        ins = item["tasks"][0]
        for idx in idxs:
            imgs = []
            for v in range(min(self.num_views, len(images))):
                imgs.append(image_aug(Image.fromarray(images[v][idx])))

            while len(imgs) < self.num_views:
                imgs.append(torch.zeros_like(imgs[0]))

            image_input = torch.stack(imgs, 0)
            cur_t = t[idx]

            q = np.linspace(cur_t, min(cur_t + qdur, float(t.max())), num_actions + 1, dtype=np.float32)

            cur_action = torch.from_numpy(interp_func(q)).float()

            if cur_action.shape[1] < self.ACTION_DIM:
                padding = torch.zeros((cur_action.shape[0], self.ACTION_DIM - cur_action.shape[1]))
                cur_action = torch.cat([cur_action, padding], dim=-1)

            if lang_aug_map is not None and ins in lang_aug_map:
                ins = random.choice(lang_aug_map[ins])

            yield {
                "language_instruction": ins,
                "image_input": image_input,
                "image_mask": image_mask,
                "abs_trajectory": cur_action,
                "idx_for_delta": self.meta.get("idx_for_delta", self.idx_for_delta),
                "idx_for_mask_proprio": self.meta.get("idx_for_mask_proprio", self.idx_for_mask_proprio)
            }

class ARXAconeLeRobotHandler(LeRobotV21Handler):
    CAMERA_VIEW = ["images.rgb.head", "images.rgb.hand_left", "images.rgb.hand_right"]
    ACTION_KEY = [
        "actions.left_joint.position",
        "actions.left_gripper.position",
        "actions.right_joint.position",
        "actions.right_gripper.position",
    ]
    idx_for_delta = [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12]
    idx_for_mask_proprio = [6, 13]

    def postprocess_actions(self, raw_actions: np.ndarray) -> np.ndarray:
        return raw_actions

class RoboTwin2Handler(LeRobotV21Handler):
    CAMERA_VIEW = ["observation.images.cam_high", "observation.images.cam_left_wrist", "observation.images.cam_right_wrist"]
    ACTION_KEY = ["observation.state"] 
    idx_for_delta = []
    idx_for_mask_proprio = []