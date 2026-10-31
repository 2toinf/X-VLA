import sys
from pathlib import Path
# Add the absolute path of RoboTwin
robowin_root = Path("/home/dodo/fyc/RoboTwin")
# Ensure RoboTwin root is the first search in order to use envs
if str(robowin_root) not in sys.path:
    sys.path.insert(0, str(robowin_root))
# Ensure CWD is RoboTwin in order to load assets
import os
os.chdir(robowin_root)

import argparse
import collections
from collections import Counter, defaultdict
import logging
import os
import importlib
import numpy as np
import torch
import yaml
import json_numpy
import requests
import PIL.Image as Image 
from tqdm import tqdm
import traceback
import json
import random
import imageio
import cv2
import sys
import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
logger = logging.getLogger(__name__)
torch.set_default_dtype(torch.float32)

ALL_TASKS = [
   "adjust_bottle", 
    "beat_block_hammer", 
    "blocks_ranking_rgb", 
    "blocks_ranking_size",
    "click_alarmclock", "click_bell", "dump_bin_bigbin", "grab_roller", "handover_block",
    "handover_mic", "hanging_mug", "lift_pot", 
    "move_can_pot", 
    "move_pillbottle_pad",
    "move_playingcard_away", "move_stapler_pad", "open_laptop", "open_microwave",
    "pick_diverse_bottles", 
    "pick_dual_bottles", 
    "place_a2b_left", "place_a2b_right",
    "place_bread_basket", "place_bread_skillet", "place_burger_fries", "place_can_basket",
    "place_cans_plasticbox", "place_container_plate", 
    "place_dual_shoes", 
    "place_empty_cup",
    "place_fan", "place_mouse_pad", 
    "place_object_basket", 
    "place_object_scale", "place_object_stand", "place_phone_stand", "place_shoe", "press_stapler",
    "put_bottles_dustbin", 
    "put_object_cabinet", "rotate_qrcode", "scan_object",
    "shake_bottle_horizontally", "shake_bottle", 
    "stack_blocks_three", "stack_blocks_two", 
    "stack_bowls_three", 
    "stack_bowls_two", "stamp_seal", "turn_switch"
]

print("Number of tasks evaluating:", len(ALL_TASKS))

def _env_flag(name: str, default: bool = False) -> bool:
    v = os.environ.get(name)
    if v is None:
        return default
    return v.strip().lower() in ("1", "true", "yes", "on")


# Quaternion order (see issue #93).
# RoboTwin reports and consumes end-effector quaternions scalar-first (wxyz,
# transforms3d convention), while SciPy's Rotation uses scalar-last (xyzw).
# The legacy behaviour (scalar_first=False) passes the 4 numbers through
# unchanged on both the observation and the action side; the mismatch cancels
# out end-to-end and this is what the released X-VLA-RoboTwin2 checkpoint was
# trained with. scalar_first=True converts wxyz <-> xyzw explicitly so the
# rotation-6D matches the physical rotation; use it together with
# XVLA_ROBOTWIN_QUAT_SCALAR_FIRST=1 on the training side.
def quat_to_rotate6D(q: np.ndarray, scalar_first: bool = False) -> np.ndarray:
    q = np.asarray(q)
    if scalar_first:
        q = q[..., [1, 2, 3, 0]]  # wxyz -> xyzw (works on any SciPy version)
    return R.from_quat(q).as_matrix()[..., :, :2].reshape(q.shape[:-1] + (6,))


def rotate6D_to_quat(v6: np.ndarray, scalar_first: bool = False) -> np.ndarray:
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
    q = R.from_matrix(rot_mats).as_quat()           # xyzw
    if scalar_first:
        q = q[..., [3, 0, 1, 2]]  # xyzw -> wxyz
    return q

def decode_image_from_bytes(camera_rgb_image):
    if isinstance(camera_rgb_image, (bytes, bytearray)): camera_rgb_image = np.frombuffer(camera_rgb_image, dtype=np.uint8)
    rgb = cv2.imdecode(camera_rgb_image, cv2.IMREAD_COLOR)
    if rgb is None: 
        rgb = np.frombuffer(camera_rgb_image, dtype=np.uint8) 
        if rgb.size == 2764800: 
            rgb = rgb.reshape(720, 1280, 3) 
        elif rgb.size == 921600: 
            rgb = rgb.reshape(480, 640, 3)
    return Image.fromarray(rgb)


class ClientModel:
    def __init__(self, host, port, quat_scalar_first: bool = False):
        self.url = f"http://{host}:{port}/act"
        self.vision_record = []
        # Whether RoboTwin's wxyz quaternions are converted to/from SciPy's
        # xyzw order (True) or passed through unchanged (False, legacy).
        self.quat_scalar_first = quat_scalar_first
        
    def set_instruction(self, instruction):
        self.instruction = instruction
        
    def return_vision_record(self):
        video = np.stack(self.vision_record)
        self.vision_record = []
        return video
    
    def step(self, obs):
        head_view = obs['observation']['head_camera']['rgb']
        left_view = obs['observation']['left_camera']['rgb']
        right_view = obs['observation']['right_camera']['rgb']
        front_view = obs['observation']['front_camera']['rgb']
        image_obs = np.stack([head_view, left_view, right_view, front_view])[None, ]
        left_ee = np.expand_dims(np.array(obs["endpose"]["left_endpose"]), axis=0)     # shape (T, 7)
        right_ee = np.expand_dims(np.array(obs["endpose"]["right_endpose"]), axis=0)    # shape (T, 7)
        left_grip = np.expand_dims(np.array(obs["endpose"]["left_gripper"]), axis=0)    # shape (T,)
        right_grip = np.expand_dims(np.array(obs["endpose"]["right_gripper"]), axis=0)  # shape (T,)
        left_grip = 1 - left_grip * 2
        right_grip = 1 - right_grip * 2
        abs_eef = np.concatenate([
            left_ee[:, :3],
            quat_to_rotate6D(left_ee[:, 3:], self.quat_scalar_first),   # (T,6)
            left_grip[:, None],             # (T,1)
            right_ee[:, :3],
            quat_to_rotate6D(right_ee[:, 3:], self.quat_scalar_first),  # (T,6)
            right_grip[:, None]             # (T,1)
        ], axis=-1)
        self.vision_record.append(image_obs)
        query = {
                "domain_id": 6,
                "proprio": json_numpy.dumps(abs_eef.squeeze(0)), # (1, 14)
                "language_instruction": self.instruction,
                "image0": json_numpy.dumps(head_view),
                "image1": json_numpy.dumps(left_view),
                "image2": json_numpy.dumps(right_view)}
        
        response = requests.post(self.url, json=query)
        action = np.array(response.json()['action'])
        return action

def class_decorator(task_name):
    envs_module = importlib.import_module(f"envs.{task_name}")
    try:
        env_class = getattr(envs_module, task_name)
        env_instance = env_class()
    except:
        raise SystemExit("No such task")
    return env_instance


def get_embodiment_config(robot_file):
    robot_config_file = os.path.join(robot_file, "config.yml")
    with open(robot_config_file, "r", encoding="utf-8") as f:
        embodiment_args = yaml.load(f.read(), Loader=yaml.FullLoader)
    return embodiment_args


def load_env(task_name, task_config):
    CONFIGS_PATH = "task_config"
    task = class_decorator(task_name)

    config_path = os.path.join(CONFIGS_PATH, f"{task_config}.yml")
    with open(config_path, "r", encoding="utf-8") as f:
        args = yaml.load(f.read(), Loader=yaml.FullLoader)

    args['task_name'] = task_name

    embodiment_type = args.get("embodiment")
    embodiment_config_path = os.path.join(CONFIGS_PATH, "_embodiment_config.yml")

    with open(embodiment_config_path, "r", encoding="utf-8") as f:
        _embodiment_types = yaml.load(f.read(), Loader=yaml.FullLoader)

    def get_embodiment_file(embodiment_type):
        robot_file = _embodiment_types[embodiment_type]["file_path"]
        if robot_file is None:
            raise ValueError("missing embodiment files")
        return robot_file

    with open(os.path.join(CONFIGS_PATH, "_camera_config.yml"), "r", encoding="utf-8") as f:
        _camera_config = yaml.load(f.read(), Loader=yaml.FullLoader)

    head_camera_type = args["camera"]["head_camera_type"]
    args["head_camera_h"] = _camera_config[head_camera_type]["h"]
    args["head_camera_w"] = _camera_config[head_camera_type]["w"]

    if len(embodiment_type) == 1:
        args["left_robot_file"] = get_embodiment_file(embodiment_type[0])
        args["right_robot_file"] = get_embodiment_file(embodiment_type[0])
        args["dual_arm_embodied"] = True
    elif len(embodiment_type) == 3:
        args["left_robot_file"] = get_embodiment_file(embodiment_type[0])
        args["right_robot_file"] = get_embodiment_file(embodiment_type[1])
        args["embodiment_dis"] = embodiment_type[2]
        args["dual_arm_embodied"] = False
    else:
        raise ValueError("number of embodiment config parameters should be 1 or 3")

    args["left_embodiment_config"] = get_embodiment_config(args["left_robot_file"])
    args["right_embodiment_config"] = get_embodiment_config(args["right_robot_file"])

    args["embodiment_name"] = "+".join(embodiment_type) if len(embodiment_type) > 1 else embodiment_type[0]
    args["task_config"] = task_config

    return task, args


def _rollout(env, policy):
    success_flag = False
    error_flag = False
    env._update_render()
    if env.render_freq: env.viewer.render()
    env.actor_pose = True
    images = []
    idx = 0
    obs = env.get_obs() # get observation
    # If it is not successful within the specified number of steps (here, 10 steps), it is judged as a failure.‘
    scalar_first = getattr(policy, "quat_scalar_first", False)
    for j in range(10):
        actions = policy.step(obs)
        left_xyz = actions[:, :3]  # (B, 3)
        left_rotate6d = actions[:, 3:9]  # (B, 6)
        left_gripper = actions[:, 9:10]  # (B, 1)  # Use 9:10 to keep 2D shape
        # Convert 6D rotation to quaternion (4 elements)
        left_quat = rotate6D_to_quat(left_rotate6d, scalar_first)  # Should return (B, 4)
        left_grip = 1 - 2 * (left_gripper > 0.7)
        # Ensure all have shape (B, features) and concatenate along axis 1
        left_new = np.concatenate([left_xyz, left_quat, left_grip], axis=1)  # (B, 3+4+1) = (B, 8)
        # Do the same for right arm
        right_xyz = actions[:, 10:13].reshape(-1, 3)  # (B, 3)
        right_rotate6d = actions[:, 13:19].reshape(-1, 6)  # (B, 6)
        right_quat = rotate6D_to_quat(right_rotate6d, scalar_first)  # (B, 4)
        right_gripper = actions[:, 19:20].reshape(-1, 1)  # (B, 1)
        # reset model gripper output
        right_grip = 1 - 2 * (right_gripper > 0.7)
        right_new = np.concatenate([right_xyz, right_quat, right_grip], axis=1)  # (B, 8)
        # Final rollout action (16 dimensions total)
        rollout_action = np.concatenate([left_new, right_new], axis=1)  # (B, 16)
        for action in tqdm(rollout_action):
            env.take_action(action, action_type='ee')  # target_pose: np.array([x, y, z, qw, qx, qy, qz]) (RoboTwin is wxyz)
            obs = env.get_obs()
            obs['endpose']['left_endpose'] = list(action[:7].reshape(7,))
            obs['endpose']['right_endpose'] = list(action[8:-1].reshape(7,))
            images.append(obs["observation"]["head_camera"]["rgb"] )
            idx += 1
            if env.check_success():
                success_flag = True
                break
            if env.actor_pose == False: 
                print('false actor_pose')
                error_flag = True
                break
        if error_flag:
            print("\nfail due to false actor_pose!")
            return 0, images

        if success_flag:
            print("\nsuccess!")
            env.suc +=1
            return 1, images
        
        if env.actor_pose == False:
            print('false actor_pose2')
            break
        
        j += 1
        env._update_render()
    print("\nfail!")
    return 0, images


def eval_episodes(task_name, task_config, policy, test_num=10, seed=0, eval_log_dir=None, instruction_type=None):
    """
    rollout several episodes and log the mean episode return
    """
    if not os.path.exists(os.path.join(eval_log_dir, task_name)):
        print('save to', os.path.join(eval_log_dir, task_name))
        os.makedirs(os.path.join(eval_log_dir, task_name))
    
    TASK_ENV, args = load_env(task_name, task_config)

    st_seed = 2000 * (1 + seed)

    expert_check = True
    TASK_ENV.suc = 0
    TASK_ENV.test_num = 0
    now_id = 0
    succ_seed = 0
    suc_test_seed_list = []
    now_seed = st_seed
    clear_cache_freq = args["clear_cache_freq"]
    args['policy_name'] = 'V4'

    args["eval_mode"] = True
    args["render_freq"] = 0
    args['ckpt_setting'] = '60k'
    while succ_seed < test_num:
        render_freq = args["render_freq"]
        print('Running test', now_id)
        if expert_check:
            try:
                TASK_ENV.setup_demo(now_ep_num=now_id, seed=now_seed, is_test=True, **args)
                episode_info = TASK_ENV.play_once()
                TASK_ENV.close_env()
            except Exception as e:
                # print(" -------------")
                print("Error: ", e)
                # print(" -------------")
                TASK_ENV.close_env()
                now_seed += 1
                args["render_freq"] = render_freq
                continue

        if (not expert_check) or (TASK_ENV.plan_success and TASK_ENV.check_success()):
            succ_seed += 1
            suc_test_seed_list.append(now_seed)
        else:
            now_seed += 1
            args["render_freq"] = render_freq
            continue

        args["render_freq"] = render_freq

        TASK_ENV.setup_demo(now_ep_num=now_id, seed=now_seed, is_test=True, **args)
        instruction = args["task_name"].replace('_', ' ') # use task name as instruction
        print('instruction:', instruction)
        policy.set_instruction(instruction=instruction)  # set language instruction
        
        try:
            status, images = _rollout(TASK_ENV, policy)
        except Exception as e:  
            TASK_ENV.close_env()
            now_seed += 1
            args["render_freq"] = render_freq
            continue
        save_path = f'{eval_log_dir}/{task_name}/{now_id}_{status}.mp4'
        save_video(save_path, images)
        # log per episode results
        metrics = {f'sim/{task_name}': status}
        save_path = f'{eval_log_dir}/results.json'
        _log_results(metrics, save_path)
        now_id += 1
        TASK_ENV.close_env(clear_cache=((succ_seed + 1) % clear_cache_freq == 0))
        if TASK_ENV.render_freq:
            TASK_ENV.viewer.close()

        TASK_ENV.test_num += 1

        print(f"Success rate: {round(TASK_ENV.suc/TASK_ENV.test_num*100, 1)}%")
        now_seed += 1
        try:
            point = round(TASK_ENV.suc/TASK_ENV.test_num*100, 1)
            test_done = True
        except Exception as e:
            test_done = False
            print('redo tests due to:', e)
    metrics = {f'sim/{task_name}': point}
    _log_results(metrics, f'{eval_log_dir}/perc.json')

    return now_seed, TASK_ENV.suc


def save_video(output_path, frames, fps=30):
    print('saving video to ', output_path)
    imageio.mimsave(output_path, frames, fps=fps) 


def _log_results(metrics, log_path):
    with open(log_path, 'a+') as f:
        line = json.dumps(metrics)
        f.write(line+'\n')


def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained model on multistep sequences with language goals.")
    parser.add_argument("--host", default='0.0.0.0', help="Your client host ip")
    parser.add_argument("--port", default='8001', help="Your client port")
    parser.add_argument("--eval_log_dir", default='/home/dodo/fyc/HeteroDiffusionPolicy/AbsEEFFlowV4/runnings/RoboTwin/', type=str, help="Where to log the evaluation results.")
    parser.add_argument("--device", default=0, type=int, help="CUDA device")
    parser.add_argument("--num_episodes", default=1000, type=int)
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--task_name", type=str, required=True, help="Name of the task (envs.<task>)")
    parser.add_argument("--task_config", type=str, required=True, help="Task config name (without .yml)")
    parser.add_argument("--output_path", type=str, required=True, help="Where to save the output video")
    parser.add_argument("--instruction_type", type=str, required=False, help="Where to save the output video")
    parser.add_argument("--quat_scalar_first", action="store_true",
                        default=_env_flag("XVLA_ROBOTWIN_QUAT_SCALAR_FIRST", False),
                        help="Treat RoboTwin quaternions as wxyz and convert to/from SciPy's xyzw. "
                             "Off by default for compatibility with the released X-VLA-RoboTwin2 checkpoint; "
                             "enable for checkpoints trained with XVLA_ROBOTWIN_QUAT_SCALAR_FIRST=1.")
    args = parser.parse_args()
    kwargs = vars(args)
    
    model = ClientModel(host=kwargs['host'], port=kwargs['port'], quat_scalar_first=args.quat_scalar_first)
    if args.task_name == 'all':
        for task in tqdm(ALL_TASKS):
            print(f"Evaluating task {task} for {kwargs['num_episodes']} episodes...")
            rewards = eval_episodes(task_name=task, 
                                    task_config=args.task_config,
                                    policy=model, 
                                    seed=args.seed,
                                    test_num=kwargs['num_episodes'],
                                    eval_log_dir=kwargs['eval_log_dir'],
                                    instruction_type=args.instruction_type)
    else:
        print(f"Evaluating task {args.task_name} for {kwargs['num_episodes']} episodes...")
        rewards = eval_episodes(task_name=args.task_name, 
                                task_config=args.task_config,
                                policy=model, 
                                seed=args.seed,
                                test_num=kwargs['num_episodes'],
                                eval_log_dir=kwargs['eval_log_dir'],
                                instruction_type=args.instruction_type)
if __name__ == "__main__":
    main()
