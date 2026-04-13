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

DATA_WEIGHTS = {
    "robomind-franka": 0.1,
    "robomind-ur": 0.1,
    "Droid-Left": 0.15,
    "Droid-Right": 0.15,
    "AGIBOT": 0.4,
    "robomind-agilex": 0.07,
    "robomind-franka-dual": 0.03,
    
    
    # agibot world challenge
    "agiworld-on-site-pack": 0.8,
    "agiworld-on-site-pack-extra": 0.2,
    "agiworld-on-site-conveyor": 0.8,
    "agiworld-on-site-conveyor-extra": 0.2,
    "agiworld-on-site-restock": 1.,
    "agiworld-on-site-pour": 1.,
    "agiworld-on-site-microwave": 1.2,
    "agiworld-on-site-cloth": 1.2,
    "agiworld-on-site-cloth-2": 0.1,

    # lerobot real-world
    "arx_acone": 1.0,
}

DATA_DOMAIN_ID = {
    # ft
    # "Bridge": 0,
    # "RT1": 1,
    # "Calvin": 2,
    # "libero": 3,
    # "widowx-air": 4,
    # "AIR-AGILEX-HQ": 5,
    # "robotwin2_abs_ee": 6,
    # "robotwin2_clean": 6,
    # "robocasa-human": 7,
    # "VLABench": 8,
    # "AGIBOT-challenge": 9,
    # "AIR-AGILEX": 10,
    # "AIRBOT": 18,
    
    # # pretraining
    # "robomind-franka": 11,
    # "robomind-ur": 12,
    # "Droid-Left": 13,
    # "Droid-Right": 14,
    # "AGIBOT": 15,
    # "robomind-agilex": 16,
    # "robomind-franka-dual": 17,
    
    # # agibot world challenge
    # "agiworld-on-site-pack": 0, # 20,
    # "agiworld-on-site-pack-extra": 0, # 20,
    # "agiworld-on-site-conveyor": 0, # 21,
    # "agiworld-on-site-conveyor-extra": 0, #26,
    # "agiworld-on-site-restock": 0, #22,
    # "agiworld-on-site-pour": 0, # 23,
    # "agiworld-on-site-microwave": 0, #24,
    # "agiworld-on-site-cloth": 0, #25,
    # "agiworld-on-site-cloth-2": 0, #27,
    
    
    # lerobot-sim
    "lift2": 0,
    "arx_acone": 5,

    # x2robot
    "x2robot": 0,

    # starvla
    
    "Bridge": 0,
    "RT1": 1,
    "libero": 2,
    "aloha": 3, # robotwin-2.0-lerobot
    "GR1ArmsAndWaistFourierHands": 4,
}
