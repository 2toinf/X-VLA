

# 🤖 X-VLA: Soft-Prompted Transformer as a Scalable Cross-Embodiment Vision-Language-Action Model

| 📄 **Paper** | 🌐 **Project Page** | 🤗 **Hugging Face** |
| :---: | :---: | :---: |
| [Read the Full Research](https://arxiv.org/pdf/2510.10274) | [Explore the Demos](https://thu-air-dream.github.io/X-VLA/) | [Access Models & Datasets](https://huggingface.co/collections/2toINF/x-vla) |


## 🏆 Highlights & News

### 🎉 Exciting News: X-VLA Accepted to ICLR 2026  
We are thrilled to announce that **X-VLA has been accepted to ICLR 2026**.

### 🚀 Now Supported in LeRobot  
X-VLA is now natively integrated into the [LeRobot platform](https://huggingface.co/docs/lerobot/xvla).  
Give it a try! We sincerely appreciate the support and collaboration from the Hugging Face team.

### 🥇 Champion Winner at IROS 2025  
X-VLA won **1st Place (Champion)** at the **AgiBot World Challenge**, held at **IROS 2025**.


---

## 🧩 Overview

Successful generalist **Vision–Language–Action (VLA)** models depend on scalable, cross-platform training across diverse robotic embodiments.  
To leverage the heterogeneity of large-scale robot datasets, **X-VLA** introduces a **soft prompt** mechanism — embodiment-specific learnable embeddings that guide a unified Transformer backbone toward effective multi-domain policy learning.

The resulting architecture — **X-VLA-0.9B** — achieves **state-of-the-art generalization** across six simulation platforms and three real-world robots, surpassing prior VLA approaches in dexterity, adaptability, and efficiency.

https://github.com/user-attachments/assets/c047bac4-17c3-4d66-8036-badfab2b8c41

---

## 🚀 Quick Start: Installation & Deployment

### 1️⃣ Installation

```bash
# Clone the repository
git clone https://github.com/2toinf/X-VLA.git
cd X-VLA
```

```bash
# Create and activate Conda environment
conda create -n XVLA python=3.10 -y
conda activate XVLA

# Install dependencies
pip install -r requirements.txt
```

or 

```bash
conda env create -f environment.yml
conda activate xvla-stable
```

---
### 2️⃣ Deploying X-VLA for Inference

X-VLA adopts a **Server–Client** architecture to separate the model environment from simulation or robot-specific dependencies.
This design avoids package conflicts and supports distributed inference across GPUs, SLURM clusters, or edge devices.

#### 🧠 Available Pre-trained Models

- [ ] We observed a slight performance drop (around 1% across different datasets) after converting our models to the HF format, and we’re actively investigating the cause.

#### 🧠 About Libero Setup and Evaluation

- [x] For questions about converting relative actions to absolute actions and our implementation, please first refer to issue [#2](https://github.com/2toinf/X-VLA/issues/2) and [#15](https://github.com/2toinf/X-VLA/issues/15). We have updated full preprocessing guidance [here](https://github.com/2toinf/X-VLA/blob/main/evaluation/libero/preprocess.md).

#### 🔥 Update: We have released the LoRA fine-tuning code, along with checkpoints and the associated inference code.

| Model ID                                                                                           | Embodiment        | Description                                                                                     |   Performance   | Evaluation Guidance |
| :------------------------------------------------------------------------------------------------- | :---------------- | :---------------------------------------------------------------------------------------------- | :--------------: | :-----------------: |
| [`2toINF/X-VLA-Pt`](https://huggingface.co/2toINF/X-VLA-Pt)                                        | Foundation        | Pretrained on large-scale heterogeneous robot–vision–language datasets for general transfer.     | —                | —                   |
| [`2toINF/X-VLA-AgiWorld-Challenge`](https://huggingface.co/2toINF/X-VLA-AgiWorld-Challenge)        | Agibot-G1          | Fine-tuned for AgiWorld Challenge.       | **Champion🥇**        | -  |
| [`2toINF/X-VLA-Calvin-ABC_D`](https://huggingface.co/2toINF/X-VLA-Calvin-ABC_D)                    | Franka     | Fine-tuned on CALVIN benchmark (ABC_D subset)              | **4.43**        | [Calvin Eval](evaluation/calvin/README.md)          |
| [`2toINF/X-VLA-Google-Robot`](https://huggingface.co/2toINF/X-VLA-Google-Robot)                    | Google Robot      |  Fine-tuned on large-scale Google Robot dataset                | **83.5%(VM) 76.4%(VA)**        | [Simpler Eval](evaluation/simpler/README.md)   |
| [`2toINF/X-VLA-Libero`](https://huggingface.co/2toINF/X-VLA-Libero)                                | Franka            | Fine-tuned on LIBERO benchmark                     | **98.1%**        | [LIBERO Eval](evaluation/libero/README.md)         |
| [`2toINF/X-VLA-VLABench`](https://huggingface.co/2toINF/X-VLA-VLABench)                                | Franka            | Fine-tuned on VLABench benchmark                     | **51.1(score)**        | [VLABench Eval](evaluation/vlabench/README.md)        |
| [`2toINF/X-VLA-RoboTwin2`](https://huggingface.co/2toINF/X-VLA-RoboTwin2)                          | Agilex        | Trained on RoboTwin2 dataset for dual-arm coordinated manipulation(50 demos for each task).                     | **70%**        |   [RoboTwin2.0 Eval](evaluation/robotwin-2.0/README.md)    |
| [`2toINF/X-VLA-WidowX`](https://huggingface.co/2toINF/X-VLA-WidowX)                | WidowX  | Fine-tuned on BridgeDataV2 (Simpler benchmark).                                                  | **95.8%**        | [Simpler Eval](evaluation/simpler/README.md) |
| [`2toINF/X-VLA-SoftFold`](https://huggingface.co/2toINF/X-VLA-SoftFold)                            | Agilex          | Fine-tuned on Soft-Fold Dataset. Specialized in deformable object manipulation (e.g., folding and cloth control).                 | cloth folding with a 100% success rate in 2 hours.  |  [SoftFold-Agilex](evaluation/SoftFold-Agilex/readme.md)   |
| LoRA Adapters | ||  | |
| [`2toINF/X-VLA-libero-spatial-peft`](https://huggingface.co/2toINF/X-VLA-libero-spatial-peft)                                | Franka            | Fine-tuned on LIBERO benchmark                     | **96.2%**        | [LIBERO Eval](evaluation/libero/README.md)         |
| [`2toINF/X-VLA-libero-object-peft`](https://huggingface.co/2toINF/X-VLA-libero-object-peft)                                | Franka            | Fine-tuned on LIBERO benchmark                     | **96%**        | [LIBERO Eval](evaluation/libero/README.md)         |
| [`2toINF/X-VLA-libero-goal-peft`](https://huggingface.co/2toINF/X-VLA-libero-goal-peft)                                | Franka            | Fine-tuned on LIBERO benchmark                     | **94.4%**        | [LIBERO Eval](evaluation/libero/README.md)         |
| [`2toINF/X-VLA-libero-long-peft`](https://huggingface.co/2toINF/X-VLA-libero-long-peft)                                | Franka            | Fine-tuned on LIBERO benchmark                     | **83.2%**        | [LIBERO Eval](evaluation/libero/README.md)         |
| [`2toINF/X-VLA-simpler-widowx-peft`](https://huggingface.co/2toINF/X-VLA-simpler-widowx-peft)                | WidowX  | Fine-tuned on BridgeDataV2 (Simpler benchmark).                                                  | **66.7%**        | [Simpler Eval](evaluation/simpler/README.md) |

---

## 🧩 Notes

- All models share a consistent architecture: `configuration_xvla.py`, `modeling_xvla.py`, and unified tokenizer (`tokenizer.json`).
- The **X-VLA-Pt** model is the *foundation checkpoint*, trained across multiple robot domains.
- Each embodiment is fine-tuned for its respective environment while retaining cross-embodiment alignment.
- Evaluation scripts (in `evaluation/`) follow a standardized format for reproducible benchmarking.

---

> 📊 Performance metrics follow standard evaluation protocols detailed in the [paper](https://arxiv.org/pdf/2510.10274).

---

### 3️⃣ Launching the Inference Server

```python
from transformers import AutoModel, AutoProcessor
import json_numpy

# Load model and processor
model = AutoModel.from_pretrained("2toINF/X-VLA-WidowX", trust_remote_code=True)
processor = AutoProcessor.from_pretrained("2toINF/X-VLA-WidowX", trust_remote_code=True)

# Start the inference server
print("🚀 Starting X-VLA inference server...")
model.run(processor, host="0.0.0.0", port=8000)
```

Once launched, the API endpoint is available at:

```
POST http://<server_ip>:8000/act
```

---

### 4️⃣ Client Interaction & Action Prediction

The client communicates via HTTP POST, sending multimodal data (vision + language + proprioception) as a JSON payload.

#### Payload Structure

| Key                    | Type                      | Description                                           |
| :--------------------- | :------------------------ | :---------------------------------------------------- |
| `proprio`              | `json_numpy.dumps(array)` | Current proprioceptive state (e.g., joint positions). |
| `language_instruction` | `str`                     | Task instruction (e.g., "Pick up the red block").     |
| `image0`               | `json_numpy.dumps(array)` | Primary camera image (RGB).                           |
| `image1`, `image2`     | *optional*                | Additional camera views if applicable.                |
| `domain_id`            | `int`                     | Identifier for the current robotic embodiment/domain. |
| `steps`                | `int`                     | denoising steps for flow-matching based generation (e.g., 10).         |

#### Example Client Code

```python
import requests
import numpy as np
import json_numpy

server_url = "http://localhost:8000/act"
timeout = 5

# Prepare inputs
proprio = np.zeros(7, dtype=np.float32)
image = np.zeros((256, 256, 3), dtype=np.uint8)
instruction = "Move the gripper to the target position"

payload = {
    "proprio": json_numpy.dumps(proprio),
    "language_instruction": instruction,
    "image0": json_numpy.dumps(image),
    "domain_id": 0,
    "steps": 10
}

try:
    response = requests.post(server_url, json=payload, timeout=timeout)
    response.raise_for_status()
    result = response.json()
    actions = np.array(result["action"], dtype=np.float32)
    print(f"✅ Received {actions.shape[0]} predicted actions.")
except Exception as e:
    print(f"⚠️ Request failed: {e}")
    actions = np.zeros((30, 20), dtype=np.float32)
```

#### Expected Output

```
[Server] Model loaded successfully on cuda:0
[Server] Listening on 0.0.0.0:8000
[Client] Sending observation to server...
✅ Received 30 predicted actions.
```

---

### 5️⃣ Standardized Control Interface: EE6D

To ensure consistency across embodiments, **X-VLA** adopts a unified **EE6D (End-Effector 6D)** control space.

| Component           | Specification                                                              | Notes                                         |
| :------------------ | :------------------------------------------------------------------------- | :-------------------------------------------- |
| **Proprio Input**   | Current EE6D pose (position + orientation)                                 | Must align with training-space normalization. |
| **Action Output**   | Predicted target delta/absolute pose (EE6D)                                | Executed by downstream controller.            |
| **Dimensionality**  | 20-D vector = 3 (EE Pos) + 6 (Rotation in 6D) + 1 (Gripper) + 10 (Padding) |                                               |
| **Single-arm Case** | If only one arm exists, pad with zeros to maintain 20D vector.             |                                               |

> ⚙️ **Reference Post-processing:**
>
> ```python
> from datasets.utils import rotate6d_to_xyz
> action_final = np.concatenate([
>     action_pred[:3],
>     rotate6d_to_xyz(action_pred[3:9]),
>     np.array([1.0 if action_pred[9] > 0.5 else 0])
> ])
> ```
>
> When feeding proprioception to the model, apply the **inverse transformation** accordingly.

---

### 6️⃣ Reference Client Implementations

Each released model includes a corresponding **reference client** under
[`evaluation/<domain>/<robot>/client.py`](evaluation/) for reproducing exact deployment behaviors.
We strongly recommend adapting from these clients when connecting to physical or simulated robots.

---

### 7️⃣ SLURM & Cluster Deployment

For large-scale or distributed training/deployment (e.g., HPC clusters, AgiBot nodes):

```bash
python -m deploy --model_path /path/to/your/model
```

This script automatically detects SLURM environment variables, launches distributed servers, and writes connection metadata to `info.json`.

---

## ⚙️ Training / Fine-tuning on Custom Data

X-VLA supports fine-tuning on new demonstrations via a modular and extensible dataset interface.

### Data Preparation Workflow

1. **Prepare Meta JSONs** — each domain has a `meta.json` listing trajectory file paths.
2. **Implement Custom Handler** — write a domain loader class with `iter_episode(traj_idx)` generator.
3. **Register Domain** — update:

   * `datasets/domain_handler/registry.py`
   * `datasets/domain_config.py`

### Example Handlers

| Handler       | Dataset               | Description                               |
| :------------ | :-------------------- | :---------------------------------------- |
| `"lerobot"`   | Agibot-Beta           | Optimized for LEROBOT format              |
| `"h5py"`      | RoboMind / Simulation | Efficient loading from `.h5` trajectories |
| `"scattered"` | AGIWorld              | Handles scattered trajectory storage      |

---

### Launch Training with Accelerate

```bash
accelerate launch \
    --mixed_precision bf16 \
    train.py \
    --models '2toINF/X-VLA-Pt' \
    --train_metas_path /path/to/meta_files.json \
    --learning_rate 1e-4 \
    --learning_coef 0.1 \
    --iters 50000 \
    --freeze_steps 1000 \
    --warmup_steps 2000
```

| Argument             | Description                            |
| :------------------- | :------------------------------------- |
| `--models`           | Base model (e.g., `'2toINF/X-VLA-Pt'`) |
| `--train_metas_path` | Path to meta JSON file(s)              |
| `--batch_size`       | Batch size                             |
| `--learning_rate`    | Base LR                                |
| `--learning_coef`    | LR multiplier for soft prompts         |
| `--iters`            | Total training iterations              |
| `--freeze_steps`     | Steps to freeze backbone               |
| `--warmup_steps`     | Warmup iterations                      |

---


## 📚 Citation

If you use X-VLA in your research, please cite:

```bibtex
@article{zheng2025x,
  title   = {X-VLA: Soft-Prompted Transformer as Scalable Cross-Embodiment Vision-Language-Action Model},
  author  = {Zheng, Jinliang and Li, Jianxiong and Wang, Zhihao and Liu, Dongxiu and Kang, Xirui
             and Feng, Yuchun and Zheng, Yinan and Zou, Jiayin and Chen, Yilun and Zeng, Jia and others},
  journal = {arXiv preprint arXiv:2510.10274},
  year    = {2025}
}
```

---

## 🪪 License

This repository is licensed under the **Apache License 2.0**.
You may freely use, modify, and distribute the code under the terms of the license.

```
Copyright 2025 2toINF (https://github.com/2toinf)
Licensed under the Apache License, Version 2.0.
```

---

**Maintained by [2toINF](https://github.com/2toinf)**
💬 Feedback, issues, and contributions are welcome via GitHub Discussions or Pull Requests.
