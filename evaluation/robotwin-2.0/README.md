# 🧪 Evaluation on RoboTwin-2.0

We evaluate **X-VLA** on the **RoboTwin-2.0** benchmark to assess its ability to handle bimanual tabletop manipulation with multiple object sets, shifting layouts, and varied visual randomness.

---

## 1️⃣ Environment Setup

Follow the official instructions from the original RoboTwin-2.0 repository:  
👉 [https://robotwin-platform.github.io/doc/usage/index.html](https://robotwin-platform.github.io/doc/usage/index.html)

No additional modifications are required for X-VLA evaluation.

---

## 2️⃣ Start the X-VLA Server

Run the X-VLA model as an inference server (in a clean environment to avoid dependency conflicts):

```bash
conda activate X-VLA
python -m deploy --model_path 2toINF/X-VLA-RoboTwin2
```
---

## 3️⃣ Run the Client Evaluation
Add the absolute path of your RoboTwin repository at line 4 of `X-VLA/evaluation/robotwin-2.0/client.py`:

```
robowin_root = Path("/home/dodo/fyc/RoboTwin") # <- Add your path
```

Launch the RoboTwin-2.0 evaluation client to connect to your X-VLA server:

```bash
cd evaluation/robotwin-2.0
bash eval_robotwin.sh
```
You can configure custome evaluation in `eval_robotwin.sh`, such as log directry, server port number, number of episodes evaluated, task config, etc.

The client will stream observations (images, proprioception, and language) to the X-VLA model, receive predicted actions, and execute them within the RoboTwin-2.0 environment.

### Quaternion order (`--quat_scalar_first`)

RoboTwin-2.0 reports and consumes end-effector quaternions **scalar-first** (`wxyz`, transforms3d convention), whereas SciPy's `Rotation` defaults to **scalar-last** (`xyzw`). Historically both the dataset handler (`RobotWin2Handler`) and this client passed the four numbers to SciPy unchanged. Because the same convention is used on the observation and the action side, this cancels out end-to-end and the released `2toINF/X-VLA-RoboTwin2` checkpoint works as-is, but the internal rotation-6D does not correspond to the physical end-effector rotation (see [issue #93](https://github.com/2toinf/X-VLA/issues/93)).

| Checkpoint | Training (`RobotWin2Handler`) | Evaluation (`client.py`) |
| :-- | :-- | :-- |
| Released `X-VLA-RoboTwin2` / legacy (default) | nothing to set | nothing to set |
| New runs with correct `wxyz` handling | `export XVLA_ROBOTWIN_QUAT_SCALAR_FIRST=1` before `train.py` | `--quat_scalar_first` (or the same env var) |

The two sides must use the same setting.

---

## 📊 Results (Using RoboTwin-2.0 Leaderboard Settings)

|    **Settings**   |   Easy  |   Hard  |
| :--------------------: | :--: | :--: |
|     **Success (%)**    | 70.0 | 39.0 |



