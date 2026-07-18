# quadruped-rl: learning to walk from scratch with PPO

Training a simulated Unitree Go1 quadruped to walk from scratch with
reinforcement learning (PPO) in MuJoCo — no reference motions, no
demonstrations, just a shaped reward and trial and error.

**Status: work in progress.**

Planned phases:

1. ✅ Environment setup — MuJoCo + Unitree Go1 model (from
   [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie),
   BSD-3-Clause, vendored under `env/assets/unitree_go1/`), actuation and
   rendering sanity checks
2. PPO training pipeline (Stable-Baselines3) + short validation run
3. Reward shaping + full training run
4. Robustness evaluation (push recovery, randomized episodes)
5. Visualization: training-progression video, interactive push-recovery demo
6. Results, analysis, and final writeup

## Setup

```
python -m venv .venv
.venv/Scripts/pip install torch --index-url https://download.pytorch.org/whl/cpu
.venv/Scripts/pip install mujoco "gymnasium[mujoco]" stable-baselines3 tensorboard imageio
```

## Phase 1 sanity check

```
.venv/Scripts/python env/sanity_check.py
```

Renders three rollouts to `results/phase1/`: the robot collapsing with
actuation disabled (gravity check), holding its standing pose (actuator
check), and flailing under random position targets (action-space check).
