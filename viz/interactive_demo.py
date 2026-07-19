"""Live interactive rollout: watch the policy walk, shove it with keys.

Opens the MuJoCo viewer with the trained policy running in real time.

Keys (with the viewer window focused):
    Up / Down    push from behind / from the front (100 N, 0.2 s)
    Left / Right push from the right / from the left (80 N, 0.2 s)
    Space        big random-direction push (140 N)
    R            reset the robot
    Esc          quit

Usage (from repo root):
    .venv/Scripts/python viz/interactive_demo.py \
        --checkpoint results/checkpoints/best_8500k.zip \
        --vecnormalize results/checkpoints/best_8500k_vecnormalize.pkl
"""

import argparse
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import glfw
import mujoco
import mujoco.viewer
import numpy as np

from env.go1_env import CTRL_DT, Go1WalkEnv
from viz.policy_runner import Policy

LATERAL_N = 80.0
SAGITTAL_N = 100.0
RANDOM_N = 140.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--vecnormalize", required=True)
    parser.add_argument("--selftest", action="store_true",
                        help="run 3 s headless (no window) and exit")
    args = parser.parse_args()

    policy = Policy(args.checkpoint, args.vecnormalize)
    env = Go1WalkEnv()
    obs, _ = env.reset(seed=0)
    state = {"obs": obs, "resets": 0, "pushes": 0}
    rng = np.random.default_rng()

    def control_step() -> None:
        action = policy(state["obs"])
        state["obs"], _, term, trunc, _ = env.step(action)
        if term or trunc:
            state["obs"], _ = env.reset()
            if term:
                state["resets"] += 1
                print(f"fell (resets: {state['resets']})")

    if args.selftest:
        for _ in range(150):
            control_step()
            if _ == 50:
                env.queue_push((0.0, LATERAL_N), 0.2)
        print(f"selftest ok: 3 s simulated, falls: {state['resets']}")
        return

    def key_callback(keycode: int) -> None:
        pushes = {
            glfw.KEY_UP: (SAGITTAL_N, 0.0, "100 N from behind"),
            glfw.KEY_DOWN: (-SAGITTAL_N, 0.0, "100 N from the front"),
            glfw.KEY_RIGHT: (0.0, -LATERAL_N, "80 N from the left"),
            glfw.KEY_LEFT: (0.0, LATERAL_N, "80 N from the right"),
        }
        if keycode in pushes:
            fx, fy, name = pushes[keycode]
            env.queue_push((fx, fy), 0.2)
            state["pushes"] += 1
            print(f"push {state['pushes']}: {name}")
        elif keycode == glfw.KEY_SPACE:
            angle = rng.uniform(0, 2 * np.pi)
            env.queue_push(
                (RANDOM_N * np.cos(angle), RANDOM_N * np.sin(angle)), 0.2
            )
            state["pushes"] += 1
            print(f"push {state['pushes']}: {RANDOM_N:.0f} N random direction")
        elif keycode in (glfw.KEY_R,):
            state["obs"], _ = env.reset()
            print("manual reset")

    print(__doc__)
    with mujoco.viewer.launch_passive(
        env.model, env.data, key_callback=key_callback
    ) as viewer:
        while viewer.is_running():
            t0 = time.perf_counter()
            control_step()
            viewer.sync()
            # hold real-time rate
            leftover = CTRL_DT - (time.perf_counter() - t0)
            if leftover > 0:
                time.sleep(leftover)


if __name__ == "__main__":
    main()
