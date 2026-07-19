"""Scripted push-recovery demo GIF: pushes from several directions mid-walk.

The robot walks; at scripted times a labeled horizontal force hits the
trunk. The label bar shows each push as it happens.

Usage (from repo root):
    .venv/Scripts/python viz/push_demo.py \
        --checkpoint results/checkpoints/best_8500k.zip \
        --vecnormalize results/checkpoints/best_8500k_vecnormalize.pkl \
        --out results/phase5/push_recovery.gif
"""

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import imageio.v2 as imageio

from env.go1_env import Go1WalkEnv
from viz.policy_runner import Policy, label_frame

FPS = 25
STEPS_PER_FRAME = 2

# (control step, force N, direction name, (fx, fy)); 0.2 s each
PUSHES = [
    (150, 80.0, "80 N push from the right", (0.0, 1.0)),
    (300, 80.0, "80 N push from the left", (0.0, -1.0)),
    (450, 100.0, "100 N push from the front", (-1.0, 0.0)),
    (600, 100.0, "100 N push from behind", (1.0, 0.0)),
]
LABEL_HOLD_STEPS = 75  # keep each push label up 1.5 s


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--vecnormalize", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--seconds", type=float, default=16.0)
    args = parser.parse_args()

    policy = Policy(args.checkpoint, args.vecnormalize)
    env = Go1WalkEnv(render_mode="rgb_array")
    obs, _ = env.reset(seed=args.seed)

    frames, falls, label, label_until = [], 0, "walking...", 0
    for step in range(int(args.seconds * 50)):
        for t, force, name, (fx, fy) in PUSHES:
            if step == t:
                env.queue_push((fx * force, fy * force), 0.2)
                label, label_until = name, step + LABEL_HOLD_STEPS
        action = policy(obs)
        obs, _, term, trunc, _ = env.step(action)
        if term or trunc:
            if term:
                falls += 1
                label, label_until = "fell -- resetting", step + LABEL_HOLD_STEPS
            obs, _ = env.reset()
        if step % STEPS_PER_FRAME == 0:
            text = label if step <= label_until else "walking..."
            frames.append(label_frame(env.render(), text, f"falls: {falls}")[::2, ::2])

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(out, frames, fps=FPS, loop=0)
    print(f"saved {out} ({len(frames)} frames), falls: {falls}")


if __name__ == "__main__":
    main()
