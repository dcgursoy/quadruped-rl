"""Side-by-side training-progression demo: early vs mid vs final policy.

Runs each checkpoint in its own environment from the same seed, renders
them in parallel panels (auto-resetting on falls, so the early policy's
flailing stays on screen), and writes a labeled GIF.

Usage (from repo root):
    .venv/Scripts/python viz/progression.py \
        --run-dir results/checkpoints \
        --stages early_500k "0.5M steps" mid_2M "2M steps" best_8500k "8.5M steps (final)" \
        --out results/phase5/progression.gif
"""

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import imageio.v2 as imageio
import numpy as np

from env.go1_env import Go1WalkEnv
from viz.policy_runner import Policy, label_frame

FPS = 25
STEPS_PER_FRAME = 2  # 50 Hz control -> 25 fps


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True,
                        help="directory containing <stage>.zip + <stage>_vecnormalize.pkl")
    parser.add_argument("--stages", nargs="+", required=True,
                        help="pairs: <checkpoint-stem> <label> ...")
    parser.add_argument("--seconds", type=float, default=8.0)
    parser.add_argument("--seed", type=int, default=3)
    parser.add_argument("--scale", type=float, default=0.5,
                        help="output resolution scale per panel")
    parser.add_argument("--stochastic-stages", nargs="*", default=[],
                        help="checkpoint stems rolled out with sampled (not "
                             "mean) actions -- shows exploration behavior "
                             "for untrained/early checkpoints")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    if len(args.stages) % 2 != 0:
        sys.exit("--stages must be <stem> <label> pairs")
    stages = [(args.stages[i], args.stages[i + 1])
              for i in range(0, len(args.stages), 2)]

    run_dir = REPO_ROOT / args.run_dir
    panels = []
    for stem, label in stages:
        policy = Policy(run_dir / f"{stem}.zip",
                        run_dir / f"{stem}_vecnormalize.pkl",
                        stochastic=stem in args.stochastic_stages)
        env = Go1WalkEnv(render_mode="rgb_array")
        obs, _ = env.reset(seed=args.seed)
        panels.append({"policy": policy, "env": env, "obs": obs,
                       "label": label, "falls": 0})

    n_ctrl_steps = int(args.seconds * 50)
    frames = []
    for step in range(n_ctrl_steps):
        for p in panels:
            action = p["policy"](p["obs"])
            p["obs"], _, term, trunc, _ = p["env"].step(action)
            if term or trunc:
                if term:
                    p["falls"] += 1
                p["obs"], _ = p["env"].reset()
        if step % STEPS_PER_FRAME == 0:
            row = []
            for p in panels:
                f = p["env"].render()
                f = label_frame(f, p["label"], f"falls: {p['falls']}")
                if args.scale != 1.0:
                    k = int(1 / args.scale)
                    f = f[::k, ::k]
                row.append(f)
            frames.append(np.concatenate(row, axis=1))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(out, frames, fps=FPS, loop=0)
    print(f"saved {out} ({len(frames)} frames, "
          f"{frames[0].shape[1]}x{frames[0].shape[0]})")
    for p in panels:
        print(f"  {p['label']}: {p['falls']} falls in {args.seconds:.0f} s")


if __name__ == "__main__":
    main()
