"""Roll out a trained checkpoint and report metrics; optionally save a GIF.

Usage (from repo root):
    .venv/Scripts/python evaluation/rollout.py \
        --checkpoint checkpoints/phase2_validation/final.zip \
        --vecnormalize checkpoints/phase2_validation/final_vecnormalize.pkl \
        --episodes 5 --gif results/phase2/validation_policy.gif
"""

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import imageio.v2 as imageio
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from env.go1_env import Go1WalkEnv


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--vecnormalize", required=True)
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--gif", type=str, default=None,
                        help="save a GIF of the first episode to this path")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    render = args.gif is not None
    raw_env = Go1WalkEnv(render_mode="rgb_array" if render else None)
    venv = DummyVecEnv([lambda: raw_env])
    venv = VecNormalize.load(args.vecnormalize, venv)
    venv.training = False      # freeze normalization stats
    venv.norm_reward = False   # report raw rewards

    model = PPO.load(args.checkpoint)

    ep_rewards, ep_lengths, ep_speeds, falls = [], [], [], 0
    frames = []
    for ep in range(args.episodes):
        obs = venv.reset()
        total_r, length, vxs = 0.0, 0, []
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, r, dones, infos = venv.step(action)
            done = bool(dones[0])
            total_r += float(r[0])
            length += 1
            vxs.append(infos[0]["vx_world"])
            if render and ep == 0 and length % 2 == 0:  # 25 fps
                frames.append(raw_env.render())
            if done and infos[0].get("TimeLimit.truncated", False) is False and length < 1000:
                falls += 1
        ep_rewards.append(total_r)
        ep_lengths.append(length)
        ep_speeds.append(float(np.mean(vxs)))
        print(f"episode {ep}: reward={total_r:8.1f} length={length:4d} "
              f"mean vx={np.mean(vxs):+.2f} m/s")

    print(f"\nover {args.episodes} episodes: "
          f"reward {np.mean(ep_rewards):.1f} +/- {np.std(ep_rewards):.1f} | "
          f"length {np.mean(ep_lengths):.0f} | "
          f"mean forward speed {np.mean(ep_speeds):+.2f} m/s | "
          f"falls {falls}/{args.episodes}")

    if render and frames:
        out = Path(args.gif)
        out.parent.mkdir(parents=True, exist_ok=True)
        imageio.mimsave(out, frames, fps=25, loop=0)
        print(f"saved {out}")


if __name__ == "__main__":
    main()
