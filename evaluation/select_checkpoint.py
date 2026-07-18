"""Evaluate several checkpoints head-to-head and report a selection table.

PPO training is noisy late in a run, so the last checkpoint is not
necessarily the best. This evaluates each candidate over N deterministic
episodes (different seeds per episode) and ranks by mean reward, reporting
falls and mean forward speed alongside.

Usage (from repo root):
    .venv/Scripts/python evaluation/select_checkpoint.py \
        --run-dir checkpoints/phase3_full \
        --steps 8000000 8500000 9000000 9500000 10000000 \
        --episodes 10
"""

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from env.go1_env import EPISODE_LENGTH, Go1WalkEnv


def evaluate(checkpoint: Path, vecnorm: Path, episodes: int) -> dict:
    venv = DummyVecEnv([Go1WalkEnv])
    venv = VecNormalize.load(str(vecnorm), venv)
    venv.training = False
    venv.norm_reward = False
    model = PPO.load(str(checkpoint))

    rewards, lengths, speeds, falls = [], [], [], 0
    for ep in range(episodes):
        venv.seed(1000 + ep)
        obs = venv.reset()
        total_r, length, vxs, done = 0.0, 0, [], False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, r, dones, infos = venv.step(action)
            done = bool(dones[0])
            total_r += float(r[0])
            length += 1
            vxs.append(infos[0]["vx_world"])
        if length < EPISODE_LENGTH:
            falls += 1
        rewards.append(total_r)
        lengths.append(length)
        speeds.append(float(np.mean(vxs)))
    venv.close()
    return {
        "reward_mean": float(np.mean(rewards)),
        "reward_std": float(np.std(rewards)),
        "length_mean": float(np.mean(lengths)),
        "speed_mean": float(np.mean(speeds)),
        "falls": falls,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--steps", type=int, nargs="+", required=True)
    parser.add_argument("--episodes", type=int, default=10)
    args = parser.parse_args()

    run_dir = REPO_ROOT / args.run_dir
    print(f"{'checkpoint':>12s} {'reward':>16s} {'ep_len':>7s} {'speed':>9s} {'falls':>6s}")
    results = {}
    for steps in args.steps:
        ckpt = run_dir / f"ppo_go1_{steps}_steps.zip"
        vecnorm = run_dir / f"ppo_go1_vecnormalize_{steps}_steps.pkl"
        r = evaluate(ckpt, vecnorm, args.episodes)
        results[steps] = r
        print(
            f"{steps:>12,d} {r['reward_mean']:>9.1f} ± {r['reward_std']:<5.0f}"
            f"{r['length_mean']:>7.0f} {r['speed_mean']:>7.2f} m/s"
            f"{r['falls']:>4d}/{args.episodes}"
        )

    best = max(results, key=lambda s: results[s]["reward_mean"])
    print(f"\nbest checkpoint: {best:,} steps "
          f"(reward {results[best]['reward_mean']:.1f}, "
          f"{results[best]['falls']}/{args.episodes} falls)")


if __name__ == "__main__":
    main()
