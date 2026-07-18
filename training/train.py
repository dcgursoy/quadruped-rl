"""PPO training for Go1 locomotion (Stable-Baselines3).

Usage (from repo root):
    .venv/Scripts/python training/train.py --run-name phase2_validation --steps 500000

Checkpoints (policy + VecNormalize stats) go to checkpoints/<run-name>/,
TensorBoard logs to runs/<run-name>/.
"""

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize

from env.go1_env import Go1WalkEnv


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--steps", type=int, default=500_000)
    parser.add_argument("--n-envs", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--checkpoint-every", type=int, default=100_000,
                        help="total env steps between checkpoints")
    parser.add_argument("--resume", type=str, default=None,
                        help="path to a .zip checkpoint to continue training from")
    args = parser.parse_args()

    ckpt_dir = REPO_ROOT / "checkpoints" / args.run_name
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    vec_env = make_vec_env(
        Go1WalkEnv,
        n_envs=args.n_envs,
        seed=args.seed,
        vec_env_cls=SubprocVecEnv,
    )
    vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True, clip_obs=10.0)

    if args.resume:
        model = PPO.load(args.resume, env=vec_env, tensorboard_log=str(REPO_ROOT / "runs"))
    else:
        model = PPO(
            "MlpPolicy",
            vec_env,
            learning_rate=3e-4,
            n_steps=512,          # per env -> 4096-step rollouts with 8 envs
            batch_size=1024,
            n_epochs=5,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.0,
            policy_kwargs=dict(net_arch=[256, 256], activation_fn=nn.ELU),
            seed=args.seed,
            verbose=1,
            tensorboard_log=str(REPO_ROOT / "runs"),
        )

    checkpoint_cb = CheckpointCallback(
        save_freq=max(1, args.checkpoint_every // args.n_envs),
        save_path=str(ckpt_dir),
        name_prefix="ppo_go1",
        save_vecnormalize=True,
    )

    model.learn(
        total_timesteps=args.steps,
        callback=checkpoint_cb,
        tb_log_name=args.run_name,
        reset_num_timesteps=not args.resume,
        progress_bar=False,
    )

    model.save(ckpt_dir / "final")
    vec_env.save(str(ckpt_dir / "final_vecnormalize.pkl"))
    print(f"saved final policy + normalization stats to {ckpt_dir}")


if __name__ == "__main__":
    main()
