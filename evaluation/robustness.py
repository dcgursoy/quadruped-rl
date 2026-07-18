"""Phase 4 robustness evaluation of a trained policy.

Two experiments:

1. Push recovery: mid-walk, apply a horizontal force to the trunk for 0.2 s
   (direction x magnitude sweep, randomized timing, N trials each). Success =
   the robot is still up 5 s after the push. Impulse context: the Go1 is
   12.7 kg, so a 100 N x 0.2 s push is a ~1.6 m/s instant velocity change.

2. Sensor noise: Gaussian noise added to every observation (IMU-and-encoder
   scaled sigmas), falls and speed over N full episodes.

Writes a markdown summary + CSV to results/phase4/.

Usage (from repo root):
    .venv/Scripts/python evaluation/robustness.py \
        --checkpoint results/checkpoints/best_8500k.zip \
        --vecnormalize results/checkpoints/best_8500k_vecnormalize.pkl
"""

import argparse
import csv
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from env.go1_env import EPISODE_LENGTH, Go1WalkEnv

PUSH_DIRECTIONS = {
    "left": (0.0, 1.0),
    "right": (0.0, -1.0),
    "backward": (-1.0, 0.0),
    "forward": (1.0, 0.0),
}
PUSH_MAGNITUDES_N = [25, 50, 75, 100, 125, 150]
PUSH_DURATION_S = 0.2
RECOVERY_WINDOW_STEPS = 250  # 5 s at 50 Hz

# Per-block observation noise sigmas (gravity, lin vel, ang vel, joint pos,
# joint vel, prev action) -- roughly realistic IMU / joint-encoder noise.
OBS_NOISE_SIGMA = np.concatenate([
    np.full(3, 0.02),    # projected gravity
    np.full(3, 0.05),    # lin vel, m/s
    np.full(3, 0.10),    # ang vel, rad/s
    np.full(12, 0.005),  # joint pos, rad
    np.full(12, 0.75),   # joint vel, rad/s
    np.full(12, 0.0),    # prev action (internal, not a sensor)
])


class NoisyObs(gym.ObservationWrapper):
    def __init__(self, env, scale: float, seed: int = 0):
        super().__init__(env)
        self.scale = scale
        self.rng = np.random.default_rng(seed)

    def observation(self, obs):
        return obs + self.rng.normal(0.0, OBS_NOISE_SIGMA * self.scale)


def load(checkpoint: str, vecnormalize: str, env_fn):
    venv = DummyVecEnv([env_fn])
    venv = VecNormalize.load(vecnormalize, venv)
    venv.training = False
    venv.norm_reward = False
    return PPO.load(checkpoint), venv


def push_trial(model, venv, direction, magnitude, seed) -> bool:
    """Returns True if the robot survives 5 s past the push.

    The push lands at a random mid-walk step in [250, 450); the loop never
    reaches the 1000-step time limit, so any `done` here is a fall. (A fall
    *before* the push counts as failure too; the unpushed policy never falls
    in evaluation, so this effectively never triggers.)
    """
    rng = np.random.default_rng(seed)
    push_step = int(rng.integers(250, 450))
    venv.seed(seed)
    obs = venv.reset()
    for step in range(push_step + RECOVERY_WINDOW_STEPS):
        action, _ = model.predict(obs, deterministic=True)
        obs, _, dones, _ = venv.step(action)
        if step == push_step:
            fx, fy = direction
            venv.env_method(
                "queue_push", (fx * magnitude, fy * magnitude), PUSH_DURATION_S
            )
        if dones[0]:
            return False
    return True


def run_push_sweep(model, venv, trials: int):
    print(f"\n=== push recovery ({PUSH_DURATION_S} s push, "
          f"success = upright 5 s later, {trials} trials/cell) ===")
    header = "direction " + "".join(f"{m:>8d}N" for m in PUSH_MAGNITUDES_N)
    print(header)
    rows = []
    for di, (dname, dvec) in enumerate(PUSH_DIRECTIONS.items()):
        cells = []
        for mi, mag in enumerate(PUSH_MAGNITUDES_N):
            ok = sum(
                push_trial(model, venv, dvec, mag,
                           seed=7000 + di * 10000 + mi * 1000 + t)
                for t in range(trials)
            )
            cells.append(ok)
            rows.append({"direction": dname, "magnitude_N": mag,
                         "recovered": ok, "trials": trials})
        print(f"{dname:>9s} " + "".join(f"{c:>6d}/{trials}" for c in cells))
    return rows


def run_noise_eval(checkpoint, vecnormalize, scale: float, episodes: int):
    model, venv = load(checkpoint, vecnormalize,
                       lambda: NoisyObs(Go1WalkEnv(), scale))
    falls, speeds = 0, []
    for ep in range(episodes):
        venv.seed(3000 + ep)
        obs = venv.reset()
        vxs, length, done = [], 0, False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, dones, infos = venv.step(action)
            done = bool(dones[0])
            length += 1
            vxs.append(infos[0]["vx_world"])
        if length < EPISODE_LENGTH:
            falls += 1
        speeds.append(float(np.mean(vxs)))
    venv.close()
    return falls, float(np.mean(speeds))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--vecnormalize", required=True)
    parser.add_argument("--push-trials", type=int, default=10)
    parser.add_argument("--noise-episodes", type=int, default=20)
    args = parser.parse_args()

    out_dir = REPO_ROOT / "results" / "phase4"
    out_dir.mkdir(parents=True, exist_ok=True)

    model, venv = load(args.checkpoint, args.vecnormalize, Go1WalkEnv)
    rows = run_push_sweep(model, venv, args.push_trials)
    venv.close()

    with open(out_dir / "push_recovery.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)

    print(f"\n=== sensor-noise eval ({args.noise_episodes} episodes each) ===")
    noise_results = []
    for scale in (0.0, 1.0, 2.0):
        falls, speed = run_noise_eval(
            args.checkpoint, args.vecnormalize, scale, args.noise_episodes
        )
        noise_results.append((scale, falls, speed))
        print(f"noise x{scale:.0f}: {falls}/{args.noise_episodes} falls, "
              f"mean speed {speed:+.2f} m/s")

    with open(out_dir / "robustness_results.md", "w") as f:
        f.write("# Robustness results\n\n")
        f.write(f"Policy: `{args.checkpoint}`\n\n")
        f.write(f"## Push recovery ({PUSH_DURATION_S} s horizontal push to "
                f"trunk mid-walk; success = upright 5 s later; "
                f"{args.push_trials} trials/cell)\n\n")
        f.write("| direction | " + " | ".join(f"{m} N" for m in PUSH_MAGNITUDES_N)
                + " |\n")
        f.write("|---" * (len(PUSH_MAGNITUDES_N) + 1) + "|\n")
        for dname in PUSH_DIRECTIONS:
            cells = [r for r in rows if r["direction"] == dname]
            f.write(f"| {dname} | " + " | ".join(
                f"{r['recovered']}/{r['trials']}" for r in cells) + " |\n")
        f.write("\n## Observation noise (IMU/encoder-scaled Gaussian)\n\n")
        f.write("| noise scale | falls | mean speed |\n|---|---|---|\n")
        for scale, falls, speed in noise_results:
            f.write(f"| x{scale:.0f} | {falls}/{args.noise_episodes} | "
                    f"{speed:.2f} m/s |\n")
    print(f"\nwrote {out_dir / 'robustness_results.md'}")


if __name__ == "__main__":
    main()
