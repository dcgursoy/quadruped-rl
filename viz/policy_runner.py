"""Shared helpers for demo/visualization scripts.

Loads an SB3 policy + its VecNormalize statistics and exposes a plain
function interface over a raw (unwrapped) Go1WalkEnv, so viz scripts can
drive the env/data directly (frame capture, viewer integration, scripted
pushes) without VecEnv plumbing.
"""

import pickle
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import numpy as np
from PIL import Image, ImageDraw
from stable_baselines3 import PPO


class Policy:
    """Trained policy + observation normalization, callable on raw obs.

    stochastic=True samples from the action distribution instead of taking
    its mean -- use for untrained/early checkpoints, where the deterministic
    mean is ~0 (= "hold the home pose") and the *sampled* exploration noise
    is what training rollouts actually looked like.
    """

    def __init__(
        self,
        checkpoint: str | Path,
        vecnormalize: str | Path,
        stochastic: bool = False,
    ):
        self.model = PPO.load(str(checkpoint), device="cpu")
        self.stochastic = stochastic
        with open(vecnormalize, "rb") as f:
            vn = pickle.load(f)
        self._mean = vn.obs_rms.mean
        self._var = vn.obs_rms.var
        self._eps = vn.epsilon
        self._clip = vn.clip_obs

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        norm = (obs - self._mean) / np.sqrt(self._var + self._eps)
        norm = np.clip(norm, -self._clip, self._clip)
        action, _ = self.model.predict(norm, deterministic=not self.stochastic)
        return action


def label_frame(frame: np.ndarray, text: str, sub: str = "") -> np.ndarray:
    """Draw a label (with dark backing bar) onto a rendered frame."""
    img = Image.fromarray(frame)
    draw = ImageDraw.Draw(img, "RGBA")
    draw.rectangle([0, 0, img.width, 46 if sub else 28], fill=(0, 0, 0, 160))
    draw.text((8, 6), text, fill=(255, 255, 255, 255))
    if sub:
        draw.text((8, 24), sub, fill=(200, 200, 200, 255))
    return np.asarray(img)
