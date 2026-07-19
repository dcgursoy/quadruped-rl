"""Training-curve figure from an exported TensorBoard CSV.

Two stacked panels sharing an x-axis (steps): episode reward and episode
length. Raw trace in a light step of the hue, EMA-smoothed trace on top,
selected checkpoint marked and directly labeled.

Usage (from repo root):
    .venv/Scripts/python viz/plot_curve.py \
        --csv results/phase3/training_curve.csv \
        --best-steps 8500000 --out results/phase5/training_curve.png
"""

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# palette: single-series line charts -> one hue (blue), raw = light step
SURFACE = "#fcfcfb"
RAW = "#9ec5f4"
SMOOTH = "#2a78d6"
INK = "#0b0b0b"
SECONDARY = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"


def ema(x: np.ndarray, alpha: float = 0.02) -> np.ndarray:
    out = np.empty_like(x)
    acc = x[0]
    for i, v in enumerate(x):
        acc = (1 - alpha) * acc + alpha * v
        out[i] = acc
    return out


def style(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(BASELINE)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    parser.add_argument("--best-steps", type=int, default=None)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    steps, rew, length = [], [], []
    with open(args.csv) as f:
        for row in csv.DictReader(f):
            steps.append(int(row["timesteps"]))
            rew.append(float(row["ep_rew_mean"]))
            length.append(float(row["ep_len_mean"]) if row["ep_len_mean"] else np.nan)
    steps = np.asarray(steps) / 1e6
    rew, length = np.asarray(rew), np.asarray(length)

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(8, 5.5), sharex=True, height_ratios=[3, 2],
        facecolor=SURFACE,
    )

    ax1.plot(steps, rew, color=RAW, linewidth=1.0)
    ax1.plot(steps, ema(rew), color=SMOOTH, linewidth=2.0)
    ax1.set_title("Episode reward (mean per rollout)", loc="left",
                  fontsize=11, color=INK)
    ax1.text(steps[len(steps) // 2], np.nanmin(rew), "raw", color=RAW,
             fontsize=8, va="bottom")
    if args.best_steps is not None:
        bx = args.best_steps / 1e6
        i = int(np.argmin(np.abs(steps - bx)))
        by = ema(rew)[i]
        ax1.plot([bx], [by], "o", color=SMOOTH, markersize=7,
                 markerfacecolor=SURFACE, markeredgewidth=2)
        ax1.annotate(
            f"selected checkpoint ({args.best_steps / 1e6:.1f}M)",
            (bx, by), textcoords="offset points", xytext=(-8, -16),
            ha="right", fontsize=9, color=SECONDARY,
        )

    ax2.plot(steps, length, color=RAW, linewidth=1.0)
    ax2.plot(steps, ema(length), color=SMOOTH, linewidth=2.0)
    ax2.set_title("Episode length (steps, cap 1000)", loc="left",
                  fontsize=11, color=INK)
    ax2.set_xlabel("training steps (millions)", fontsize=10, color=SECONDARY)

    for ax in (ax1, ax2):
        style(ax)

    fig.tight_layout()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
