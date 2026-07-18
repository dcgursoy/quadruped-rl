"""Phase 1 sanity check for the Unitree Go1 MuJoCo model.

Loads the MJCF scene, prints model info, and runs three short rollouts:

1. passive   -- actuation disabled: the robot collapses under gravity
2. home_hold -- actuators hold the 'home' keyframe pose: the robot stands
3. random    -- uniform-random position targets each control step: flailing

Each rollout is rendered to a GIF in results/phase1/ plus a final-frame PNG.
This validates the physics, the actuators, and offscreen rendering before
any RL code is written.
"""

from pathlib import Path

import imageio.v2 as imageio
import mujoco
import numpy as np

SCENE_XML = Path(__file__).parent / "assets" / "unitree_go1" / "scene.xml"
OUT_DIR = Path(__file__).parent.parent / "results" / "phase1"

RENDER_W, RENDER_H = 640, 480
FPS = 30


def print_model_info(model: mujoco.MjModel) -> None:
    print(f"nq (position dims) : {model.nq}")
    print(f"nv (velocity dims) : {model.nv}")
    print(f"nu (actuators)     : {model.nu}")
    print(f"timestep           : {model.opt.timestep} s")
    print(f"total mass         : {sum(model.body_mass):.3f} kg")
    print("actuators (ctrlrange, forcerange):")
    for i in range(model.nu):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
        lo, hi = model.actuator_ctrlrange[i]
        flo, fhi = model.actuator_forcerange[i]
        print(f"  {name:10s} ctrl [{lo:+.3f}, {hi:+.3f}]  force [{flo:+.1f}, {fhi:+.1f}] N*m")


def rollout(
    model: mujoco.MjModel,
    renderer: mujoco.Renderer,
    ctrl_fn,
    seconds: float,
    disable_actuation: bool = False,
    ctrl_hz: float = 50.0,
) -> tuple[list[np.ndarray], dict]:
    """Simulate from the 'home' keyframe, returning rendered frames and stats."""
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)  # 'home' standing pose

    saved_flags = model.opt.disableflags
    if disable_actuation:
        model.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_ACTUATION

    steps_per_frame = max(1, round(1.0 / (FPS * model.opt.timestep)))
    steps_per_ctrl = max(1, round(1.0 / (ctrl_hz * model.opt.timestep)))
    n_steps = round(seconds / model.opt.timestep)

    frames = []
    heights = []
    for step in range(n_steps):
        if step % steps_per_ctrl == 0:
            data.ctrl[:] = ctrl_fn(data)
        mujoco.mj_step(model, data)
        heights.append(data.qpos[2])
        if step % steps_per_frame == 0:
            renderer.update_scene(data, camera="tracking")
            frames.append(renderer.render())

    model.opt.disableflags = saved_flags
    stats = {
        "final_height_m": float(data.qpos[2]),
        "min_height_m": float(np.min(heights)),
        "max_height_m": float(np.max(heights)),
        "sim_seconds": float(data.time),
    }
    return frames, stats


def save(frames: list[np.ndarray], name: str) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(OUT_DIR / f"{name}.gif", frames, fps=FPS, loop=0)
    imageio.imwrite(OUT_DIR / f"{name}_last_frame.png", frames[-1])


def main() -> None:
    model = mujoco.MjModel.from_xml_path(str(SCENE_XML))
    print_model_info(model)
    renderer = mujoco.Renderer(model, height=RENDER_H, width=RENDER_W)

    home_ctrl = model.key_ctrl[0].copy()
    ctrl_lo = model.actuator_ctrlrange[:, 0]
    ctrl_hi = model.actuator_ctrlrange[:, 1]
    rng = np.random.default_rng(0)

    scenarios = {
        "passive": dict(ctrl_fn=lambda d: 0.0, seconds=2.0, disable_actuation=True),
        "home_hold": dict(ctrl_fn=lambda d: home_ctrl, seconds=3.0),
        "random": dict(ctrl_fn=lambda d: rng.uniform(ctrl_lo, ctrl_hi), seconds=3.0),
    }

    import time

    for name, kwargs in scenarios.items():
        t0 = time.perf_counter()
        frames, stats = rollout(model, renderer, **kwargs)
        wall = time.perf_counter() - t0
        save(frames, name)
        rtf = stats["sim_seconds"] / wall
        print(
            f"[{name:9s}] height final={stats['final_height_m']:.3f} m "
            f"min={stats['min_height_m']:.3f} max={stats['max_height_m']:.3f} | "
            f"{stats['sim_seconds']:.1f} s sim in {wall:.1f} s wall ({rtf:.0f}x realtime, incl. rendering)"
        )

    # raw physics throughput (no rendering) -- informs training-time estimates
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    data.ctrl[:] = home_ctrl
    n = 10_000
    t0 = time.perf_counter()
    for _ in range(n):
        mujoco.mj_step(model, data)
    dt = time.perf_counter() - t0
    print(f"raw physics: {n / dt:,.0f} steps/s single-threaded (no rendering)")

    print(f"outputs written to {OUT_DIR}")


if __name__ == "__main__":
    main()
