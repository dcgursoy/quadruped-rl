"""Gymnasium environment for Unitree Go1 locomotion in MuJoCo.

Design choices (standard for learned quadruped locomotion):

- Actions (12-dim, [-1, 1]): target joint-angle *offsets* around the standing
  "home" pose, scaled by ACTION_SCALE and tracked by the model's built-in
  position actuators (kp=100). Position control around a nominal stance is far
  more trainable than raw torques.
- Control at 50 Hz (10 physics substeps at dt=0.002).
- Observations (45-dim, proprioception only -- nothing a real robot couldn't
  measure): projected gravity (3), trunk linear velocity in body frame (3),
  trunk angular velocity in body frame (3), joint positions relative to home
  (12), joint velocities (12), previous action (12). Absolute world position
  and height are deliberately excluded.
- Episode ends early if the trunk falls below MIN_HEIGHT or tilts past
  ~55 degrees; otherwise truncates at EPISODE_LENGTH control steps (20 s).

The reward is a weighted sum of components defined in REWARD_WEIGHTS so that
reward-shaping iterations (Phase 3) are explicit and diffable. Per-step
component values are exposed in info["reward_components"].

`queue_push(force_xy, duration_s)` applies an external force to the trunk
during subsequent steps -- used for robustness evaluation and the interactive
push-recovery demo.
"""

from pathlib import Path

import gymnasium as gym
import mujoco
import numpy as np

SCENE_XML = Path(__file__).parent / "assets" / "unitree_go1" / "scene.xml"

PHYSICS_STEPS_PER_CTRL = 10  # 0.002 s * 10 = 50 Hz control
CTRL_DT = 0.02               # s per control step
EPISODE_LENGTH = 1000        # 20 s
ACTION_SCALE = 0.3           # rad, offset range around home pose
TARGET_HEIGHT = 0.27         # m, nominal standing trunk height
MIN_HEIGHT = 0.18            # m, trunk height below this = fallen
MAX_TILT_GZ = -0.6           # body-frame gravity z above this = tipped over
AIR_TIME_TARGET = 0.2        # s, swing shorter than this is penalized

# Reward v2 (Phase 3). Weights are per-control-step.
# v1 (Phase 2 baseline: forward_velocity, alive, orientation, torque,
# action_rate only) produced a stable low crawl -- see
# docs/reward_shaping.md for the full iteration log and reasoning.
REWARD_WEIGHTS = {
    "forward_velocity": 2.0,   # * clip(world v_x, -1.0, +1.0) m/s
    "alive": 0.5,              # constant while not fallen
    "height": -30.0,           # * (trunk z - TARGET_HEIGHT)^2, anti-crawl
    "orientation": -2.0,       # * (g_x^2 + g_y^2), penalize tilt
    "lateral_velocity": -1.0,  # * v_y^2 (body frame), walk straight
    "yaw_rate": -0.5,          # * w_z^2 (body frame), walk straight
    "abduction_posture": -0.3, # * sum(abduction q^2), anti-leg-splay
    "air_time": 2.0,           # * sum_feet (t_air - AIR_TIME_TARGET) at
                               #   touchdown: rewards real swings, punishes
                               #   foot-dragging / vibrating contacts
    "torque": -1e-4,           # * sum(actuator torque^2), energy proxy
    "action_rate": -0.01,      # * ||a_t - a_{t-1}||^2, smoothness
}

FOOT_GEOM_NAMES = ("FR", "FL", "RR", "RL")
ABDUCTION_QPOS_IDX = np.array([7, 10, 13, 16])  # hip abduction joints in qpos


class Go1WalkEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 50}

    def __init__(self, render_mode: str | None = None, seed: int | None = None):
        self.model = mujoco.MjModel.from_xml_path(str(SCENE_XML))
        self.data = mujoco.MjData(self.model)
        self.render_mode = render_mode
        self._renderer = None

        self._home_qpos = self.model.key_qpos[0].copy()
        self._home_ctrl = self.model.key_ctrl[0].copy()
        self._ctrl_lo = self.model.actuator_ctrlrange[:, 0].copy()
        self._ctrl_hi = self.model.actuator_ctrlrange[:, 1].copy()
        self._trunk_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "trunk"
        )
        self._floor_geom = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_GEOM, "floor"
        )
        self._foot_geoms = np.array(
            [
                mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, n)
                for n in FOOT_GEOM_NAMES
            ]
        )
        self._foot_air = np.zeros(4)

        self.action_space = gym.spaces.Box(-1.0, 1.0, shape=(12,), dtype=np.float32)
        self.observation_space = gym.spaces.Box(
            -np.inf, np.inf, shape=(45,), dtype=np.float64
        )

        self._rng = np.random.default_rng(seed)
        self._prev_action = np.zeros(12)
        self._step_count = 0
        self._push_force = np.zeros(3)
        self._push_steps_left = 0

    # ------------------------------------------------------------- helpers

    def _body_frame(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Projected gravity, linear velocity, angular velocity in body frame."""
        quat = self.data.qpos[3:7]
        neg_quat = np.empty(4)
        mujoco.mju_negQuat(neg_quat, quat)

        gravity = np.empty(3)
        mujoco.mju_rotVecQuat(gravity, np.array([0.0, 0.0, -1.0]), neg_quat)

        lin_vel = np.empty(3)
        mujoco.mju_rotVecQuat(lin_vel, self.data.qvel[0:3], neg_quat)

        ang_vel = self.data.qvel[3:6].copy()  # free joint: already body-frame
        return gravity, lin_vel, ang_vel

    def _get_obs(self) -> np.ndarray:
        gravity, lin_vel, ang_vel = self._body_frame()
        joint_pos = self.data.qpos[7:] - self._home_qpos[7:]
        joint_vel = self.data.qvel[6:]
        return np.concatenate(
            [gravity, lin_vel, ang_vel, joint_pos, joint_vel, self._prev_action]
        )

    def _is_fallen(self) -> bool:
        gravity, _, _ = self._body_frame()
        return self.data.qpos[2] < MIN_HEIGHT or gravity[2] > MAX_TILT_GZ

    def _foot_contacts(self) -> np.ndarray:
        """Bool[4]: is each foot geom in contact with the floor."""
        contacts = np.zeros(4, dtype=bool)
        for i in range(self.data.ncon):
            c = self.data.contact[i]
            pair = (c.geom1, c.geom2)
            if self._floor_geom in pair:
                other = pair[1] if pair[0] == self._floor_geom else pair[0]
                hit = np.flatnonzero(self._foot_geoms == other)
                if hit.size:
                    contacts[hit[0]] = True
        return contacts

    def _air_time_reward(self) -> float:
        """Accumulate swing time; score each swing when the foot touches down."""
        contacts = self._foot_contacts()
        total = 0.0
        for i in range(4):
            if contacts[i]:
                if self._foot_air[i] > 0.0:  # touchdown ends a swing
                    total += self._foot_air[i] - AIR_TIME_TARGET
                self._foot_air[i] = 0.0
            else:
                self._foot_air[i] += CTRL_DT
        return total

    # ------------------------------------------------------------- gym API

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        mujoco.mj_resetDataKeyframe(self.model, self.data, 0)
        self.data.qpos[7:] += self._rng.uniform(-0.1, 0.1, size=12)
        self.data.qvel[:] = self._rng.uniform(-0.1, 0.1, size=self.model.nv)
        mujoco.mj_forward(self.model, self.data)

        self._prev_action = np.zeros(12)
        self._step_count = 0
        self._push_force = np.zeros(3)
        self._push_steps_left = 0
        self._foot_air = np.zeros(4)
        return self._get_obs(), {}

    def step(self, action: np.ndarray):
        action = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
        self.data.ctrl[:] = np.clip(
            self._home_ctrl + ACTION_SCALE * action, self._ctrl_lo, self._ctrl_hi
        )

        for _ in range(PHYSICS_STEPS_PER_CTRL):
            if self._push_steps_left > 0:
                self.data.xfrc_applied[self._trunk_id, :3] = self._push_force
                self._push_steps_left -= 1
                if self._push_steps_left == 0:
                    self.data.xfrc_applied[self._trunk_id, :3] = 0.0
            mujoco.mj_step(self.model, self.data)

        self._step_count += 1
        fallen = self._is_fallen()

        gravity, lin_vel, ang_vel = self._body_frame()
        vx_world = self.data.qvel[0]
        torques = self.data.actuator_force
        components = {
            "forward_velocity": REWARD_WEIGHTS["forward_velocity"]
            * float(np.clip(vx_world, -1.0, 1.0)),
            "alive": REWARD_WEIGHTS["alive"] * (0.0 if fallen else 1.0),
            "height": REWARD_WEIGHTS["height"]
            * float((self.data.qpos[2] - TARGET_HEIGHT) ** 2),
            "orientation": REWARD_WEIGHTS["orientation"]
            * float(gravity[0] ** 2 + gravity[1] ** 2),
            "lateral_velocity": REWARD_WEIGHTS["lateral_velocity"]
            * float(lin_vel[1] ** 2),
            "yaw_rate": REWARD_WEIGHTS["yaw_rate"] * float(ang_vel[2] ** 2),
            "abduction_posture": REWARD_WEIGHTS["abduction_posture"]
            * float(np.sum(self.data.qpos[ABDUCTION_QPOS_IDX] ** 2)),
            "air_time": REWARD_WEIGHTS["air_time"] * self._air_time_reward(),
            "torque": REWARD_WEIGHTS["torque"] * float(np.sum(torques**2)),
            "action_rate": REWARD_WEIGHTS["action_rate"]
            * float(np.sum((action - self._prev_action) ** 2)),
        }
        reward = float(sum(components.values()))
        self._prev_action = action

        terminated = fallen
        truncated = self._step_count >= EPISODE_LENGTH
        info = {
            "reward_components": components,
            "vx_world": float(vx_world),
            "trunk_height": float(self.data.qpos[2]),
        }
        return self._get_obs(), reward, terminated, truncated, info

    # ---------------------------------------------------------- utilities

    def queue_push(self, force_xy: tuple[float, float], duration_s: float = 0.2):
        """Apply a horizontal force (N) to the trunk for the given duration."""
        self._push_force = np.array([force_xy[0], force_xy[1], 0.0])
        self._push_steps_left = max(1, round(duration_s / self.model.opt.timestep))

    def render(self):
        if self._renderer is None:
            self._renderer = mujoco.Renderer(self.model, height=480, width=640)
        self._renderer.update_scene(self.data, camera="tracking")
        return self._renderer.render()

    def close(self):
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None


if __name__ == "__main__":
    # Smoke test: API check + timed random rollout
    import time

    from gymnasium.utils.env_checker import check_env

    check_env(Go1WalkEnv(), skip_render_check=True)
    print("gymnasium check_env: OK")

    env = Go1WalkEnv(seed=0)
    obs, _ = env.reset(seed=0)
    print(f"obs shape: {obs.shape}, action shape: {env.action_space.shape}")

    n, ep_len, ep_lens = 5000, 0, []
    t0 = time.perf_counter()
    for _ in range(n):
        obs, r, term, trunc, info = env.step(env.action_space.sample())
        ep_len += 1
        if term or trunc:
            ep_lens.append(ep_len)
            ep_len = 0
            obs, _ = env.reset()
    dt = time.perf_counter() - t0
    print(f"{n} env steps in {dt:.1f} s = {n / dt:,.0f} steps/s single env")
    print(f"random-policy episode lengths: {ep_lens[:10]} (should fall fast)")
