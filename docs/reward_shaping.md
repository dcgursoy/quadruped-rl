# Reward shaping log

The reward is a weighted sum of per-step components (`REWARD_WEIGHTS` in
`env/go1_env.py`). This file records each iteration: what the policy actually
learned, why that happened, and what changed in response. Reward totals are
not comparable across versions.

## v1 — baseline (Phase 2 validation, 500k steps)

| component | weight | signal |
|---|---|---|
| forward_velocity | 2.0 | clip(world v_x, ±1) |
| alive | 0.5 | 1 while not fallen |
| orientation | −0.5 | g_x² + g_y² (projected gravity) |
| torque | −1e−4 | Σ τ² |
| action_rate | −0.01 | ‖a_t − a_{t−1}‖² |

**Result:** learned quickly (ep_rew −31 → +480), 0/5 falls, 0.44 m/s — but
the gait is a **low crawl**: trunk near the ground, legs splayed sideways,
feet mostly dragging. See `results/phase2/validation_strip.png`.

**Diagnosis:** nothing in v1 prices posture. Crawling is the *easy* local
optimum: a low, wide stance is passively stable (no balance problem to
solve), satisfies `alive`, and still scores forward velocity. The
termination threshold (trunk < 0.15 m) was low enough that a crawl never
terminates. The orientation penalty doesn't fire because a crawling trunk is
still level.

## v2 — anti-crawl posture shaping (Phase 3)

Changes, each targeting a specific failure mode of v1:

| change | targets |
|---|---|
| `height`: −30 · (z − 0.27)² | crawling directly: standing height 0.27 m is worth ~0.4/step over crawling at 0.15 m |
| `MIN_HEIGHT` 0.15 → 0.18 | makes deep crawls *terminal*, not just costly |
| `abduction_posture`: −0.3 · Σ q_abd² | the splayed-leg wide stance (abduction joints pushed to their ±0.86 rad limits) |
| `air_time`: +2.0 · Σ_feet (t_air − 0.2) at touchdown | foot-dragging: swings shorter than 0.2 s *lose* reward, real steps gain; standing still is neutral (no touchdowns) |
| `orientation` −0.5 → −2.0 | pitch/roll excursions once the trunk is up high |
| `lateral_velocity`: −1.0 · v_y², `yaw_rate`: −0.5 · ω_z² | sideways drift and turning (v1 only asked for +x velocity) |

Kept from v1: forward_velocity, alive, torque, action_rate (unchanged).

**Iteration protocol:** each reward variant gets a ~2M-step run (~25 min);
gait judged from rendered rollouts, not just reward totals. Full 10M-step
run only after the gait looks right.

**Result (2M steps, ~33 min):** fixed the crawl in one iteration. The policy
walks upright with a dynamic trot — diagonal leg pairs alternating, trunk at
0.291 m mean (target 0.27), legs under the body. Eval: **1.08 m/s mean
forward speed, 0/5 falls, episode reward 2038 ± 42** (vs the v1 crawl at
0.44 m/s). Component audit over 10 s: forward_velocity +984 and alive +250
dominate; all penalties < 100 combined, i.e. the policy satisfies the
shaping rather than fighting it. `air_time` sits slightly negative (−10)
because swings average just under the 0.2 s target — acceptable; the term's
job was to make dragging unprofitable during learning, and the stride strip
(`results/phase3/v2_stride.png`) shows real swing phases.

**Decision:** freeze this reward for the full 10M-step run (fresh seed, same
env), keeping every 500k-step checkpoint for the training-progression demo.
