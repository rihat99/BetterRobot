# W2-H — adversarial review of the BR wave-1 diff (mars_aux, read-only; saved by the orchestrator)

Paths relative to /home/rikhat.akizhanov/better/dev/BetterRobot.

| # | Finding | file:line | Sev | Evidence | Fix |
|---|---|---|---|---|---|
| 1 | `to_euler` gradient NaN at the exact lock in float64 (float32 finite); docstring claims finite (C-DIFF-3); lock test checks no gradient | lie/so3.py:254 (mask :248); tests/lie/test_euler_matrix_interop.py:101 | HIGH | `q = so3.from_euler([[0.3, ±pi/2, 0.7]], f64).requires_grad_(); to_euler(q).sum().backward()` → nan. `asin(clamp(1.0))` has infinite derivative; `torch.where` makes 0·inf = nan | `torch.asin(torch.where(lock, zero, sin_pitch).clamp(-1, 1))`; assert finite gradient in the lock test |
| 2 | float32 lock band (`1 - 4·eps` on sin_pitch) ≈ 1e-3 rad: output snapped (yaw 0), round trip worse than before | so3.py:248 | MED | pitch = π/2 − 5e-4: error 4.8e-4 new vs 3.5e-5 old; at 1e-5, 1e-7 new is better; f64 band ≈ 4e-8 rad | choose the band from the angle (or a much tighter sine band), add a near-lock float32 test |
| 3 | C-NAME-5 test scans functions only — not methods, properties, nor `better_robot.viewer` | tests/contract/test_se3_vocabulary.py | MED | injected param `frame_tf` caught; method `pose_matrix` / property `world_placement` not caught | walk `inspect.getmembers` of exported classes; add viewer |
| 4 | Conflicts real: `JointModel.joint_transform` returns a local 7-vector (public extension point); `_fk_lean.placement_matrices` private | joint_models/base.py:65, joint_dispatch.py:34; _fk_lean.py:204 | MED | 9 definitions of `joint_transform` | owner decision: rename (`joint_placement`/`joint_motion`) or explicit carve-out in 01-naming.md |
| 5 | MJCF damping/frictionloss/armature wiring never runs here (mujoco absent → skip), incl. `<default>` class inheritance and ball-joint per-dof shape | io/parsers/mjcf.py (`_scalar_attr`); tests/io/test_joint_dynamics.py:32 | MED | importorskip | run once with mujoco (main checkout venv, read-only) |
| 6 | Viewer `times` persists on the layer; `add(layer)` without `times` keeps an old value; set before `_add` can fail; NaN/inf pass validation (`_span` OverflowError) | viewer/viewer.py:97-103, :291 | LOW | code read | set `layer.times` inside `_add` after success; require `np.isfinite` |
| 7 | Warm start with wrong batch gives raw RuntimeError; passing a `Result` gives AttributeError | optim/phases.py:80 | LOW | probe B=1 state into B=4 | check `warm.done.shape`, raise ShapeError; reject non-State |
| 8 | `q_permutation` raises ValueError for unknown names, not `UnknownNameError` | data_model/model_structure.py:232-234 | LOW | code read | raise UnknownNameError |
| 9 | `Trajectory` has no consumer; `at(step)` integer while Viewer resolves by time; no monotonic/dtype check | data_model/trajectory.py:37 | LOW | grep | keep minimal (owner), add monotonic `times` check; note for roadmap |
| 10 | `self_collision_pairs(ignore=)` resolves ids via `frame_id`: body ids silently treated as frame ids | collision/robot_spheres.py:82 | LOW | code read | docstring: frame ids/names, not body ids |

Confirmed: `Viewer._index` (searchsorted right−1, 1e-9 s tol, clamps to 0; `_span` = floor(t_last·fps)+1); LM warm
start carries `damping`/`growth`, warm 29 it vs cold capped at 50; diagnostics `(steps, B)` for LM/GN/LBFGS, TorchOptim
empty dict; `is_compiling` guards do not change eager and the cache holds (1.0 µs vs 182 µs); `to_euler` bit-identical
away from the lock (200k quats, f32+f64); `build_model(parse(urdf)) == load(urdf)` for Panda, friction units agree,
absent attrs → 0; `UnknownNameError` subclasses KeyError and all `*_id` lookups use it; no positional `frame=` callers left.
