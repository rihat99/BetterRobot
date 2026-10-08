# W3 — BR fixes for w2_aux_findings items 1-3, 5-10 (item 4 untouched: owner decision)

Worktree /home/rikhat.akizhanov/better/dev/BetterRobot. Not committed.

| # | Change | Test |
|---|---|---|
| 1 | `so3.to_euler`: `pitch = asin(where(lock, 0, sin_pitch).clamp(-1, 1))` | lock test now backprops; grad finite f32+f64, both signs |
| 2 | Lock band set in angle: `lock = hypot(roll_sin, roll_cos) < sqrt(eps)` (cos(pitch) estimate; ~3.5e-4 rad f32, ~1.5e-8 rad f64) instead of `|sin_pitch| >= 1 - 4 eps` (~1e-3 rad f32) | new `test_to_euler_near_gimbal_lock_float32_keeps_regular_round_trip`: [0.3, pi/2-5e-4, 0.7] f32 error 1.5e-5 <= 3.5e-5; exact-lock test unchanged and passes; bitwise off-lock test passes |
| 3 | `test_se3_vocabulary.py`: adds `better_robot.viewer`; walks `vars(klass)` of every better_robot class in each exported class's MRO (methods, properties, attributes + their parameter names) via helper `_member_names` | still 0 violations (2438 names scanned, 522 from viewer); injected `Model.pose_matrix(frame_tf)` and property `world_placement` are caught |
| 5 | Ran with main-checkout venv (mujoco 3.8.0, read-only, PYTHONPATH -> worktree src, confirmed import path): 4 passed | new `test_mjcf_default_class_inheritance_and_ball_joint_per_dof`: `<default><joint damping=0.3 armature=0.05/>`, hinge + ball(frictionloss 0.4) -> per-dof damping [0.3]*4, friction [0,0.4,0.4,0.4], armature [0.05]*4 |
| 6 | `Viewer.add`: rejects non-finite times; `layer.times` assigned inside `_add` after the duplicate-name check, always (None included) | `finite` case added; new `test_times_are_set_per_add_and_not_on_a_refused_add` |
| 7 | `optim.run`: `TypeError` if `state` is not a `State`; `ValueError` if `warm.done.shape != state.done.shape` | new `test_warm_start_rejects_wrong_batch_and_non_state` |
| 8 | `q_permutation` unknown names -> `UnknownNameError` | existing parametrized test now expects `UnknownNameError` for the "unknown" case (was ValueError) |
| 9 | `Trajectory.__post_init__`: `ValueError("times must be strictly increasing")` when T > 1 | new parametrized test (equal and decreasing times) |
| 10 | `self_collision_pairs` docstring: frame names or frame ids; body ids not accepted (an int is read as a frame id) | doc only |

## Probe (scratchpad w3/probe.py; max round-trip error over 20k random roll/yaw)
float32: d=5e-4 new 4.8e-4 = old regular formula (band not engaged); d=3e-4 2.1e-4 vs 6.0e-4; d=1e-5 7.1e-6 vs 1.4e-2;
d=0 1.2e-7 vs 0.71. float64: identical to regular formula down to d=1e-7; d=1e-9 7e-10 vs 2.6e-7; d=0 3e-16.
A pure sine band cannot work in f32: from_euler rounding puts the exact lock up to ~8 ulps below 1, which is
~1e-3 rad, so no sine threshold both catches the exact lock and spares 5e-4 rad (probed k = 4, 1, 0.5 eps).

## Deviations / notes
- Item 7 raises plain `ValueError`, not `ShapeError`: optim imports only torch (layer contract). Comment says so.
- Item 8 changes the exception type of a public method (KeyError subclass, no longer ValueError).
- The aux "old 3.5e-5" value is reproduced only for specific angles; the worst case at 5e-4 rad over random
  roll/yaw is 4.8e-4 for both the regular formula and the new code (the band is not engaged there).

## Checks
- `uv run ruff check src tests`: Found 80 errors (was 82; none in changed lines).
- `uv run mkdocs build --strict`: built, no warnings.
- Full suite: `1650 passed, 13 skipped, 20 deselected, 33 warnings in 261.74s (0:04:21)` (run before the
  test-only `_member_names` refactor; `tests/contract` rerun after it: 708 passed, 8 skipped).
- Main-venv mujoco run of tests/io/test_joint_dynamics.py: `4 passed in 1.54s`.
