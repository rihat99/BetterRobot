# Context: minimal real collision for BetterRobot + obstacle-avoiding trajectory example

Repo: `/home/rikhat.akizhanov/better/dev/BetterRobot` (git worktree, branch `dev`). Do not `cd` elsewhere.

## Background you need
- `src/better_robot/collision/` today is stubs that raise `NotImplementedError` (read all five files, 242 lines).
- New optimization layer: `src/better_robot/optim2/` (read its `CLAUDE.md`, then `problem.py` docstrings) and robot
  terms in `src/better_robot/terms/robot.py` (read fully: this is the style to match — a factory returns a
  `TermFn(fn, reads, jac=None, window=None)`; kinematic terms read a cached `KinematicState` named `"state"`).
- `KinematicState` (`src/better_robot/kinematics/state.py`): `frame_transform (B..., nframes, 3, 4)` as `[R | p]`,
  accessors `frame_rotation`, `frame_translation`; `frame_jacobian(model, state, frame)` -> `(B..., 6, nv)`
  (rows linear then angular, local-world-aligned: linear rows = velocity of the frame origin in world axes).
- Trajectory problems: `examples/optim2/02_trajectory.py` (Panda, time axis, `window=`), hard constraints:
  `examples/optim2/05_hard_constraints.py` and `optim2.solve_constrained`.

## Goal
Replace the stubs by the smallest collision layer that makes robot trajectory optimization with obstacles work.
Functional, batched, differentiable, torch only. No meshes, no broad phase, no GJK.

## Scope
1. `src/better_robot/collision/`: plain functions returning signed distance (negative = penetration), all with
   leading batch axes and documented shapes:
   - sphere–sphere, sphere–capsule (segment endpoints + radius), sphere–half-space (plane normal + offset),
     sphere–box (centre, rotation, half extents) only if it stays under ~25 lines, else leave it out;
   - sphere–SDF: trilinear lookup in a regular grid `(X, Y, Z)` with origin and voxel size (clamp outside the grid).
   - A robot is a set of spheres attached to frames: a small frozen dataclass (name it `RobotSpheres`) with
     `frame_ids (S,)`, `offsets (S, 3)` in the frame, `radii (S,)`, and a method
     `centers(state) -> (B..., S, 3)`; plus a constructor helper that places spheres along chosen links
     (e.g. from a dict `{frame_name: [(offset, radius), ...]}`). Self-collision: a function taking an index
     tensor of sphere pairs `(P, 2)` and returning `(B..., P)` distances; a helper that builds the pair list
     excluding pairs on the same or adjacent frames.
   Keep whatever of the existing public names still makes sense; delete stub names you do not implement (there
   is no backward-compatibility requirement), and update `__init__.py`.
2. `src/better_robot/terms/collision.py`: term factories for optim2
   - `sphere_clearance(robot_spheres, distance_fn, *, margin, state="state")`: residual
     `clamp(margin - d, min=0)` per sphere (hinge, so it is zero when clear), shape `(B..., S * n_obstacles)`;
     usable with `window=1` in trajectory problems (output `(B, T, m)`);
   - a self-collision variant;
   - an analytic `jac` is optional: add it only for sphere-vs-static-obstacle if it is simple (gradient of the
     distance w.r.t. the sphere centre times the linear Jacobian of that centre, which for an offset `o` in the
     frame is `J_lin - hat(R o) @ J_ang`); otherwise rely on autodiff and say so. Whatever you add must pass
     `optim2.check_jacobian`.
   Export from `src/better_robot/terms/__init__.py`.
3. `tests/collision/test_collision.py` (useful tests only): each distance function against a closed-form case
   and against a brute-force/numeric check on random inputs (include penetration and the degenerate capsule of
   zero length); gradients finite at contact; SDF lookup against an analytic sphere SDF sampled on a grid;
   `RobotSpheres.centers` against manual FK; `check_jacobian` for the term(s).
4. `examples/optim2/06_obstacle_trajectory.py`: Panda moves between two joint configurations whose straight
   joint-space path passes through a sphere obstacle (choose configurations and obstacle so that the straight path
   really collides — verify and print the penetration of the initial path). Time axis, smoothness
   (`terms.finite_difference`), fixed endpoints, clearance term with `window=1`; solve with LM as a soft penalty and
   also with `optim2.solve_constrained(..., inequalities=...)` if that works (the inequality form is
   `margin - d <= 0`, registered as a plain residual `margin - d`, not the hinge). Print: minimum clearance before
   and after, path length, solve time. Run it on CPU with `torch.set_num_threads(4)`.

## Rules
- Layer contract: `tests/contract/test_layer_dependencies.py` ranks packages (`collision`: 5, `terms`: 6,
  `optim2`: 3, `kinematics` lower than collision). `collision` must not import `optim2` or `terms`.
  The contract test that inventories `NotImplementedError` raises (find it in `tests/contract/`) must be updated to
  match the stubs you removed; do not add new `NotImplementedError`. Do not edit other contract tests.
- Do not edit: `src/better_robot/kinematics/**`, `src/better_robot/dynamics/**`, `src/better_robot/optim2/**`,
  `src/better_robot/terms/robot.py`, `src/better_robot/terms/ik.py`, `docs/**`, `pyproject.toml`, `agent_notes/**`
  (except: you may write notes into `agent_notes/tasks/2026-10-07_00-39_p2_model_kin_dyn/subagents/collision_notes.md`).
  If you find a bug or a missing feature in those, report it instead of fixing it.
- Never delete with `rm` (use `git rm` or the `trash` helper). Do not commit or push.
- Type hints and docstrings with shapes on public functions; ruff line length 120; `uv run ruff check` clean on
  the files you touched. No new dependencies. CPU only: prefix commands with `CUDA_VISIBLE_DEVICES=""`.
- Simplicity first: aim for about 250 lines in `collision/` and 80 in `terms/collision.py`.

## Checks before you report
- `CUDA_VISIBLE_DEVICES="" uv run pytest tests/collision tests/contract tests/optim2 -q` passes.
- The example runs and prints a positive minimum clearance after optimization.

## Report (final message, short)
Public API (names + shapes), numbers printed by the example, test summary line, design choices you made, anything
that did not work (be specific), files touched.
