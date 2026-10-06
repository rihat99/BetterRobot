# Context: adversarial review of the collision layer and clearance terms

Repo: `/home/rikhat.akizhanov/better/dev/BetterRobot` (git worktree, branch `dev`, commit 4503509). Read-only task:
you do not edit files, commit, or push. Run Python with `CUDA_VISIBLE_DEVICES="" uv run python ...` from the repo
root; scratch scripts go to `/tmp/claude-1003/-home-rikhat-akizhanov-better/d9220856-907e-4a64-b1a1-f6cd72ec6355/scratchpad/review_collision/`
(create it). No new dependencies.

## What to review (commit d79e570, `git show --stat d79e570`; the brief the author worked from is
`context_collision.md` in this folder)
- `src/better_robot/collision/distance.py`: `sphere_sphere`, `sphere_capsule`, `sphere_halfspace`, `sphere_box`,
  `sphere_sdf` (signed distances, negative = penetration, batched, differentiable).
- `src/better_robot/collision/robot_spheres.py`: `RobotSpheres.from_frames`, `.centers(state)`,
  `self_collision_pairs`, `self_distances`.
- `src/better_robot/terms/collision.py`: `sphere_clearance`, `self_clearance` term factories for `optim2`
  (`src/better_robot/optim2/CLAUDE.md` explains terms; `src/better_robot/terms/robot.py` is the reference style).
- `tests/collision/test_collision.py`, `examples/optim2/06_obstacle_trajectory.py`.
- `KinematicState`: `src/better_robot/kinematics/state.py` (`frame_transform (B..., nframes, 3, 4)`).

## Claims to attack
1. Each distance function is correct, including inside/penetrating cases, the degenerate capsule (zero length), the
   box corners/edges/faces and inside-the-box (signed distance to the nearest face), the SDF lookup at and beyond the
   grid border (clamping), and sphere radius handling. Write brute-force or closed-form checks of your own with random
   inputs; do not trust the existing tests.
2. Gradients are finite and correct (gradcheck in float64) away from and at contact; what happens exactly at
   distance 0, at a box corner, at the capsule axis (sphere centre on the segment)? NaN anywhere?
3. `RobotSpheres.centers(state)` is correct for batched states `(B, T, ...)` and the sphere offsets are in the frame
   (not the joint) — check against `frame_pose_of` / manual FK.
4. `self_collision_pairs` excludes pairs on the same and adjacent frames as documented; is "adjacent" defined by
   joint parent/child or by frame? Does it miss pairs it should include (grand-parent links that can collide) or
   include impossible pairs?
5. Terms: shapes `(B..., S * n_obstacles)`, `window=1` use in trajectory problems, `check_jacobian` passes with
   autodiff, the hinge is zero when clear; `solve_constrained` inequality form is `margin - d` (plain). Run
   `examples/optim2/06_obstacle_trajectory.py` (CPU) and confirm the printed numbers in
   `../REPORT.md` section M5 (initial penetration 9.2 cm, soft +1.6 cm, hard +2.0 cm).
6. Design: dead code, unused stubs left, names inconsistent with `docs/conventions/naming.md`, layer rule
   (`collision` must not import `optim2` or `terms`; see `tests/contract/test_layer_dependencies.py`), anything the
   brief asked for that is missing or anything added beyond it.

## Report (final message, ≤ 50 lines)
Ranked findings: severity (major/minor/nit), file:line, what is wrong, how you showed it (command + numbers), proposed
fix. Then one line per claim above: confirmed / refuted / not checked.
