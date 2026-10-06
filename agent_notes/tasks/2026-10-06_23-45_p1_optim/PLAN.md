# PLAN — P1 new optim (BetterRobot)

Task id `2026-10-06_23-45_p1_optim`, branch `dev`. Owner is asleep and said: work without asking.
Design: `/home/rikhat.akizhanov/better/dev/BetterRobot/agent_notes/tasks/2026-10-06_20-13_architecture_plan/DESIGN.md` section D3 + section 6;
API detail: `.../2026-10-06_20-13_architecture_plan/subagents/study_optim_design.md` section 3-4.
P0 results: `/home/rikhat.akizhanov/better/dev/BetterRobot/agent_notes/tasks/2026-10-06_21-40_p0_measure_and_fix/REPORT.md`.

## Objective
A new optimization package beside the old one: terms are plain functions of named tensors, the
same terms run under torch.optim, Gauss-Newton and LM. Old `optim`, `residuals`, `tasks` stay
untouched until the owner has seen the new one (deleting is the owner's call).

## Package layout (temporary names; renamed to `optim` when the old one is deleted)
- `src/better_robot/optim2/` — knows nothing about robots, imports only torch (+ lie later):
  `manifold.py`, `kernels.py`, `problem.py`, `linear.py`, `solvers.py`, `phases.py`.
- `src/better_robot/terms/` — robot terms as plain functions with optional analytic `jac`,
  `RobotManifold`; reads `KinematicState`.

## Milestones and checks
M1 core + IK (this night)
 - Problem / cached / residual / scalar / loss / linearize (dense) / retract / bounds / free masks.
 - Solvers: LM, GaussNewton, TorchOptim adapter; Phase + run.
 - Robot terms: frame pose, frame position, joint limits, rest; RobotManifold.
 - Checks: (a) autodiff and analytic Jacobians agree with finite differences (`check_jacobian`);
   (b) several terms sharing one cached FK: FK runs once per evaluation, Jacobian not zero;
   (c) Panda IK batch 1024 float32 on the stored reference problems
   (`.../2026-10-06_21-40_p0_measure_and_fix/golden/ik_panda.npz`): every reachable target within
   1e-4 m and 1e-3 rad, or an honest count of those that are not, compared with the old solver;
   (d) the same Problem solved by torch.optim.Adam and by LM.
M2 temporal: stencil terms, block-banded normal equations, trajectory optimization of the Panda
   example vs the stored reference; batched L-BFGS.
M3 `solve_layer` (implicit / unroll); augmented-Lagrangian recipe; four use-case examples.
M4 port BVR person fit (read-only repo: port as an example inside BetterRobot notes, compare losses).
M5 adversarial review by a clean-context mars_aux; REPORT.md; commit.

## Rules
- No new dependencies. Tests only where they pin tricky invariants (Jacobians, cache, masks).
- Commit on `dev` after each milestone. Never push.
- Carried over from P0: GPU baselines when a GPU is FREE; BVR clip reference.
