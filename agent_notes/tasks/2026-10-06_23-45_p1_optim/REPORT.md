# REPORT — P1 new optim (interim, 2026-10-07)

Status: core built, reviewed, committed on `dev` (code commit `65b63bc`). Not pushed. Old `optim`,
`residuals`, `tasks` packages are untouched; deleting them is the owner's decision.

## What exists

Two new packages, about 1,600 lines together.

- `/home/rikhat.akizhanov/better/dev/BetterRobot/src/better_robot/optim2/` knows nothing about robots.
  - `problem.py`: `Problem` holds named batch-first tensors. A term is a plain function; its argument
    names say which variables, data tensors or cached values it reads. `residual(...)` adds a
    least-squares term, `scalar(...)` a general cost, `cached(...)` a shared intermediate (FK runs once).
    `loss` is differentiable; `linearize` builds Gauss-Newton normal equations. Jacobians come from
    autodiff by default (all terms in one traced pass) or from a term's own `jac`.
    `check_jacobian` compares the two.
  - `linear.py`: dense and block-tridiagonal normal equations with the same interface, Marquardt damping,
    active-set box bounds.
  - `solvers.py`: `LevenbergMarquardt`, `GaussNewton`, batched `LBFGS`, `TorchOptim` (any `torch.optim` class).
  - `phases.py`: `Phase`, `run`, `solve`, `solve_layer` (a solve usable as a differentiable layer).
  - `manifold.py`, `kernels.py` (Huber, Cauchy, Tukey, Geman-McClure).
- `/home/rikhat.akizhanov/better/dev/BetterRobot/src/better_robot/terms/`: robot terms (`frame_pose`,
  `frame_position`, `joint_limits`, `rest`, `finite_difference`), `RobotManifold`, the IK recipe
  (`ik_problem`, `solve_ik` with restarts).
- Time axis: `Problem(time=(...))`; a term with `window=k` reads `k` consecutive knots. When every
  variable has the time axis and every residual term is time-local, the solver is block-tridiagonal
  and cost grows linearly with clip length.
- Tests: `tests/optim2/` (16). Examples: `examples/optim2/01_ik.py`, `02_trajectory.py`,
  `03_clip_fit_phases.py`, `04_ik_layer.py`.

## Numbers

All CPU. "Reached" = position error below 1e-4 m and rotation error below 1e-3 rad.
Problems: the stored Panda reference set `.../2026-10-06_21-40_p0_measure_and_fix/golden/ik_panda.npz`
with noise spread 0.3 rad (easy) and 1.0 rad (hard) around the true solution.

| Check | New | Old |
|---|---|---|
| IK batch 1024, float32, 1 thread, easy | 95.5% reached, 2.5 s | 79%, 3.7 s (regularizers off) |
| same, hard | 72.5% | 66% |
| same with 4 restarts | 98.8% / 98.2%, 12.8 s | not available |
| old solver with its defaults | | 7% / 1% (stops about 0.4 mm away) |
| float64, 8 restarts, 256 problems | 100% | |
| Panda trajectory example | 0.086 s, same result | 0.3 s |
| G1 tracking, 300 frames, 10,500 unknowns, 30 LM steps | 8.8 s, banded equals dense | not run |
| `solve_layer` gradient vs finite differences | agrees to 1e-5 on zero-residual rows | |

Raw files: `subagents/m1_ik_check.json`, `m1_ik_batch1024_1thread.json`, `m2_trajopt_check.json`,
`m2_long_horizon.json` in this task folder.

Plan check (c) asked for every reachable target in float32. That is not met: 98–99% with restarts.

## Review record

A clean-context `mars_aux` reviewed `optim2` (`subagents/review_optim2.md`). It did not review
`terms/` or the IK recipe. Findings and what was done:

- Critical: `solve_layer` derivative wrong with scalar terms. Now raises `ValueError`.
- Major: a `(B,)` tensor weight silently broadcast over the wrong axis. Shape is now checked.
- Major: `Problem.with_` shared dictionaries with the parent. Now copied.
- `TorchOptim` state aliased the optimizer's parameters. Returns detached copies.
- LM never reported `done` in float32. Stops at maximum damping.
- Own finding after the review: `solve_layer` moved an unconverged solution by a full step. The value is
  now exactly the solver's result; only the derivative comes from the Newton step.

## Missing from design D3

- Augmented-Lagrangian recipe for hard constraints.
- Unroll mode of `solve_layer`.
- Variables shared over time (body shape) under the banded solver; today they force the dense path.
- BVR person-fit port.
- Trajectory optimization around obstacles (needs collision, P2).
- Docs and the CLAUDE.md guides still describe the old packages.

## Not verified

- Nothing was run on a GPU (all busy).
- G1 FK oracle mismatch (2e-8 vs 1e-10 tolerance, test failed before this work) is not investigated.
- URDF parser rounds values to float32, so analytic and autodiff Jacobians agree only to about 1e-7
  on float64 models. Left for P2.

## Assumptions (made off-hours)

- Temporary package names `optim2`, `terms`; names `KinematicState`, `forward_kinematics_state`.
- IK recipe default: hard joint limits, no rest or limit regularizers.
- Term names have their own namespace; variables, data and cached values share one.
- float32 LM stops on gradient 1e-7 or maximum damping only.
- `solve_layer`: implicit mode only; rejects scalar terms.
