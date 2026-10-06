# REPORT — P1 new optim (2026-10-07)

Status: built, reviewed twice, committed on `dev`. Not pushed. Old `optim`,
`residuals`, `tasks` packages are untouched; deleting them is the owner's decision.

## What exists

Two new packages, about 2,000 lines together.

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
  residual term is time-local or reads no time variable, the solver is block-tridiagonal and cost
  grows linearly with clip length. Variables without a time axis (body shape, a camera) become a
  dense border of that matrix (`ArrowNormal`), so they no longer force the dense path.
- `constrained.py`: `solve_constrained` enforces `term = 0` and `term <= 0` for named residual terms
  (augmented Lagrangian) on top of any solver.
- For first-order fits like BVR's: `TorchOptim(..., scheduler=)` steps a learning-rate schedule,
  a free mask may cover one knot and repeats over time, `Phase.on_step` is a logging hook, and
  `Phase.on_start` may write `problem.data` (contact labels, scene candidates).
- Tests: `tests/optim2/` (21). Examples: `examples/optim2/01_ik.py`, `02_trajectory.py`,
  `03_clip_fit_phases.py`, `04_ik_layer.py`, `05_hard_constraints.py`.
- Agent guide: `/home/rikhat.akizhanov/better/dev/BetterRobot/src/better_robot/optim2/CLAUDE.md`.

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
| G1 300 frames + 18 unknowns shared over the clip | bordered solve 0.05 s per step; loss equals dense to 12 digits (checked at 30 and 100 frames) | dense not feasible at 300 |
| Panda, 64 problems, hand at a point (equality) and elbow under a ceiling (inequality) | worst violation 6e-7 m after 10 rounds, 189 LM steps | penalty only: 1.2e-2 m |

The G1 run with shared unknowns takes 33 s for 30 LM steps, against 8.8 s without them. The extra
time is autodiff of the marker term through FK (1.1 s per linearization), not the solver. That
synthetic problem does not pin down the shared offsets (six tracked points, 35 joints), so it shows
agreement with the dense path and timing only, not parameter recovery.

Raw files: `subagents/m1_ik_check.json`, `m1_ik_batch1024_1thread.json`, `m2_trajopt_check.json`,
`m2_long_horizon.json`, `m3_shared_variable.json` in this task folder.

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

Second review, of the additions after `65b63bc` (`subagents/review_optim2_additions.md`): the bordered
solver matched the dense path in every case the reviewer built. One major finding: Jacobian columns of
shared variables came out zero, silently, when the term used an operation that cannot be differentiated
twice (a custom autograd function, a Warp kernel). Now checked on every linearization; a mismatch raises
an error. Two minor findings fixed or documented. The reviewer did not check: LM conditioning as the
penalty grows, windowed constraints, multipliers at active bounds.

## BVR person fit: can the new layer express it?

A read-only study (`subagents/study_bvr_fit.md`, code reading, nothing run) listed BVR's variables,
terms and phases and nine gaps. State after tonight:

- Closed: learning-rate decay inside a phase; shared `betas` beside per-frame `q`; phase-start
  state read by terms; a per-DOF free mask repeated over time; per-term logging.
- Works as is: strong-Wolfe L-BFGS through `TorchOptim(torch.optim.LBFGS)`; terms with
  non-differentiable parts (visibility, nearest neighbours) as scalar terms.
- Stays first-order: chamfer and score-normalized reprojection have normalizers that depend on the
  variables, so they are scalar terms; LM does not apply to them without reformulation.
- Not done: the port itself. It needs BVR's body model and data, which belong to P4 (BetterHuman).
  The design's acceptance test "ported fit reproduces BVR's losses" is therefore open.

## Missing from design D3

- BVR person-fit port (see above; planned with P4).
- Unroll mode of `solve_layer`, and the stationarity check with fallback the design describes.
  Today `solve_layer` is implicit only and trusts convergence. Recommendation: add the check, skip
  unroll until a use case needs it.
- Trajectory optimization around obstacles (needs collision, P2).
- `docs/` and the top-level CLAUDE.md still describe the old packages; they change when the old
  packages are deleted (owner's call).

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
