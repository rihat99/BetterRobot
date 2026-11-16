# 0055 Constraint terms are kinds on `Problem`, held by one solver, `SequentialQP`

**Status:** Proposed (integrator, awaiting owner). **Date:** 2026-11-16

## Context
The library held constraints two ways: a weighted hinge cost, which trades the constraint against the pose, and
`solve_constrained`, an augmented-Lagrangian loop around any solver, which names residual terms as constraints at
solve time. Neither gives an inverse-kinematics run that ends within the clearances it was asked to keep: on the
benchmark's IK protocol with clearance rows (Panda 7-dof and G1, 1024 targets, 50 steps, float64), the hinge at
weight 1e4 leaves 13 % of the Panda floor targets and 84 % of the G1 self-collision targets over 1e-4 m, and the
augmented Lagrangian, at the same tolerance, takes 120 to 210 solver steps on Panda (table below). Record 0054
added the batched quadratic program; this record says how a problem declares constraints and which solver holds
them.

Two facts shaped the solver, both measured on closed forms (64 problems of three variables pulled to targets
outside the constraint):

- A Gauss-Newton model of the loss has none of the curvature of a constraint term. On the unit ball with a target
  at distance 7, the true Hessian of the Lagrangian is 7 times the model's: plain steps overshoot along the
  boundary and fall into a cycle at radius 4. The damping has to stand in for that curvature.
- An acceptance rule that never lets the iterate leave the tolerance (a strict envelope) cannot follow a curved
  boundary: a tangential step leaves it at second order, so steps shrink to the square root of the tolerance, and a
  point within the tolerance of an equality stops there, short of the optimum (38 of 64 ball problems and every
  sphere problem ended over 1e-3 from the optimum; half the plane problems ended 4e-5 from it).

## Decision
1. **Kinds.** `Problem.equality(name, fn, *, weight, reads, jacobian)` holds `fn = 0` and
   `Problem.inequality(name, fn, *, lower=None, upper=None, weight, reads, jacobian)` holds `lower <= fn <= upper`
   (`None` leaves a side open; both `None` raises `ArgumentError`). `Term.kind` is one of `residual`, `scalar`,
   `equality`, `inequality`; the bounds live on the term (`Term.lower`, `Term.upper`). The function is the one a
   residual takes, a `TermFn` with its analytic Jacobian included, so one function serves as a cost or as a
   constraint. A constraint term is no part of `loss`, `evaluate` or `gradient`. Its weight scales its rows and
   bounds linearly: a number 0 switches the term off and a negative one raises `ArgumentError`; an entry 0 of a
   tensor weight switches its row off, and a negative entry is not checked, since that would read values back.
2. **One solver holds them.** `optim.SequentialQP(damping=1e-3, gain=0.5, tolerance=1e-4, gradient_tolerance=,
   step_tolerance=)` is a `Solver` beside LM and Gauss-Newton. LM, Gauss-Newton, L-BFGS and `TorchOptim` raise
   `ArgumentError` on a problem with an active constraint term; so do `solve_layer` (its derivative has no rows)
   and `solve_constrained` (it would re-register the term as a residual).
3. **The step.** One `solve_qp` call per step, started from the last solution (`State.extra["qp"]`, a warm key):
   the damped Gauss-Newton model under the rows `gain · (lower - c) <= J δ <= gain · (upper - c)` of the constraint
   terms and the box of `problem.step_bounds`. The row count is fixed between steps.
4. **Acceptance and damping.** A step is taken where it lowers the merit `loss + ν · (summed weighted violation)`.
   The price `ν` is the largest multiplier magnitude of the step's program, capped by the larger of two prices that
   multipliers drifting apart on redundant rows do not raise: the largest pull of the multipliers along one row,
   `|J_i · Jᵀ dual| / |J_i|²`, and the loss the step gives up per unit of violation it repairs,
   `max(0, -predicted loss decrease) / (0.9 · repair)`. No memory, no floor. The merit charges the rows of the
   `closest` pairs selected again at the trial point. The damping follows the ratio of the merit's decrease to the
   predicted one by Nielsen's rule with its whole factor (a taken step with a ratio under one half raises it). No
   second derivative of a term is used.
5. **The radius.** The program's step is scaled back so that its largest entry is at most a radius, `∞` at first;
   the program itself is unchanged, so the warm start's layout is too. A taken step doubles the radius. A refused
   step leaves the radius alone while the damping is under 1; once it is at least 1 the radius becomes the step's
   length, and a quarter of it where the step had been scaled back or was no shorter than 0.9 of the radius.
   `State.extra["radius"]` is a warm key.
6. **What a solve returns** is [record 0056](0056-sequential-qp-reports-its-last-point-within-tolerance.md): the last
   iterate within `tolerance` of every bound, with the iterate in `State.extra["iterate"]`.
7. **Stopping.** A problem is done once its iterate is within `tolerance` and either the gradient of the loss plus
   the rows' multipliers is under `gradient_tolerance` or a taken step is under `step_tolerance`, or when its
   damping reaches the maximum or its radius `1e-12`. LM's test on the loss decrease is left out: the merit is flat
   along a step that only repairs a row. A done problem keeps its damping and radius. `tolerance` must be above 0.

## Example
```python { .no-run }
problem = terms.ik_problem(model, {"panda_hand": goal}, q0=q0)
problem.inequality("self", terms.self_clearance(spheres, pairs=pairs, margin=0.02, hinge=False, closest=8), upper=0.0)
result = optim.solve(problem, optim.SequentialQP(), iterations=50)
result.diagnostics["phase"]["violation"][-1]        # (B,) metres over the margin, 0 where the clearance holds
```

## Consequences
Closed forms, float64, default settings, 8 draws of 256 problems at two scales, the step from which every problem
stays within 1e-6 of the optimum: box 2, plane 20 to 22 (3 at `gain=1.0`), unit ball 36 to 44, unit sphere 35 to
65 (`tests/optim/test_constraints.py` holds 5, 25, 50, 75 on every draw). The guide's circle under a line from 256
random starts: none ends outside the line or the circle. A sphere equality from near the origin converges at
weights 1 and 1e4, and a ball and a sphere on the same surface (redundant rows) converge on 64 of 64 problems in
100 steps (largest multiplier 3.9e2).

Scenes, 1024 targets, 50 steps, CPU, margin 0.02 m, `closest` 8 (Panda) and 64 (G1); rows recomputed in float64 over
every sphere pair from the returned configuration. Success is the benchmark's 1 mm and 0.01 rad over all targets,
reachable or not; float32 draws its own targets. "Start" counts targets whose reported point is the unmoved start;
"excursion" is the largest violation of an accepted iterate (record 0056).

| Scene | success, float64 | over 1e-4 | start | excursion | success, float32 | over 1e-4 | start |
|---|---|---|---|---|---|---|---|
| Panda, limits | 0.853 | 0 | 0 | 0 | 0.860 | 0 | 0 |
| Panda, floor | 0.753 | 0 | 0 | 0.372 m | 0.768 | 0 | 0 |
| Panda, obstacle sphere | 0.760 | 0 | 0 | 0.139 m | 0.777 | 0 | 0 |
| Panda, self-collision | 0.766 | 0 | 0 | 0.191 m | 0.780 | 0 | 0 |
| Panda, all | 0.633 | 0 | 1 | 0.161 m | 0.646 | 0 | 0 |
| G1, limits | 0.771 | 0 | 0 | 0 | 0.768 | 0 | 0 |
| G1, floor | 0.751 | 0 | 0 | 0.052 m | 0.737 | 0 | 0 |
| G1, obstacle sphere | 0.674 | 0 | 0 | 0.155 m | 0.691 | 0 | 0 |
| G1, self-collision | 0.427 | 0 | 104 | 0.178 m | 0.426 | 0 | 99 |
| G1, all | 0.346 | 0 | 155 | 0.173 m | 0.350 | 0 | 176 |

Almost every G1 target configuration is in self-collision (0.5 % satisfy the scene), so success there mostly counts
poses another, collision-free configuration reaches.

Against the alternatives, 256 targets, float64, eight threads of a shared host. Each cell: success / share of
targets with every row within 1e-4 / solver steps / seconds. `solve_constrained` runs LM at its defaults with
`tolerance=1e-4`, the tolerance `SequentialQP` holds; its second time is with the analytic Jacobian of its
inequality terms restored (it drops it, Open questions), success and steps unchanged.

| Scene (256 targets) | `SequentialQP` | hinge cost, weight 1e4, LM | `solve_constrained`, LM | same, analytic Jacobian |
|---|---|---|---|---|
| Panda, limits | 0.875 / 1.000 / 50 / 0.3 | 0.812 / 1.000 / 50 / 0.2 (bounds) | no constraint term | - |
| Panda, floor | 0.785 / 1.000 / 50 / 0.6 | 0.738 / 0.879 / 50 / 0.4 | 0.742 / 1.000 / 180 / 12.4 | 1.5 |
| Panda, obstacle sphere | 0.797 / 1.000 / 50 / 0.5 | 0.766 / 0.941 / 50 / 0.3 | 0.773 / 1.000 / 120 / 9.0 | 0.8 |
| Panda, self-collision | 0.777 / 1.000 / 50 / 0.5 | 0.734 / 0.848 / 50 / 0.4 | 0.742 / 1.000 / 180 / 4.7 | 1.5 |
| Panda, all | 0.676 / 1.000 / 50 / 0.9 | 0.637 / 0.730 / 50 / 0.6 | 0.656 / 1.000 / 210 / 138 | 3.3 |
| G1, limits | 0.773 / 1.000 / 50 / 1.5 | 0.594 / 1.000 / 50 / 1.0 (bounds) | no constraint term | - |
| G1, floor | 0.746 / 1.000 / 50 / 3.6 | 0.598 / 0.996 / 50 / 2.5 | 0.629 / 1.000 / 60 / 80 | 3.1 |
| G1, obstacle sphere | 0.711 / 1.000 / 50 / 3.4 | 0.520 / 0.887 / 50 / 2.3 | 0.652 / 1.000 / 180 / 257 | 9.6 |
| G1, self-collision | 0.461 / 1.000 / 50 / 6.5 | 0.180 / 0.160 / 50 / 5.6 | 0.238 / 0.777 / 300 / 882 | 44.7 |
| G1, all | 0.363 / 1.000 / 50 / 9.8 | 0.133 / 0.113 / 50 / 6.7 | 0.199 / 0.750 / 300 / stopped at 1800 | 63.8 |

On G1 with every row the library's `solve_constrained` was stopped after 30 minutes; success, share and steps there
are those of the run with the analytic Jacobian, which equal the library's on every other scene.

- The step rule before the radius and the capped price (the largest multiplier alone) reached the same success on
  Panda and within 3 targets of 1024 on G1, but stalled on redundant rows (48 of 64 at 100 steps, multipliers up to
  7.9e4) and ended done at an infeasible point where a row forced the step (a sphere from near the origin at weight
  1e4, 2 of 256 circle problems).
- `gain` 1.0 against 0.5: linear equalities in 3 steps instead of 20 to 22, the ball faster, but 8 targets of 1024
  fewer on Panda all and more targets whose iterate leaves the rows (Panda all 776 against 683). 0.5 is the default.
- In float32 a problem whose loss stays large at the optimum stops where a step's decrease drops under the rounding
  of that loss (box: 12 of 64 problems end over 1e-4 from the optimum, up to 7e-4); inverse kinematics, whose loss
  falls to zero at a reached pose, is not such a problem.
- Cost per step: one `solve_qp` call and one linearization, also where the step is refused. The radius and the
  capped price add 881 operators to the counter row `ik_constrained/panda/B64` (now 31393), about 42 a step.
- The normal equations must be dense; a `time=` problem needs `Problem(dense=True)`.

## Alternatives
- A `kind=` argument on `residual`, or constraint names passed to the solver as `solve_constrained` takes them: the
  problem then does not say what it holds, and a solver that minimizes only the loss cannot refuse it.
- LM taking a quadratic program when the problem has constraint terms: a switch that changes the algorithm, its
  cost and its state without the caller asking.
- The strict envelope as the acceptance rule, with or without the merit's ratio driving the damping: Context.
- A restoration step from the trial point before refusing it (a second program and linearization per step): ball
  5 of 64 and sphere 1 of 64 within 1e-6 under the envelope.
- The radius as box rows inside the program (the textbook Sl1QP form): with the box binding in every coordinate and
  a term row active the solution is a degenerate vertex, about half of the program's solutions come back inexact,
  those steps are refused and the radius collapses: 1000 to 1250 of 2048 ball and sphere problems never within 1e-6
  in 200 steps.
- The scaled-back radius with a textbook update (a quarter on every refusal): closed forms 117 to 200 steps, and
  Panda loses 4 to 11 targets of 1024. Recording every refused length, also under a damping of 1: Panda 1 to 5
  targets fewer on four scenes and 5 more on one (the shipped rule equals the rule before it on Panda), sphere 37 to
  59 steps against 35 to 65. Shrinking only where the loss model rises: ball 65 to 76 and sphere 67 to 76 steps, the
  sphere from near the origin at weight 1e4 still ends infeasible, redundant rows stall (59 of 64 at 400 steps).
- Damping that also scales the rows' right-hand side: closed forms 150 to 200 steps.
- A penalty that never decreases (the largest multiplier seen so far times 1.1): closed forms 78 to 120 steps and
  G1 all success 95 to 86 of 256. With a factor over 1 and no memory: 73 to 90 steps. The pull along a row alone:
  two rows at an angle carrying opposite multipliers price it too low (4 of 256 circle problems end done outside).
- A floor on the penalty of 3 or 10 times the stationarity measure, which fades at a solution: excursions fall to
  0.10 to 0.13 m (3) or 0.05 to 0.08 m (10) and the G1 targets that return the start to 15 and 17 of 256 (10), but
  G1 self success falls from 117 to 108 of 256 (10).
- A funnel, a cap on the violation of an accepted step at 100 times the tolerance: 343 to 586 of 2048 closed-form
  problems never within 1e-6.
- Exact second derivatives of the constraint terms in the model: ruled out by the owner's brief.

## Compatibility
Additive on `Problem`, `Term` and the solver list. `solve_constrained` and `solve_layer` now raise on a problem that
holds an active constraint term (the weight 0 switches one off), which no problem could before. `State.extra` of
`SequentialQP` has keys of its own.

## Enforcement
`tests/optim/test_constraints.py` (rows, weights and bounds of the kinds; the four closed forms within their step
counts on every draw; a step a row forces, the circle under a line, redundant rows, zero and negative weights; the
reported point within tolerance while the iterate is not; no problem done while its iterate is outside; the
refusals),
`tests/optim/test_solver_state.py::test_sequential_qp_reads_nothing_back_and_warm_starts_from_its_last_program`,
`tests/terms/test_constrained_ik.py` (every row of a Panda scene within 1e-4 after 50 steps, all sphere pairs;
compiled step equals eager), the counter row `ik_constrained/panda/B64` of `benchmarks/run.py`.

## Open questions
- Whether `solve_constrained` retires. It is the route for banded `time=` problems and for solvers outside the
  Gauss-Newton family. On every scene of the table `SequentialQP` reached more poses in 50 steps than it did in 60
  to 300 at the same tolerance. Against its time with the analytic Jacobian restored, `SequentialQP` took 0.27 to
  0.63 of it on Panda, 0.15 to 0.35 on three G1 scenes and 1.16 times it on the G1 floor, where it stops after 60
  steps.
- `solve_constrained` drops the analytic Jacobian of its inequality terms (`_shifted` in
  `src/better_robot/optim/constrained.py`), so each of its steps differentiates them by autodiff: 3 to 42 times its
  time on the Panda scenes, 20 to 27 times on G1. Not changed here, since `solve_constrained` stays as it is in
  this work.
- The merit charges the `closest` pairs selected at the trial point, not every pair: a field on `TermFn` that hands
  the solver every pair's value would let it charge them all, at one more all-pair evaluation per linearization.
- `gain` is one number per solver. A per-term gain would let a differential-IK layer damp one clearance more than
  another; no scene here needed it.
- A safer default for rows of small scale (a larger `weight` for `solve_qp` near convergence, where the loss's
  gradient tends to 0 and only the diagonal of the Gauss-Newton matrix holds the cap): not measured.
- Evidence on record 0054's open questions from closed-loop use: the default `weight` of `solve_qp` sufficed for
  every scene, whose rows went in unscaled, in metres against radians; a row scaled to a thousandth gives way
  although it could hold (unit ball at weight 0.001: 61 of 64 problems return the start); the box rows of the joint
  limits never gave way in a run that ended within tolerance.
