# 0055 Constraint terms are kinds on `Problem`, held by one solver, `SequentialQP`

**Status:** Proposed (integrator, awaiting owner). **Date:** 2026-11-16

## Context
The library held constraints two ways: a weighted hinge cost, which trades the constraint against the pose, and
`solve_constrained`, an augmented-Lagrangian loop around any solver, which names residual terms as constraints at
solve time. Neither gives an inverse-kinematics run that ends within the clearances it was asked to keep: on the
benchmark's IK protocol with clearance rows (Panda 7-dof and G1, 1024 targets, 50 steps, float64), the hinge at
weight 1e4 leaves 13 % of the Panda floor targets and 84 % of the G1 self-collision targets over 1e-4 m, and the
augmented Lagrangian takes 210 to 300 solver steps (table below). Record 0054 added the batched quadratic program;
this record says how a problem declares constraints and which solver holds them.

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
   constraint. A constraint term is no part of `loss`, `evaluate` or `gradient`; its weight scales its rows and
   bounds linearly and must be positive.
2. **One solver holds them.** `optim.SequentialQP(damping=1e-3, gain=0.5, tolerance=1e-4, gradient_tolerance=,
   step_tolerance=)` is a `Solver` beside LM and Gauss-Newton. LM, Gauss-Newton, L-BFGS and `TorchOptim` raise
   `ArgumentError` on a problem with an active constraint term; so do `solve_layer` (its derivative has no rows)
   and `solve_constrained` (it would re-register the term as a residual).
3. **The step.** One `solve_qp` call per step, started from the last solution (`State.extra["qp"]`, a warm key):
   the damped Gauss-Newton model under the rows `gain · (lower - c) <= J δ <= gain · (upper - c)` of the constraint
   terms and the box of `problem.step_bounds`. The row count is fixed between steps.
4. **Acceptance and damping.** A step is taken where it lowers the merit `loss + ν · (summed weighted violation)`
   with `ν` the largest multiplier magnitude of that step's program. The damping follows the ratio of the merit's
   decrease to the predicted one by Nielsen's rule with its whole factor (a taken step with a ratio under one half
   raises it). No second derivative of a term is used.
5. **What a solve returns.** `State.x` is the last iterate within `tolerance` of every bound, in the terms' units
   (before any was, the closest); `State.loss` and the `violation` diagnostic describe it, and the iterate is
   `State.extra["iterate"]`. A solve that starts within `tolerance` returns a point within it after any number of
   steps.
6. **Stopping.** A problem is done once its iterate is within `tolerance` and either the gradient of the loss plus
   the rows' multipliers is under `gradient_tolerance` or a taken step is under `step_tolerance`, or when its
   damping reaches the maximum. LM's test on the loss decrease is left out: the merit is flat along a step that
   only repairs a row. A done problem keeps its damping.

## Example
```python { .no-run }
problem = terms.ik_problem(model, {"panda_hand": goal}, q0=q0)
problem.inequality("self", terms.self_clearance(spheres, pairs=pairs, margin=0.02, hinge=False, closest=8), upper=0.0)
result = optim.solve(problem, optim.SequentialQP(), iterations=50)
result.diagnostics["phase"]["violation"][-1]        # (B,) metres over the margin, 0 where the clearance holds
```

## Consequences
Closed forms, float64, default settings, the step from which all 64 problems stay within 1e-6 of the optimum: box
2, plane 20 (3 at `gain=1.0`), unit ball 39, unit sphere 57 (`tests/optim/test_constraints.py` holds 5, 25, 50, 70).

Scenes, 1024 targets, 50 steps, float64 CPU, margin 0.02 m, `closest` 8 (Panda) and 64 (G1). "within" is the share
of targets with every nonlinear row within 1e-4 (all sphere pairs); success is the benchmark's 1 mm and 0.01 rad
over all targets, reachable or not. Share of target configurations that satisfy the scene: Panda floor 0.88,
obstacle 0.93, self 0.85, all 0.71; G1 floor 0.995, obstacle 0.86, self 0.005, all 0.004.

Each cell: success / within / solver steps; `solve_constrained` at its defaults.

| Scene (1024 targets) | `SequentialQP` | hinge cost, weight 1e4, LM | `solve_constrained`, LM |
|---|---|---|---|
| Panda, floor | 0.753 / 1.000 / 50 | 0.723 / 0.868 / 50 | 0.730 / 1.000 / 240 |
| Panda, obstacle sphere | 0.760 / 1.000 / 50 | 0.758 / 0.939 / 50 | 0.769 / 1.000 / 210 |
| Panda, self-collision | 0.766 / 1.000 / 50 | 0.732 / 0.854 / 50 | 0.755 / 1.000 / 300 |
| G1, floor | 0.751 / 1.000 / 50 | 0.595 / 0.995 / 50 | 0.629 / 1.000 / 60 (256 targets) |
| G1, obstacle sphere | 0.674 / 1.000 / 50 | 0.511 / 0.862 / 50 | not measured |
| G1, self-collision | 0.425 / 1.000 / 50 | 0.176 / 0.160 / 50 | 0.238 / 0.777 / 300 (256 targets) |
| G1, self-collision, targets drawn collision free | 0.722 / 1.000 / 50 | 0.453 / 0.461 / 50 | not measured |

- The envelope rule this replaces met the tolerance too and reached fewer poses where rows are active at the
  answer: G1 self 0.248, G1 all 0.183, Panda all 0.616; elsewhere within 0.004 of the table (Panda obstacle 0.762
  against 0.760 is the one scene where it is ahead, by 2 targets of 1024).
- `gain` 0.5, 0.75 and 1.0 give the same success within 0.011 on every scene. 0.5 is the default: on the float32
  sphere it leaves 63 of 64 problems within 1e-4 against 22 at 1.0, and the iterate strays less (G1 self: at most
  9e-3 m against 0.15 m mid-run). Its price is a linear rate of one half on rows that need no reserve (the plane).
- In float32 a problem whose loss stays large at the optimum stops where a step's decrease drops under the rounding
  of that loss (box: 12 of 64 problems end over 1e-4 from the optimum, up to 7e-4); inverse kinematics, whose loss
  falls to zero at a reached pose, is not such a problem (counter row `ik_constrained/panda/B64`: violation 8e-5).
- Cost per step: one `solve_qp` call and one linearization, also where the step is refused.
- The normal equations must be dense; a `time=` problem needs `Problem(dense=True)`.

## Alternatives
- A `kind=` argument on `residual`, or constraint names passed to the solver as `solve_constrained` takes them: the
  problem then does not say what it holds, and a solver that minimizes only the loss cannot refuse it.
- LM taking a quadratic program when the problem has constraint terms: a switch that changes the algorithm, its
  cost and its state without the caller asking.
- The strict envelope as the acceptance rule, with or without the merit's ratio driving the damping: Context.
- A restoration step from the trial point before refusing it (a second program and linearization per step): ball
  5 of 64 and sphere 1 of 64 within 1e-6 under the envelope.
- A penalty with memory (the largest multiplier seen so far): one early large multiplier, from a start where the
  row's Jacobian is small, stays and slows every later step (ball 59 of 64, sphere 56 of 64).
- A penalty of 1.25, 1.5 or 2 times the multiplier: no change on the scenes, twice the steps on the ball.
- Nielsen's factor clamped at one on a taken step, as LM has it: the damping never grows along an accepted crawl
  (ball 60 of 64).
- Exact second derivatives of the constraint terms in the model: ruled out by the owner's brief.

## Compatibility
Additive on `Problem`, `Term` and the solver list. `solve_constrained` and `solve_layer` now raise on a problem that
holds a constraint term, which no problem could before. `State.extra` of `SequentialQP` has keys of its own.

## Enforcement
`tests/optim/test_constraints.py` (rows, weights and bounds of the kinds; the four closed forms within their step
counts; the reported point within tolerance while the iterate is not; the refusals),
`tests/optim/test_solver_state.py::test_sequential_qp_reads_nothing_back_and_warm_starts_from_its_last_program`,
`tests/terms/test_constrained_ik.py` (every row of a Panda scene within 1e-4 after 50 steps, all sphere pairs;
compiled step equals eager), the counter row `ik_constrained/panda/B64` of `benchmarks/run.py`.

## Open questions
- `State.x` is not the point the solver stands on. The alternative keeps the iterate in `State.x` and gives up the
  guarantee of point 5 (16 of 256 G1 self targets ended up to 3e-3 m over).
- Whether `solve_constrained` retires. It is the route for banded `time=` problems and for solvers outside the
  Gauss-Newton family; on dense problems `SequentialQP` reached more poses in a sixth of the steps (table).
- `gain` is one number per solver. A per-term gain would let a differential-IK layer damp one clearance more than
  another; no scene here needed it.
- Evidence on record 0054's open questions from closed-loop use: the default `weight` of `solve_qp` was never the
  limit (no scene needed another value); rows went in unscaled, in metres against radians; the box rows of the
  joint limits never gave way in a run that ended within tolerance.
