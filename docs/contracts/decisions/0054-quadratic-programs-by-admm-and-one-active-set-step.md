# 0054 Quadratic programs are solved by warm-started ADMM steps and one active-set step

**Status:** Proposed (integrator, awaiting owner). **Date:** 2026-11-15

## Context
Inequality constraints enter the solvers as one quadratic program per step: minimise `½ xᵀH x + gᵀx` over rows
`lower <= rows x <= upper`, with 7 to 35 variables, up to about 150 rows after the closest-pairs selection, thousands
of problems per batch, and an `H` that is a damped Gauss-Newton matrix (condition number 1e4 to 1e8 on a redundant
robot). The solve must run the same steps for every batch element, read nothing back to the host, and trace as one
graph (contract 09). Measured on Panda and G1 problems with joint-limit, floor, obstacle and self-collision rows,
closed loop over 40 constrained Gauss-Newton steps, one CPU thread, float64:

- A primal-dual interior point (Mehrotra, 12 to 20 Newton steps, each a Gram product of the rows and a
  factorization) follows the loop that solves every program exactly, at 73 to 106 times one `Normal.solve`, and
  cannot start from the last answer.
- Accelerated projected gradient on the dual box diverges: its primal point is `-H⁻¹(g + rowsᵀ dual)`, and `H⁻¹` is
  huge along the directions the damping alone holds.
- ADMM in the OSQP form alone, 20 to 50 steps, started from its last answer with the penalty carried, reaches the
  same poses but takes 1.5 to 3.5 times the outer steps: its steps are slow along those weak directions.
- Rows scaled to unit length in the metric of `H⁻¹` (whitening) make ADMM worse, not better: every box row then
  sees a weak direction.

A review of the first version, which held the rows hard, found that a long stretch of rows in conflict wound up the
multipliers (to 1.8e9) and the carried penalty (to its cap): steps of 47 against a box of 0.3 during the stretch,
457 of 512 problems still wrong ten calls after the conflict ended, and up to 314 wrong problems in one call that
reported both residuals under 1e-4. It also found that one call solves few problems cold, whatever its number of steps.

## Decision
We will solve the program by ADMM in the OSQP form on unit-diagonal variables with the rows in the caller's units:
one factorization of `H + penalty · rowsᵀ rows` and its inverse, a fixed number of steps of three matrix-vector
products, then one active-set step.

- The rows are L1-elastic with one finite `weight` (default `1e3`): the problem solved is
  `½ xᵀH x + gᵀx + weight · (sum of the rows' violations)`, so every multiplier stays within `±weight` (the dual
  box of ElastiQP). Where the rows can all hold with multipliers under `weight`, its minimiser is the one under
  hard rows; where they cannot, it is the L1-elastic minimiser: the least sum of violations, then the least
  objective. `weight` is in units of the objective per unit of a row. It scales with the objective; scaling a row
  and its bounds by `t` counts its violation `t` times and divides its multiplier by `t`; the unit-diagonal scaling
  of the variables changes neither.
- The form is two-sided: `lower == upper` is an equality, held in every active-set step with a multiplier of
  either sign; an infinite bound is one-sided; identity rows are a box; a zero row with infinite bounds is padding.
  A row whose `lower` is above its `upper` is the equality at `upper`. A zero row whose bounds exclude 0 can never
  hold: its multiplier stays at `weight` and the other rows are solved as without it.
- The active-set step solves the rows the multipliers press on as equalities. Its point is kept where every row
  sits where its multiplier holds it and the gradient vanishes, both to a relative `1e-6` (of the largest row value
  and of the largest term of the gradient): there it is the solution. Elsewhere the last ADMM step is returned.
- `primal_residual` is the largest distance of a row from where its multiplier holds it: from `upper` where
  `dual > 0`, from `lower` where `dual < 0`, from the bounds where `dual == 0` (the violation). `dual_residual` is
  the largest entry of `H x + g + rowsᵀ dual`. Both are in the caller's units. Together they vanish only at the
  solution under hard rows; at an elastic minimiser whose rows give way, `primal_residual` is the largest violation.
- One call is a warm-startable step, not a solve. `warm` carries `x`, `dual` and the penalty. Each call retunes the
  penalty for the next by OSQP's rule on the residuals of the elastic problem, by at most a factor of 10, and keeps
  it within a factor of 1000 of the penalty a cold call of the problem at hand starts with, so no stretch of calls
  leaves a state that needs a reset. The name stays `solve_qp` until the owner decides (open questions).

## Example
```python { .no-run }
solution = None
for _ in range(steps):                      # one constrained Gauss-Newton step per pass
    normal, rows, lower, upper = linearize(q)
    damped = normal.H + torch.diag_embed(damping * normal.H.diagonal(dim1=-2, dim2=-1))
    solution = solve_qp(damped, normal.g, rows=rows, lower=lower, upper=upper, warm=solution)
    q = integrate(model, q, solution.x)
```

## Consequences
Accuracy comes from repeated warm calls, not from `iterations`: the penalty that sets the pace of the steps is
tuned between calls. Random feasible problems with half of the rows active, `cond(H) = 1e4`; solved means both
residuals under `1e-8`:

| Calls | 7 variables, 30 rows: solved of 256 | median violation | 35 variables, 150 rows: solved of 128 | median violation |
|---|---|---|---|---|
| one cold call of 20 iterations | 0 | 0.34 | 0 | 0.52 |
| one cold call of 50 / 100 / 200 iterations | 0 | 0.14 / 0.12 / 0.10 | 0 | 0.19 / 0.19 / 0.18 |
| one cold call of 500 iterations | 0 | 0.063 | 0 | 0.13 |
| 5 warm calls of 20 iterations | 46 | 0.0061 | 0 | 0.030 |
| 12 warm calls of 20 iterations | 185 | 7e-16 | 0 | 0.0055 |
| 40 warm calls of 20 iterations | 247 | 4e-16 | 43 | 2.3e-4 |

The rows bound the step only in the limit. With `cond(H) = 1e8`, a box of 0.3 and random `g`, one cold call of 20
iterations returns steps up to 35 (233 of 256 problems beyond the box by more than 1e-2), 3.4 at 200 iterations;
warm calls bring it to 2.9, 0.61 and 0.30 after 2, 3 and 5 calls. On problems shaped like inverse kinematics (a
6 by 7 Jacobian, damping 1e-4) one cold call reaches 0.38 (12 of 512 problems beyond the box), 0.39 with two rows
in conflict (31 of 512), and the second warm call 0.30. `primal_residual` reports every such step (at least 0.01).

The multiplier cap ends the wind-up. In the review's three scenarios (512, 256 and 512 problems of 7 variables in a
box of 0.3, 20 iterations per warm call; two rows in conflict whose normals drift for 100 calls, one row asked both
ways for 200 calls, the same for 50 calls with a fixed row), before and after:

| Measure | hard rows | elastic rows, `weight=1e3` |
|---|---|---|
| largest step during the conflict | 46.9 / 0.46 / 0.43 | 1.77 / 0.46 / 0.43 |
| problems wrong by more than 1e-3, ten calls after the conflict ends | 457 / 256 / 511 | 0 / 0 / 0 |
| most problems wrong with both residuals under 1e-4, at any call | 314 / 11 / 509 | 0 / 0 / 0 |

Steps beyond 1 come from 3 of the 512 problems, and they are what the elastic problem asks for. The last program
of the conflict, solved to convergence (gradient of the elastic problem under 2e-9), gives steps up to 1.62 in its
largest problem: one box row gives way by 1.32 and both rows in conflict hold, where the least-squares compromise
stays at 0.79 with violations summing to 1.66; the elastic point has the lower elastic cost on all 512 problems. A
caller that needs the box to win scales the box's rows up.
With `weight=1e4`, 3, 0 and 4 problems were still wrong ten calls after the conflict; with `1e6`, all of them. The
largest multiplier of the Panda and G1 loops below, whose terms have unit weights, is 4.8: the default leaves a
margin of 200.

Closed loop (target T4: every row within `1e-4` at the end). The table counts the problems that are over `1e-4` at
step 40 although the loop that solves every program exactly is not, per scene (Panda seed 1 of 128 problems,
Panda seed 2 of 128, G1 seed 0 of 64), with 20 iterations per warm call unless stated:

| Variant | Panda s1 / Panda s2 / G1 s0 |
|---|---|
| one call per step (this record) | 4 / 2 / 1 |
| rows tightened by 5e-4 | 0 / 1 / 2 |
| rows tightened by 1e-3 | 1 / 1 / 0 |
| 2 calls per step | 2 / 1 / 1 |
| 5 calls per step | 2 / 1 / 1 (largest 1.2e-2 on Panda) |
| 10 calls on the last 10 steps | 3 / 2 / 0 |
| 2 to 5 restoration steps (gain 1 or 0.5) | 3-5 / 1-3 / 1 |
| 50 iterations per call (Panda s1) | 3 |
| 200 iterations on the last 10 steps (Panda s1) | 3 |

With one call per step the loop reaches 105, 112, 61 and 63 poses (G1 seed 1 too) against 107, 114, 61 and 63 for
the exact loop, and reaches a pose in 1.1 to 1.3 times its outer steps. No change on the solver's side meets T4 on
all three scenes: the problems that end over `1e-4` slide along a constraint without reaching the pose, and the
exact step of their last program sits at the step cap (null-space motion at damping 1e-4, repaired at the barrier
gain). T4 belongs to the outer loop that calls this solver (damping adaptation, step rejection, the barrier gain).

Cost of one call against one `Normal.solve` of the same `H` (0.65 ms at 7 variables, 26.6 ms at 35 variables),
batch 1024, float64, one CPU core and one thread on a shared host, cold calls (warm calls cost the same within the
noise of that host, up to 25 %); compiled is through `kinematics.compiled`:

| Variables, rows | Mode | 0 iterations | 10 iterations | 20 iterations |
|---|---|---|---|---|
| 7, 30 | eager | 12.6 ms (19x) | 15.6 ms (24x) | 21.1 ms (33x) |
| 7, 30 | compiled | 6.2 ms (9.6x) | 10.3 ms (16x) | 13.6 ms (21x) |
| 35, 150 | eager | 229 ms (8.6x) | 284 ms (11x) | 325 ms (12x) |
| 35, 150 | compiled | 241 ms (9.1x) | 409 ms (15x) | 491 ms (19x) |

The budget of the task, 5 to 10 times one `Normal.solve` at batch 1024 and more, compiled, is missed on one CPU
core: the call without iterations, compiled, already costs 9.1 to 9.6 times (the Gram product of the rows, two
factorizations, the inverse and the active-set step), and each iteration adds about 0.4 ms at 7 variables and 5
ms (eager) to 12 ms (compiled) at 35. Compiling helps at 7 variables and costs at 35, where inductor's code for the
batched products is slower than eager. On GPU, compiled, the first version took 6.7 to 7.6 times at 7 variables
and 30 rows (batch 1024; 18 times at 16384) and 13 to 20 times at 35 variables and 150 rows; the present form was
not measured on GPU.

The result carries no gradient.

## Alternatives
- The interior point as the only method: exact per call, three to ten times the cost, no warm start.
- ADMM alone: the lowest cost per call, more outer steps, no exact answers.
- Hard rows, an infeasible program returning the point of least squared violation: the first version. Its
  multipliers and its carried penalty grew through a long stretch of rows in conflict (Context), and the caller
  could not reset what it cannot see. Clamping the multipliers in the caller's warm start bounded the steps, but
  needed the caller to know the solver's internals.
- The penalty of a cold call at every call, carrying only `x` and `dual`: rows that had been in conflict were not
  released (all problems of the three scenarios still wrong twenty calls later).
- Retuning the penalty inside a call on a fixed schedule (once at half the steps, at thirds, at quarters): one more
  factorization each, and still no cold call solved (median violation 0.13, 0.092, 0.074 at 7 variables and 20
  iterations, against 0.34).
- Two matrix-vector products per step through `penalty · (H + penalty · rowsᵀ rows)⁻¹ rowsᵀ` (`N` by `M`), formed
  by triangular solves with `M` right-hand sides: forming it took about 250 of the 433 ms of a call without steps
  at 35 variables and 150 rows (batch 1024, one CPU core), where the chosen form takes 200 ms for the whole call.
  One product per step through the `M` by `M` map from the held rows to the rows' values: 27 against 21 ms at 7
  variables and 30 rows, 1584 against 315 ms at 35 variables (20 iterations, the same core).

## Compatibility
Additive: `optim.solve_qp` and `optim.QPSolution` are new names, and no existing solver calls them.

## Enforcement
`tests/optim/test_qp.py` (closed forms, equality rows alone as a linear solve, the optimality conditions after warm
calls, rows in conflict giving the elastic minimiser in the rows' units, a row giving way at the weight, three long
stretches of rows in conflict that neither wind up nor hide a wrong step, padding and repeated rows, a zero row
that cannot hold and crossed bounds, failed elements, a warm start of another layout, one graph and no host read),
`tests/contract/test_hot_path_lint.py` (the module is a watched hot path) and the counter rows `qp/n7m30/B1` and
`qp/n35m150/B1024` of `benchmarks/run.py`.

## Open questions
- Should the function be named for what one call is, `qp_step`, rather than `solve_qp`?
- The default `weight` suits an objective of order one. Should the outer loop set it from the scale of its
  objective (its term weights), or is a fixed default with the documented rule enough?
- The L1-elastic minimiser can break a box row by much (1.77 against 0.3) where a least-squares compromise stays in
  the box. Is scaling the box's rows up enough for the callers, or should box rows carry a weight of their own?
- Whether the constraint term kinds should hand the solver rows scaled to unit length; the solver leaves the rows as
  given so that the caller's scaling weights them.
