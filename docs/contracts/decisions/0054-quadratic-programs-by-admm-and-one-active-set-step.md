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

A second review, of the elastic form with one absolute multiplier cap of `1e3`, checked every claim against an
independent active-set solve of the L1-elastic optimality conditions and found three faults. A warm call pulled the
slack of every row that gives way back inside its bounds, so the elastic minimiser was not a fixed point: a warm
call started there moved 37, 44 and 78 problems of the three scenarios below by more than 1e-3 (up to 0.27). The
active-set step cut a held row's multiplier of the wrong sign to zero and kept the point: on problems shaped like
inverse kinematics at damping 1e-6, 8 kept points sat 3.7e-3 to 5.8e-2 from the minimiser with a primal residual
of 0 and a dual residual of 3e-7 to 2.6e-6. And one absolute cap failed both ways: at a hundredth of an objective of
order one, rows that had been in conflict let go slowly (498 of 512 problems wrong ten calls later), and at a
thousand times it, rows of feasible problems gave way (11 of 512).

## Decision
We will solve the program by ADMM in the OSQP form on unit-diagonal variables with the rows in the caller's units:
one factorization of `H + penalty · rowsᵀ rows` and its inverse, a fixed number of steps of three matrix-vector
products, then one active-set step.

- The rows are L1-elastic. Each batch element has a cap, `weight · max(‖g‖∞, largest diagonal entry of H)`, with
  `weight` a positive Python number (default `100.0`, checked at the boundary): the problem solved is
  `½ xᵀH x + gᵀx + cap · (sum of the rows' violations)`, so every multiplier stays within `±cap` (the dual box of
  ElastiQP). Where the rows can all hold with multipliers under the cap, its minimiser is the one under hard rows.
  Where they cannot, it is the L1-elastic minimiser for that cap: a row gives way where holding it would cost the
  objective more than the cap per unit of its violation. The point of least violation sum, then least objective,
  is only the limit of a large `weight`. The cap scales with the objective, so `weight` is a pure number; the
  diagonal of `H` keeps the cap from vanishing with `g` (at the end of a converging loop `‖g‖∞` falls to 1e-16).
  Scaling a row and its bounds by `t` counts its violation `t` times and divides its multiplier by `t`; the
  unit-diagonal scaling of the variables changes neither.
- The form is two-sided: `lower == upper` is an equality, held in every active-set step with a multiplier of
  either sign; an infinite bound is one-sided; identity rows are a box; a zero row with infinite bounds is padding.
  A row whose `lower` is above its `upper` is the equality at `upper`. A zero row whose bounds exclude 0 can never
  hold: its multiplier stays at the cap and the other rows are solved as without it.
- The active-set step solves the rows the multipliers press on as equalities. Its point is kept where every row
  sits where its multiplier holds it and the gradient vanishes, both to a relative `1e-6` (of the largest row value
  and of the largest term of the gradient), and no row it holds asks for a multiplier of the wrong sign or beyond
  the cap. A point that would need such a multiplier cut is refused, not cut to fit. `QPSolution.exact` marks the
  elements whose point was kept; elsewhere the last ADMM step is returned.
- `primal_residual` is the largest distance of a row from where its multiplier holds it: from `upper` where
  `dual > 0`, from `lower` where `dual < 0`, from the bounds where `dual == 0` (the violation). `dual_residual` is
  the largest entry of `H x + g + rowsᵀ dual`. Both are in the caller's units. Small residuals bound the gap in the
  objective, not the distance to the minimiser along weak curvature, and at an elastic minimiser whose rows give
  way `primal_residual` is the largest violation: `exact` is the sign that a step is the minimiser.
- One call is a warm-startable step, not a solve. `warm` carries `x`, `dual` and the penalty. Each row starts from
  its value at the given `x` (OSQP's warm start), so a warm call started at the minimiser stays there. Each call
  retunes the penalty for the next by OSQP's rule on the residuals of the elastic problem, by at most a factor of
  10, and keeps it within a factor of 1000 of the penalty a cold call of the problem at hand starts with, so no
  stretch of calls leaves a state that needs a reset. The name stays `solve_qp` until the owner decides (open
  questions).

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
| one cold call of 20 iterations | 0 | 0.33 | 0 | 0.50 |
| one cold call of 50 / 100 / 200 iterations | 0 | 0.14 / 0.12 / 0.10 | 0 | 0.19 / 0.18 / 0.18 |
| one cold call of 500 iterations | 0 | 0.062 | 0 | 0.13 |
| 5 warm calls of 20 iterations | 47 | 0.0064 | 0 | 0.029 |
| 12 warm calls of 20 iterations | 181 | 7e-16 | 0 | 0.0055 |
| 40 warm calls of 20 iterations | 247 | 4e-16 | 44 | 2.5e-4 |

The rows bound the step only in the limit. With `cond(H) = 1e8`, a box of 0.3 and random `g`, one cold call of 20
iterations returns steps up to 35 (233 of 256 problems beyond the box by more than 1e-2), 3.4 at 200 iterations;
warm calls bring it to 3.0, 0.61 and 0.30 after 2, 3 and 5 calls. On problems shaped like inverse kinematics (a
6 by 7 Jacobian, damping 1e-4) one cold call reaches 0.38 (12 of 512 problems beyond the box), 0.39 with two rows
in conflict (31 of 512), and the second warm call 0.30. `primal_residual` reports every such step (at least 0.01).

Small residuals do not place a step near the minimiser. On problems shaped like inverse kinematics at damping 1e-4,
one problem of 512 sits 1.3e-2 to 1.6e-2 from the minimiser with a primal residual of 0 and a dual residual of
4.5e-5 to 4.9e-5, after one cold call of 20 or 100 iterations and after 2, 5 and 10 warm calls; on a drifting
sequence 1 or 2 problems sit up to 1.15e-2 off. Their objective is at most 1.4e-6 above the minimum, and none of
them is `exact`. Refusing a point whose held multiplier would be cut removed every kept point that was wrong: at
damping 1e-6 and 1e-8, 6 and 7 kept points more than 1e-3 off (up to 5.8e-2) became 0, the worst kept point is
2.2e-6 and 2.4e-5 off, and the share of kept points fell from 9904 to 9883 of 12288; it did not change on the other
problems, feasible or not.

The multiplier cap ends the wind-up. In the review's three scenarios (512, 256 and 512 problems of 7 variables in a
box of 0.3, 20 iterations per warm call; two rows in conflict whose normals drift for 100 calls, one row asked both
ways for 200 calls, the same for 50 calls with a fixed row), before and after:

| Measure | hard rows | elastic rows (this record) |
|---|---|---|
| largest step during the conflict | 46.9 / 0.46 / 0.43 | 1.71 / 0.46 / 0.43 |
| problems wrong by more than 1e-3, ten calls after the conflict ends | 457 / 256 / 511 | 0 / 0 / 0 |
| most problems wrong with both residuals under 1e-4, at any call | 314 / 11 / 509 | 0 / 0 / 0 |

The largest steps are ADMM steps on the way to an elastic minimiser that itself leaves the box: the review traced
a step of 1.77 to the fourth call of the drifting stretch, 0.135 from the minimiser of that program (1.640). The
minimisers of the stretch reach 1.64 (1.76 over the review's stretch of 1000 calls): one box row gives way, and
both rows in conflict then hold, where a least-squares compromise would stay in the box. A caller that needs the
box to win scales the box's rows up.

Recovery after rows in conflict, wrong meaning more than 1e-3 from the certified minimiser of the program at hand.
No problem was `exact` while wrong in any of these runs:

| Problems, after the stretch in conflict | wrong at call 1 / 3 / 5 / 10 | last wrong call |
|---|---|---|
| drift, 512 problems, 7 variables (100 calls in conflict) | 510 / 194 / 11 / 0 | 7 |
| one row both ways, 256 (200 calls) | 253 / 97 / 7 / 0 | 6 |
| fixed row, 512 (50 calls) | 452 / 129 / 11 / 0 | 7 |
| drift, objective times 0.01 / times 1000 | 509 / 192 / 11 / 0 and 510 / 194 / 11 / 0 | 7 / 7 |
| float32: drift / both ways / fixed | 510 / 192 / 11 / 0, 254 / 97 / 7 / 0, 492 / 168 / 10 / 0 | 7 / 6 / 6 |
| drift with 21 more rows, 512 (200 calls) | 511 / 340 / 72 / 8 | 4 still wrong at call 20 |
| 35 variables, 150 rows, two pairs in conflict, 48 problems (100 calls) | 48 / 48 / 48 / 20, then 4 at call 15 and 1 at 20 (float32 the same) | 23 (float32 25) |
| shaped like inverse kinematics, damping 1e-4 / 1e-6 / 1e-8, 512 (40 calls) | 511 / 506 / 247 / 8 (all three within 3) | 8 to 10 still wrong at calls 10 to 20 |

Without the conflict, the 35-variable problems are all right from the first call. The problems shaped like inverse
kinematics leave 4 to 6 of 512 wrong after 10 to 20 warm calls without any conflict too (the weak curvature
above); after the conflict 3 or 4 of the wrong ones report both residuals under 1e-4, and none is `exact`. Under
the cap of `1e3` the review measured 46 of 48 wrong at call 10 at 35 variables, and 498 of 512 at a hundredth of the
objective.

Equality rows in conflict take longer, and neither residual shows it. On 256 problems of two equalities that cannot
both hold, wrong at call 5 / 10 / 20 / 40 / 60 / 100 / 200:

| Equalities in conflict | this record | cap of `1e3`, slack pulled inside the bounds |
|---|---|---|
| two targets 0.2 apart | 256 / 227 / 53 / 7 / 2 / 0 / 0 | 255 / 255 / 255 / 226 / 139 / 64 / 26 |
| a combination off by 0.1 | 255 / 255 / 237 / 132 / 64 / 17 / 3 | 255 at calls 5 to 60, 251, 145 |

A wrong problem there has a dual residual of 1e-11 to 1e-10 (medians) and a positive primal residual, as the
minimiser does; `exact` is unset on every wrong problem at every call.

The cap. Each variant below ran on the same tables: problems shaped like inverse kinematics with the objective
scaled by 1, 1000 and 3000 (feasible, near-opposed rows down to a wedge of 0.03), twelve cases of rows that ask for
a motion while `g` is small or zero, the recovery scenarios above, and (all but the fourth) the closed loops below.

| Cap | feasible rows that gave way at objective times 1000 / 3000 (of 512) | rows asking for a motion that gave way (12 cases) | recovery, last wrong call: drift at times 1 / 0.01 / 1000; both ways; fixed |
|---|---|---|---|
| `1e3`, absolute | 11 / 182 | 0 at times 1; 15, 53, 13 and 13 at times 1000 | 9 / 157 wrong at call 20 / rows broken, steps of 531; 7; 8 |
| `100 ‖g‖∞` | 0 / 0 | 8 of 502 where `g` is small; all where `g` is zero | 7 / 7 / 7; 6; 7 |
| `100 max(‖g‖∞, 1)` | 0 / 0 | 8, 3 and 217 at times 1000 | 7 / 14 / 7; 6; 7 |
| `100 ‖g‖∞` of the unit-diagonal problem | 1 / 12 | up to 337 of 502 | 7 / 9 / 4; 6; 8 |
| `30 max(‖g‖∞, max diag H)` | 0 / 0 (margin 8.4) | 0 | 5 / 5 / 5; 4; 5 |
| `100 max(‖g‖∞, max diag H)` (this record) | 0 / 0 (margin 28) | 0 | 7 / 7 / 7; 6; 7 |

The margin is the smallest ratio of the cap to the largest multiplier a feasible problem needs, at the wedge of
0.03. The review's rule for an absolute cap, at least 3 `‖g‖∞` and 15 `‖g‖∞` with near-opposed rows, is met 6.7
times over at the default. The closed loops are identical under every cap run on them: the cap never binds there (the largest
multiplier is 18 `‖g‖∞` on Panda, 3.8 to 5.4 on G1). A pure `‖g‖∞` scale is unsafe because `g` vanishes at the end
of a converging loop while a row may still ask for a motion; the diagonal of `H` is the scale of the objective one
unit of a variable away.

Closed loop (target T4: every row within `1e-4` at the end). The table counts the problems that are over `1e-4` at
step 40 although the loop that solves every program exactly is not, per scene (Panda seed 1 of 128 problems,
Panda seed 2 of 128, G1 seed 0 of 64), with 20 iterations per warm call unless stated. The first row is this
record; the others were measured on the first elastic form (cap of `1e3`, slack pulled inside the bounds), whose
own row was 4 / 2 / 1:

| Variant | Panda s1 / Panda s2 / G1 s0 |
|---|---|
| one call per step (this record) | 3 / 4 / 1 |
| rows tightened by 5e-4 | 0 / 1 / 2 |
| rows tightened by 1e-3 | 1 / 1 / 0 |
| 2 calls per step | 2 / 1 / 1 |
| 5 calls per step | 2 / 1 / 1 (largest 1.2e-2 on Panda) |
| 10 calls on the last 10 steps | 3 / 2 / 0 |
| 2 to 5 restoration steps (gain 1 or 0.5) | 3-5 / 1-3 / 1 |
| 50 iterations per call (Panda s1) | 3 |
| 200 iterations on the last 10 steps (Panda s1) | 3 |

The change from 4 / 2 / 1 comes from the warm start alone (that change by itself gives 3 / 4 / 1); the refusal
rule and the cap change no count. With one call per step the loop reaches 105, 112, 61 and 63 poses (G1 seed 1
too) against 107, 114, 61 and 63 for the exact loop, and reaches a pose in 1.1 to 1.3 times its outer steps. No
change on the solver's side meets T4 on all three scenes: the problems that end over `1e-4` slide along a
constraint without reaching the pose, and the exact step of their last program sits at the step cap (null-space
motion at damping 1e-4, repaired at the barrier gain). T4 belongs to the outer loop that calls this solver (damping
adaptation, step rejection, the barrier gain).

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
batched products is slower than eager. The table was taken before the second review; its changes (the cap, the
refusal test, the warm start) add no cost beyond the noise of the host: eager, 20 iterations, the forms before and
after alternated on one core, four runs each, cold 20.5 to 23.5 ms before and 23.3 to 23.7 ms after at 7
variables, 303 to 309 and 312 to 317 ms at 35; warm 20.8 to 21.1 and 18.1 to 22.2 ms, 302 to 308 and 285 to
309 ms. On one GPU (free when the run started and when it
ended, not watched in between), compiled, against a compiled `Normal.solve` (0.19 ms at 7 variables and 0.46 ms at
35, batch 1024):

| Variables, rows | Batch | 0 iterations | 10 iterations | 20 iterations |
|---|---|---|---|---|
| 7, 30 | 1024 | 0.85 ms (4.5x) | 1.08 ms (5.8x) | 1.49 ms (7.9x) |
| 35, 150 | 1024 | 3.46 ms (7.6x) | 4.84 ms (11x) | 5.54 ms (12x) |
| 7, 30 | 16384 | 2.99 ms (8.9x) | 4.54 ms (14x) | 5.62 ms (17x) |
| 35, 150 | 16384 | 49.3 ms (10x) | 70.7 ms (14x) | 83.0 ms (17x) |

The budget is met at 7 variables and batch 1024 only.

The result carries no gradient.

## Alternatives
- The interior point as the only method: exact per call, three to ten times the cost, no warm start.
- ADMM alone: the lowest cost per call, more outer steps, no exact answers.
- Hard rows, an infeasible program returning the point of least squared violation: the first version. Its
  multipliers and its carried penalty grew through a long stretch of rows in conflict (Context), and the caller
  could not reset what it cannot see. Clamping the multipliers in the caller's warm start bounded the steps, but
  needed the caller to know the solver's internals.
- One absolute cap (`1e3`, the first elastic form): rows of feasible problems give way at a large objective and rows
  in conflict let go slowly at a small one (the table of caps). A cap of `weight · ‖g‖∞` alone lets every row go
  where `g` vanishes; with a floor of 1 on `‖g‖∞` it lets rows go at a large objective and slows recovery at a small
  one; computed on the unit-diagonal problem it lets feasible rows go.
- Cutting a held multiplier of the wrong sign to zero and keeping the point (the first elastic form): kept points
  up to 5.8e-2 from the minimiser along weak curvature. Refusing only a wrong sign, or also a multiplier at the cap,
  gives the same counts as refusing any cut.
- A warm start that pulls each row's slack inside its bounds (the first elastic form), or only the slack of rows
  that do not give way: the minimiser is not a fixed point (with the second, 18 to 43 problems still move, because
  `penalty · (dual / penalty)` differs from `dual` in floating point).
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
calls, rows in conflict giving the elastic minimiser in the rows' units, a row giving way at the cap, three long
stretches of rows in conflict that stay bounded and let go with no wrong step flagged `exact`, a warm call at the
elastic minimiser staying there, `exact` meaning a minimiser where the curvature is weak, the cap following the
scale of the objective, padding and repeated rows, a zero row that cannot hold and crossed bounds, failed elements,
a warm start of another layout and a `weight` that is not a positive number, one graph and no host read),
`tests/contract/test_hot_path_lint.py` (the module is a watched hot path) and the counter rows `qp/n7m30/B1` and
`qp/n35m150/B1024` of `benchmarks/run.py`.

## Open questions
- Should the function be named for what one call is, `qp_step`, rather than `solve_qp`?
- Is `weight · max(‖g‖∞, largest diagonal entry of H)` the right cap? It compares a multiplier (objective per unit
  of a row) with a gradient (objective per unit of a variable), which assumes rows whose coefficients are of order
  one and variables in natural units: every problem of the table of caps is such a problem, and a row scaled by `t`
  or a variable in other units moves the margin by that factor. The alternatives are an absolute cap with a rule for
  the caller, or a cap per row.
- The L1-elastic minimiser can break a box row by much (1.76 against 0.3) where a least-squares compromise stays in
  the box. Is scaling the box's rows up enough for the callers, or should box rows carry a weight of their own?
- Whether the constraint term kinds should hand the solver rows scaled to unit length; the solver leaves the rows as
  given so that the caller's scaling weights them.
