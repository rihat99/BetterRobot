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

## Decision
We will solve the program by ADMM in the OSQP form on unit-diagonal variables with the rows in the caller's units:
one factorization of `H + penalty · rowsᵀ rows`, a fixed number of steps of two matrix-vector products, then one
active-set step. The active-set step solves the rows the multipliers press on as equalities and is kept only where
its point satisfies every row, every multiplier sign and stationarity to a relative `1e-6`: there it is the exact
solution. A warm start carries `x`, `dual` and the penalty, which each solve retunes for the next by OSQP's residual
ratio. The rows are hard and the form is two-sided (`lower == upper` is an equality, an infinite bound is one-sided,
identity rows are a box); a program whose rows cannot all hold returns the point of least squared violation in the
rows' own units, and its residual says so.

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
One solve at the default 20 steps costs 10 to 33 times one `Normal.solve` on one CPU thread (5 to 27 compiled), a
third of the interior point; the closed loop reaches the pose in 1.1 to 1.4 times the outer steps of the exact loop
with the same pose count within 3 %. A cold solve of an unrelated program is exact only where the active set is
found (39 of 64 random programs of 6 variables, none of 128 with 35 variables and 150 rows); calling again with
`warm=` continues it. The multipliers of rows that cannot hold grow with every step, so a caller that stays
infeasible for long should drop the warm start. The result carries no gradient.

## Alternatives
- The interior point as the only method: exact per call, three to ten times the cost, no warm start.
- ADMM alone: the lowest cost per call, more outer steps, no exact answers.
- L1-elastic rows with a multiplier cap, as ElastiQP: bounded multipliers on infeasible programs for one more
  parameter; not needed by the measured loops.

## Compatibility
Additive: `optim.solve_qp` and `optim.QPSolution` are new names, and no existing solver calls them.

## Enforcement
`tests/optim/test_qp.py` (closed forms, the optimality conditions after warm calls, rows in conflict, padding, failed
elements, one graph and no host read), `tests/contract/test_hot_path_lint.py` (the module is a watched hot path)
and the counter rows `qp/n7m30/B1` and `qp/n35m150/B1024` of `benchmarks/run.py`.

## Open questions
- Whether the multipliers need a cap (L1-elastic rows) once a control loop stays infeasible for long.
- Whether the constraint term kinds should hand the solver rows scaled to unit length; the solver leaves the rows as
  given so that the caller's scaling weights them.
