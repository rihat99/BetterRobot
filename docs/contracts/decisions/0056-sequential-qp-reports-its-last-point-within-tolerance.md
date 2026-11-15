# 0056 `SequentialQP` reports its last point within the tolerance, not its iterate

**Status:** Proposed (integrator, awaiting owner). **Date:** 2026-11-16

## Context
`SequentialQP` ([record 0055](0055-constraint-terms-and-sequential-qp.md)) takes a step where it lowers a merit, and
its rows are linear: along a curved constraint the iterate leaves it by what the rows do not see, and later steps
bring it back. On the benchmark's IK scenes (1024 targets, 50 steps, float64, margin 0.02 m) accepted iterates left
the clearances by up to 0.37 m (Panda floor) and 0.05 to 0.19 m on the other scenes with clearances. A step rule that
never lets the iterate leave the tolerance cannot follow a curved boundary (Alternatives). So the point the solver
stands on is not always a point that holds the constraints, and a caller needs to know which one `State.x` is.

## Decision
1. We will make `State.x` the last iterate that was within `tolerance` of every bound, in the terms' own units, or,
   before any was, the closest one so far. `State.loss`, `State.extra["violation"]` and the `violation` diagnostic
   describe that point.
2. The iterate and its loss are `State.extra["iterate"]`; the next step starts there, and `normal` and `rows` of
   `State.extra` are linearized there.
3. A solve that starts within `tolerance` returns a point within it after any number of steps, also on a target it
   cannot reach.
4. `done` says that a problem stopped. Whether its rows hold is the `violation` diagnostic, with no flag of its own:
   a problem whose rows cannot all hold, or whose iterates never came within the tolerance, can end done with a
   violation over it.

## Example
```python { .no-run }
result = optim.solve(problem, optim.SequentialQP(), iterations=50)
held = result.diagnostics["phase"]["violation"][-1] <= 1e-4    # (B,): every row within the tolerance at result.x
iterate, iterate_loss = result.state.extra["iterate"]          # where the solver stands
```

## Consequences
- Every nonlinear row is within `1e-4` at `State.x` on all ten scenes of the benchmark's IK protocol (Panda 7-dof
  and G1, limits, floor, obstacle sphere, self-collision, all; 1024 targets, 50 steps), in float64 and in float32,
  with the rows recomputed over every sphere pair: 0 targets over, largest violation 9.97e-5.
- Where no iterate after the start comes back within the tolerance, `State.x` is the unmoved start: G1
  self-collision 104 and G1 all 155 of 1024 targets in float64 (99 and 176 in float32), Panda all 1, the other
  scenes 0.
- The reported loss can rise during a run: `State.x` moves to a later iterate that is within the tolerance and
  presses closer on the constraint, at a higher loss. On closed forms (8 draws of 256 problems) that happens on 1083
  to 1121 of 2048 plane problems and on 1992 to 2045 of 2048 ball and sphere problems; on the IK scenes on up to 236
  of 1024 targets (G1 all).
- Cost: a selection by `torch.where` per step; no host read.

## Alternatives
- The iterate as `State.x`: a solve that starts within the tolerance may then end outside it. Under an earlier step
  rule 16 of 256 G1 self-collision targets ended up to 3e-3 m over; at the shipped rule accepted iterates leave by
  up to 0.37 m (Context).
- The best point by loss among those within the tolerance: an iterate up to `tolerance` outside an active
  constraint has a lower loss than the optimum, so it is kept and the reported point never reaches the optimum.
  Measured as a temporary change on the closed forms (8 draws of 256 problems, 200 steps): 1083 to 2044 of 2048
  plane, ball and sphere problems never came within 1e-6 of the optimum, against 0 under this rule.
- A step rule that never lets the iterate leave the tolerance (envelope acceptance), so that the iterate is the
  answer: a tangential step leaves a curved bound at second order, so steps shrink to the square root of the
  tolerance, and a point within the tolerance of an equality stops there. On 64 problems of three variables, 38 of
  the ball problems and every sphere problem ended over 1e-3 from the optimum, and half the plane problems 4e-5
  short of it.
- A flag for "done without holding": `violation` already says it, per problem and per step.

## Compatibility
New with `SequentialQP` in 0.3.0; no other solver reports this way. A caller that wants the iterate reads
`State.extra["iterate"]`.

## Enforcement
`tests/optim/test_constraints.py` (`test_the_reported_point_never_leaves_the_tolerance_while_the_iterate_does`,
`test_a_problem_is_not_done_while_its_iterate_is_outside_the_tolerance`), `tests/terms/test_constrained_ik.py`
(every row of a Panda scene within 1e-4 at `result.x` after 50 steps, over every sphere pair).

## Open questions
- Whether to give `SequentialQP` an initial radius, so that the first steps cannot leave the constraints by
  decimetres; it would lower the share of targets that return the start, at a cost not measured.
- Whether the iterate, not the reported point, should be what a caller sees by default.
