# 0057 Differential inverse kinematics is one call of `optim.track` per tick

**Status:** Proposed (integrator, awaiting owner). **Date:** 2026-11-16

## Context
`SequentialQP` (record 0055) solves one problem to convergence. A control loop whose targets move every tick (a
teleoperated hand, a planner's waypoints) needs one step per tick, from the configuration the robot is at, under
the same rows: frax's `DiffIKSolver.step`, a velocity-level QP with first-order barrier rows for joint limits,
floor, obstacles and self-collision. The owner asked for that layer over `solve_qp` (record 0054), with the five
behaviours (joint limits, floor, obstacle, self-collision, singularity) as rows, the cost forms of record 0051's
terms kept, spheres only, one obvious way and a small public surface.

Everything the tick needs is already on a `Problem`: the cost terms give the Gauss-Newton model, the constraint
terms the rows, `bounds` the box, the manifolds the retraction. Nothing in it is about robots.

## Decision
1. **One function in `optim`.** `optim.track(problem, x, *, warm=None, gain=0.5, damping=1e-3, step=None) ->
   (x, QPSolution)`: one `linearize` at `x`, one `solve_qp` started from `warm`, one `retract`. The new point and the
   program's solution are the whole state a loop carries; the targets move between calls through
   `problem.with_(data=)` and the rows keep their count (`closest=`), so the warm start fits.
2. **The program.** Minimize `½ δᵀ(H + damping·I) δ + gᵀδ`, the Gauss-Newton model of the cost terms with an
   absolute (Levenberg) damping, under `SequentialQP`'s rows at `gain`, `gain · (lower - c) <= J δ <= gain ·
   (upper - c)`, and the problem's bounds as the box `gain · (lower - x) <= δ <= gain · (upper - x)` (the velocity
   damper of the joint limits), each entry of that box clamped to `±step` (a number or `(nv,)`: velocity limits
   times the tick period). `SequentialQP` and `track` build the rows with one private function; `SequentialQP`
   keeps its full box (`box=1`).
3. **No acceptance.** The step is taken. Where the program's step is not its exact solution it is clamped to the
   box and scaled back until it leaves no row by more than 1e-6 and takes no violated row further out.
4. **Row forms of the behaviours.** Each behaviour's row is its cost term with `hinge=False`:
   `terms.manipulability(..., hinge=False)` (`threshold - w`, new) and `terms.sphere_clearance(..., closest=k)` (the
   `k` smallest distances, new, as `self_clearance` already had); the joint limits are the bounds.
5. **No posture term, no task gain by default.** A posture term is the caller's `terms.rest` (`ik_problem(
   rest_weight=)`); the task error enters at full weight.
6. **A moving obstacle needs no new argument.** Its distance function reads a tensor the loop moves in place
   (`copy_`); the problem, its rows and a compiled tick stay as they are.

## Example
```python { .no-run }
problem = terms.ik_problem(model, {"panda_hand": pose}, q0=q0)
problem.inequality("self", terms.self_clearance(spheres, pairs=pairs, margin=0.02, hinge=False, closest=8), upper=0.0)
x, warm = dict(problem.variables), None
for pose in stream:
    x, warm = optim.track(problem.with_(data={"target/panda_hand": pose}), x, warm=warm, step=0.005)
```

## Consequences
Measured with 256 tracking tasks per scene, 75 % of them on paths that cross the scene's region, the path taking
2 s and held for 1 s, tick 0.01 s (300 ticks) or 0.0025 s (1200 ticks), step = min(velocity limit, 2.5 rad/s) times
the tick, float64 unless said, CPU. Largest violation over every tick and task (all sphere pairs, not the closest
`k`); "cost" registers each behaviour as its hinge at weight 1e3, "free" leaves it out.

| Scene | Panda rows, 25 mrad | Panda rows, 6.3 mrad | Panda cost / free, 25 mrad | G1 rows, 25 mrad | G1 rows, 6.3 mrad | G1 cost / free, 25 mrad |
|---|---|---|---|---|---|---|
| joint limits | 0 | 0 | 2.5e-2 / 2.1 | 0 | 0 | 2.5e-2 / 2.9 |
| floor | 3.1e-4 | 2.9e-5 | 8.6e-3 / 0.42 | 2.6e-6 | 5.5e-6 | 2.8e-3 / 3.2e-2 |
| obstacle | 5.0e-4 | 4.0e-5 | 1.3e-2 / 0.18 | 9.0e-4 | 5.8e-5 | 5.9e-2 / 0.19 |
| self-collision | 2.3e-4 | 6.4e-5 | 9.4e-3 / 0.21 | 4.1e-4 | 9.9e-5 | 2.8e-2 / 0.20 |
| singularity | 9.2e-5 | 6.9e-5 | 1.3e-2 / 2.9e-2 | 1.2e-4 | 7.7e-5 | 7.2e-3 / 1.7e-2 |
| all five | 3.9e-4 | 7.5e-5 | 2.4e-2 / 0.61 | 5.9e-4 | 2.9e-4 | 3.5e-2 / 3.0 |

- The joint-limit box holds exactly (to rounding in float32, 6e-8). Curved rows are left by what their
  linearization misses, which grows with the square of the step: within 1e-4 takes about 6.3 mrad per tick on the
  Panda (float32 too: at most 1.0e-4). At that step the G1 holds each behaviour alone within 1e-4 (float32: 8.9e-5)
  and is just over with all five at once: 2.9e-4, 3 of 256 tasks over 1e-4 (float32: 3.0e-4, 5 tasks).
- Tracking where the target is clear: median error 0.02 to 0.07 mm at 25 mrad in every form; targets the rows hold
  off from end farther (Panda all five: 5 of 146 clear final targets missed by over 1 cm, free 0).
- Largest joint acceleration is about the step over the tick squared in every form, rows or not (the velocity
  bound saturates and reverses).
- `solve_qp` at its 20 iterations returns an inexact step on 2 to 7 % of Panda ticks and 30 to 60 % of G1 ticks with
  self-collision rows; the scale-back keeps those within the table. On the G1 (64 tasks, self-collision alone / all
  five): 60 iterations took 6.3 mrad from 9.9e-5 / 1.01e-4 to 2.6e-5 / 6.8e-5 and did not help at 25 mrad (2.2e-4 /
  4.7e-4 became 3.3e-4 / 6.7e-4: the curvature dominates); `closest=128` instead of 64 gave 3.0e-5 / 1.15e-4.
- A moving obstacle: the rows see it where the tick starts, so one closing in by `v` per tick ends a tick up to
  `v / gain` inside the margin (measured 2.6 mm at 1.3 mm per tick); padding its radius by `v / gain` holds it
  within 1e-4. Moved in place, it compiles nothing new (`tests/terms/test_constrained_ik.py`).
- Damping 1e-4, 1e-3, 1e-2: median tracking error 0.02, 0.04, 0.3 mm (Panda, B=64, all five).

## Alternatives
- A function in `terms` or a controller class holding the tasks: it would need a public way to read the rows out
  of `optim`, or a second row builder; a class adds a state type where a `QPSolution` suffices.
- A task gain below 1 (`v_des = -gain · error`, as frax): the largest acceleration stayed (308 against 474 rad/s²)
  and the median tracking error grew from 0.04 to 14 mm.
- A posture term by default: weight 1e-2 toward the start biased the median tracking error to 40 mm.
- Marquardt damping (the diagonal of `H`): it leaves joints that no target sees, and a rank-deficient `H`,
  unregularized.
- More `solve_qp` calls per tick on the same program: no gain over the iterations of one call.
- Hinge cost forms only (record 0051): 1e-2 to 6e-2 violations at the same steps (table).
- A data name on `sphere_clearance` for the obstacle centre (frax's `ref`): a new parameter and a new
  `distance_fn` signature for what an in-place tensor already does.

## Compatibility
Additive: `optim.track`; `hinge=` on `terms.manipulability` (default `True`, the old form);
`closest=` on `terms.sphere_clearance` (default `None`, every distance). `SequentialQP` takes the same steps as
before (its box at `box=1`, the counter row `ik_constrained/panda/B64` keeps its operator count).

## Enforcement
`tests/optim/test_track.py` (a linear row and the box close in by at most the gain per call and hold, step bound,
refusals), `tests/terms/test_constrained_ik.py` (each behaviour's row within 1e-4 over 100 ticks of 5 mrad on a
Panda whose targets break every behaviour; no host read per tick; compiled tick equals eager in both modes; an
obstacle moved in place compiles nothing new and is held to `v / gain`), `tests/terms/test_robot_terms.py` and
`tests/collision/test_collision.py` (the new row forms against the hinge forms and `check_jacobian`), the counter
row `track/panda/B64` of `benchmarks/run.py`.

## Open questions
- One `gain` for every row. frax gives each constraint its own; no scene here needed it.
- `track` has no `iterations=` for its `solve_qp` call; the G1 with every behaviour at 6.3 mrad is the one case
  where more (60) helped.
- An obstacle's velocity as a term of its row (`J δ >= -gain (d - margin) + v`) would remove the lag of a moving
  obstacle without padding; it needs the velocity as data.
