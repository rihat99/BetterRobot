# Solve inverse kinematics under constraints

A clearance that a robot must keep, a contact that must stay closed, a limit that must hold: register it as a
constraint term and solve with `optim.SequentialQP`. After this guide you can hold equalities and inequalities in
any `Problem`, read how far a solution is from them, and choose between this solver, a weighted hinge cost and
`optim.solve_constrained`.

## A constraint is a term registered with its bounds

`problem.equality(name, fn)` holds `fn = 0`; `problem.inequality(name, fn, lower=, upper=)` holds
`lower <= fn <= upper`. The function is the one a residual term takes. The toy problem of
[Hard constraints](hard-constraints.md): get as close as possible to the point `(1, 1)` while staying on the unit
circle and below the line `y = 0.5`. The answer is `(√0.75, 0.5) ≈ (0.866, 0.5)`.

```python
import torch
from better_robot import optim

start = torch.tensor([[2.0, -1.0]], dtype=torch.float64)          # p (B=1, 2)
problem = optim.Problem({"p": start}, data={"target": torch.tensor([[1.0, 1.0]], dtype=torch.float64)})
problem.residual("close", lambda p, target: p - target)
problem.equality("circle", lambda p: (p * p).sum(-1, keepdim=True) - 1.0)
problem.inequality("below", lambda p: p[:, 1:2], upper=0.5)

result = optim.solve(problem, optim.SequentialQP(), iterations=60)
print(result.x["p"])                                          # ≈ [[0.866, 0.500]]
print(result.iterations, bool(result.done.all()))
```
```output
tensor([[0.8660, 0.5000]], dtype=torch.float64)
30 True
```

`result.loss` is the loss of the cost terms only (here `close`). `result.diagnostics["phase"]["violation"]` is
`(steps, B)`: the largest distance of a constraint term's output from its bounds at `result.x` after each step, in
that term's unit, and 0 where every constraint holds. `done` says that a problem stopped, not that its rows hold: a
problem whose rows cannot all hold ends done at the point that breaks them least. Read the violation:

```python
held = result.diagnostics["phase"]["violation"][-1] <= 1e-4    # (B,): every row within the tolerance at result.x
print(held, result.done & ~held)                               # done, yet not held: no point within it was found
```
```output
tensor([True]) tensor([False])
```

## A Panda reaches poses and keeps clear of a sphere and of itself

The clearance terms of `better_robot.terms` return `margin - distance`. With `hinge=False` that value is the row
`margin - distance <= 0`; with the default hinge it is a penalty. One function, two registrations:

```python
import math

import better_robot as br
from robot_descriptions import panda_description
from better_robot import collision, forward_kinematics, terms

model = br.load(panda_description.URDF_PATH, dtype=torch.float64)
spheres, ignore = collision.load_spheres(collision.SPHERE_SETS / "panda.json", model)
pairs = collision.self_collision_pairs(model, spheres, ignore=ignore)
obstacle, radius, margin = torch.tensor([0.45, 0.25, 0.45], dtype=torch.float64), 0.10, 0.02


def to_obstacle(centers: torch.Tensor, radii: torch.Tensor) -> torch.Tensor:   # (B, S, 3), (S,) -> (B, S)
    return (centers - obstacle).norm(dim=-1) - radii - radius


ready = [0.0, -math.pi / 4, 0.0, -3 * math.pi / 4, 0.0, math.pi / 2, math.pi / 4, 0.04, 0.04]
q0 = torch.tensor(ready[: model.nq], dtype=torch.float64).expand(64, -1)
q_target = terms.random_start(model, q0, generator=torch.Generator().manual_seed(0))
hand = model.frame_id("panda_hand")
goal = forward_kinematics(model, q_target).frame_poses[:, hand]              # 64 poses; 12 of them lie in collision


def build(*, rows: bool) -> optim.Problem:
    problem = terms.ik_problem(model, {"panda_hand": goal}, q0=q0)
    avoid = terms.sphere_clearance(spheres, to_obstacle, margin=margin, hinge=not rows)
    apart = terms.self_clearance(spheres, pairs=pairs, margin=margin, hinge=not rows, closest=8)
    if rows:
        problem.inequality("obstacle", avoid, upper=0.0)
        problem.inequality("self", apart, upper=0.0)
    else:
        problem.residual("obstacle", avoid, weight=1e4)
        problem.residual("self", apart, weight=1e4)
    return problem


def report(q: torch.Tensor) -> str:
    state = forward_kinematics(model, q)
    clear = torch.cat([to_obstacle(spheres.centers(state), spheres.radii),
                       collision.self_distances(spheres, state, pairs=pairs)], dim=-1)
    deepest = (margin - clear).amax(-1).clamp_min(0.0)                       # over every pair, not the closest 8
    reached = (state.frame_poses[:, hand, :3] - goal[:, :3]).norm(dim=-1) < 1e-3
    inside = int((deepest > 1e-4).sum())
    return f"reached {int(reached.sum())} of 64, {inside} inside the margin, deepest {1e3 * float(deepest.max()):.1f} mm"


held = optim.solve(build(rows=True), optim.SequentialQP(), iterations=50)
cost = optim.solve(build(rows=False), optim.LevenbergMarquardt(), iterations=50)
print("rows:", report(held.x["q"]))
print("cost:", report(cost.x["q"]))
```
```output
rows: reached 46 of 64, 0 inside the margin, deepest 0.0 mm
cost: reached 41 of 64, 17 inside the margin, deepest 5.2 mm
```

The joint limits need no term: `ik_problem` passes them as `bounds=`, and `SequentialQP` holds bounds as a box on
every step. `closest=8` keeps the eight nearest sphere pairs of each configuration as rows, so the row count is the
same at every step; the report above checks all pairs.

## How a step is taken

Each step solves one quadratic program with `optim.solve_qp`: the damped Gauss-Newton model of the loss under the
constraint terms linearized at the iterate, started from the last step's solution. The step is scaled back to a
radius that refused steps set, and taken where it lowers a merit, the loss plus the priced violation of the rows
([Optimization](../concepts/optimization.md#constraint-terms) has the rule).

| Argument | Default | Meaning |
|---|---|---|
| `damping` | `1e-3` | initial damping; it then follows the steps, as in `LevenbergMarquardt` |
| `gain` | `0.5` | share of a row's distance to its bound that one step may use, and of a violation that it repairs; `1.0` is the plain linearization |
| `tolerance` | `1e-4` | distance from a bound, in the term's unit, within which a point counts as holding the constraints; above 0 |
| `gradient_tolerance`, `step_tolerance` | as LM | a problem is done once its iterate is within `tolerance` and one of them is met |

- **What you get back.** Linear rows miss the curvature of a constraint, so the iterate leaves a curved one and the
  next steps bring it back. On the benchmark's scenes (1024 targets, 50 steps, float64) accepted iterates left the
  clearances by up to 0.37 m on the Panda floor scene and by 0.05 to 0.19 m on the other scenes with clearances.
  `result.x` is therefore the last iterate that was within `tolerance` of every bound: a solve that starts there
  ends there after any number of steps, also on a target it cannot reach. Where no later iterate comes back within
  `tolerance`, `result.x` is the unmoved start: 104 of 1024 G1 self-collision targets and 155 of 1024 with every
  G1 row; on Panda 1 of 1024 with every row, none on the other scenes. The iterate is
  `result.state.extra["iterate"]`. Start from a configuration that holds the constraints; from one that does not,
  `result.x` is the closest iterate so far until one comes within `tolerance`.
- **Units.** `tolerance` and the `violation` diagnostic are in the units of the terms: metres for a clearance.
- **Weights and scale.** A constraint term's `weight` scales its rows. `solve_qp` lets a row give way once holding
  it would take a multiplier over its cap, 100 times the larger of the loss gradient's largest entry and the
  largest diagonal entry of the damped Gauss-Newton matrix. Rows that cannot all hold give way where holding them
  costs most per unit of weighted violation. A row whose coefficients are a hundredth or less can give way although
  it could hold: on the unit ball with a weight of 0.001, 61 of 64 problems returned the start, not done, with
  `violation` 0 (at 0.01 and above every problem reached its optimum). Leave the weight at 1 for clearances in
  metres against joint angles in radians; scale a row in other units with `weight` until its coefficients are of
  order one, and raise a weight where one term must win a conflict.
- **`closest`.** The step's merit charges the `closest` pairs selected again at the trial point, so a step that
  breaks more pairs than that at once is charged for those only (3.5 to 3.7 % of the accepted iterates on the Panda
  scenes at `closest=8`, 0.7 to 1.2 % on G1 at 64). The `violation` is exact over every pair. Raise `closest` where
  many pairs can break at once.
- **Fixed number of steps.** `optim.run(problem, [optim.Phase(optim.SequentialQP(), 50, stop_every=0)])` runs
  exactly 50 steps and reads nothing back to the host; `kinematics.compiled(solver.step)` compiles the step.
- **Dense problems only.** A `time=` problem needs `Problem(dense=True)`.

## Rows, a hinge cost or `solve_constrained`

Measured on the benchmark's inverse-kinematics protocol (Panda with fixed fingers and G1, one start, 256 targets,
float64, margin 0.02 m, eight threads of a shared host). Each cell is pose success over all targets / share of
targets with every clearance within 1e-4 m at the end / solver steps / seconds. `SequentialQP` and the hinge run 50
steps; `solve_constrained` runs LM at its defaults with `tolerance=1e-4`, the tolerance `SequentialQP` holds, until
it stops. Its last column is the time of the same run where the inequality terms keep their analytic Jacobian:
`solve_constrained` drops it and differentiates them by autodiff, which is most of its time. Almost every G1 target
configuration is in self-collision, so success there mostly counts poses that another configuration reaches.

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

`SequentialQP` reaches more poses than either other route on every scene and holds every clearance.
`solve_constrained` takes 60 to 300 steps; even with the analytic Jacobian (last column) it takes 1.6 to 7 times the
time of `SequentialQP`, except on the G1 floor, where it stops after 60 steps.

- A preference, or a constraint that may give a little: a weighted term and any solver.
- A box on a variable: `bounds=`, with any solver.
- Constraints that must hold at the end, at the cost of one quadratic program per step: constraint terms and
  `SequentialQP`.
- Constraints on a `time=` problem with banded normal equations, or with a solver other than the Gauss-Newton
  family: `solve_constrained`, which wraps any solver in an augmented-Lagrangian loop
  ([Hard constraints](hard-constraints.md)).

## Where next

- [Optimization](../concepts/optimization.md#constraint-terms): the rows, the acceptance rule and what was measured.
- [Collision](../concepts/collision.md): the sphere distances behind the clearance terms.
- [Hard constraints](hard-constraints.md): the augmented-Lagrangian route.
