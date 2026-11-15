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
`(steps, B)`: the largest distance of a constraint term's output from its bounds after each step, in that term's
unit, and 0 where every constraint holds.

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
constraint terms linearized at the iterate, started from the last step's solution.

| Argument | Default | Meaning |
|---|---|---|
| `damping` | `1e-3` | initial damping; it then follows the steps, as in `LevenbergMarquardt` |
| `gain` | `0.5` | share of a row's distance to its bound that one step may use, and of a violation that it repairs; `1.0` is the plain linearization |
| `tolerance` | `1e-4` | distance from a bound, in the term's unit, within which a point counts as holding the constraints |
| `gradient_tolerance`, `step_tolerance` | as LM | a problem is done once its iterate is within `tolerance` and one of them is met |

- **What you get back.** Linear rows miss the curvature of a constraint, so a step can land slightly outside one,
  and the next steps repair it. `result.x` is the last iterate that was within `tolerance` of every bound: a solve
  that starts there ends there after any number of steps, also on a target it cannot reach. Start from a
  configuration that holds the constraints; from one that does not, `result.x` is the closest iterate so far until
  one comes within `tolerance`.
- **Units.** `tolerance` and the `violation` diagnostic are in the units of the terms: metres for a clearance.
- **Weights.** A constraint term's `weight` scales its rows. Rows that cannot all hold give way where holding them
  costs most, counted per unit of weighted violation; leave the weight at 1 unless one term must win such a conflict.
- **Fixed number of steps.** `optim.run(problem, [optim.Phase(optim.SequentialQP(), 50, stop_every=0)])` runs
  exactly 50 steps and reads nothing back to the host; `kinematics.compiled(solver.step)` compiles the step.
- **Dense problems only.** A `time=` problem needs `Problem(dense=True)`.

## Rows, a hinge cost or `solve_constrained`

Measured on the benchmark's inverse-kinematics protocol (Panda with fixed fingers and G1, one start, float64, margin
0.02 m; `solve_constrained` at its defaults). Each cell is pose success over all targets / share of targets with
every clearance within 1e-4 m at the end / solver steps. Almost every G1 target configuration is in self-collision,
so the last row draws targets that are not.

| Scene (1024 targets) | `SequentialQP` | hinge cost, weight 1e4, LM | `solve_constrained`, LM |
|---|---|---|---|
| Panda, floor | 0.753 / 1.000 / 50 | 0.723 / 0.868 / 50 | 0.730 / 1.000 / 240 |
| Panda, obstacle sphere | 0.760 / 1.000 / 50 | 0.758 / 0.939 / 50 | 0.769 / 1.000 / 210 |
| Panda, self-collision | 0.766 / 1.000 / 50 | 0.732 / 0.854 / 50 | 0.755 / 1.000 / 300 |
| G1, floor | 0.751 / 1.000 / 50 | 0.595 / 0.995 / 50 | 0.629 / 1.000 / 60 (256 targets) |
| G1, obstacle sphere | 0.674 / 1.000 / 50 | 0.511 / 0.862 / 50 | not measured |
| G1, self-collision | 0.425 / 1.000 / 50 | 0.176 / 0.160 / 50 | 0.238 / 0.777 / 300 (256 targets) |
| G1, self-collision, targets drawn collision free | 0.722 / 1.000 / 50 | 0.453 / 0.461 / 50 | not measured |

On the Panda floor scene the three runs took 2.1 s, 1.3 s and 178 s on eight threads of a shared host.

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
