# Enforce hard constraints

A weighted term is a soft wish: the solver trades it against everything else. When a term must hold exactly
(`c(x) = 0`) or must never be exceeded (`c(x) <= 0`), solve with `optim.solve_constrained`. After this guide you
can turn any registered term into a hard constraint: first on a 2-D toy problem, then on a Panda arm, then on a
whole trajectory that must clear an obstacle.

## Constraints are existing terms, named at solve time

Constraints are ordinary residual terms already registered on the `Problem`. You name them when you call the
solver; nothing about the terms changes. Their weight and kernel are ignored while they act as constraints.

The toy problem: get as close as possible to the point `(1, 1)` while staying on the unit circle (equality) and
below the line `y = 0.5` (inequality). The answer is `(√0.75, 0.5) ≈ (0.866, 0.5)`.

```python
import torch
from better_robot import optim

start = torch.tensor([[2.0, -1.0]], dtype=torch.float64)          # p (B=1, 2)
problem = optim.Problem({"p": start}, data={"target": torch.tensor([[1.0, 1.0]], dtype=torch.float64)})
problem.residual("close", lambda p, target: p - target)
problem.residual("circle", lambda p: (p * p).sum(-1, keepdim=True) - 1.0)
problem.residual("below", lambda p: p[:, 1:2] - 0.5)

result = optim.solve_constrained(
    problem, optim.LevenbergMarquardt(), equalities=("circle",), inequalities=("below",), tolerance=1e-8
)
print(result.x["p"])                      # ≈ [[0.866, 0.500]]
print(result.history["violation"][-1])    # largest violation per batch element, (B,)
```
```output
tensor([[0.8660, 0.5000]], dtype=torch.float64)
tensor([4.7526e-09], dtype=torch.float64)
```

`result.loss` is the loss of the other terms only (here `close`). `result.history["violation"]` is `(rounds, B)`:
the largest constraint violation of each batch element after each outer round.

## An augmented Lagrangian turns constraints into a sequence of soft solves

`solve_constrained` runs an augmented-Lagrangian loop around any `Solver`. Each outer round replaces every
constraint $c$ by the least-squares term

$$\tfrac{1}{2}\,\rho\,\lVert c + \lambda/\rho \rVert^2,$$

runs the solver for `iters` steps, then updates the multipliers $\lambda$. For an inequality the shifted residual
is clipped at zero, so it only pushes when the constraint is active. The penalty $\rho$ is kept per batch element:
it starts at `penalty` and grows by `growth` whenever that element's violation did not fall to a quarter of the
previous round. The loop stops when every batch element's largest violation is below `tolerance`, or after
`outer` rounds.

| Argument | Default | Meaning |
|---|---|---|
| `equalities`, `inequalities` | `()` | names of residual terms read as `c = 0` and `c <= 0` |
| `outer` | `10` | maximum augmented-Lagrangian rounds |
| `iters` | `30` | solver steps per round |
| `penalty`, `growth` | `10.0`, `10.0` | initial $\rho$ and its growth factor |
| `tolerance` | `1e-6` | stop when every element's largest violation is below this |
| `x` | `None` | start values; defaults to the problem's variables |

A name that is not a residual term of the problem raises `KeyError`. So does calling it with no constraints.

!!! note "Units matter"
    `tolerance` compares raw residual values, so it is in the units of the term: metres for a position, radians
    for an angle. Scale a term if its natural units are far from the others.

## A robot reaches exactly and keeps its elbow under a ceiling

`examples/05_hard_constraints.py` keeps a Panda close to its start pose (the cost), puts the hand on a goal point
exactly (equality) and keeps the elbow at least 3 cm lower than at the start (inequality), for a batch of goals.

```python
import better_robot as br
from robot_descriptions import panda_description
from better_robot import terms
from better_robot.kinematics import forward_kinematics_state

model = br.load(panda_description.URDF_PATH, dtype=torch.float64)
hand, elbow = model.frame_id("body_panda_hand"), model.frame_id("body_panda_link4")
q_start = torch.tensor([[0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785, 0.02, 0.02]], dtype=torch.float64)
q_start = q_start[:, : model.nq].expand(8, -1)
start = forward_kinematics_state(model, q_start).frame_pose
goal = start[:, hand, :3] + 0.15 * (torch.rand(8, 3, dtype=torch.float64) - 0.5)
ceiling = start[:, elbow, 2:3] - 0.03

problem = optim.Problem(
    {"q": q_start},
    manifolds={"q": terms.RobotManifold(model)},
    bounds={"q": (model.lower_pos_limit, model.upper_pos_limit)},
    data={"goal": goal, "ceiling": ceiling, "q_start": q_start},
)
problem.cached(terms.kinematic_state(model), name="state")
problem.residual("stay", lambda q, q_start: q - q_start)
problem.residual("reach", terms.frame_position(model, frame="body_panda_hand", target="goal"))
problem.residual("under", lambda state, ceiling: state.frame_pose[:, elbow, 2:3] - ceiling)

hard = optim.solve_constrained(
    problem, optim.LevenbergMarquardt(), equalities=("reach",), inequalities=("under",), tolerance=1e-8
)
pose = forward_kinematics_state(model, hard.x["q"]).frame_pose
print("worst reach error:", float((pose[:, hand, :3] - goal).norm(dim=-1).max()))
print("elbow over ceiling:", float((pose[:, elbow, 2:3] - ceiling).max()))
```
```output
worst reach error: 7.27354373075532e-08
elbow over ceiling: 9.844031578776224e-08
```

The `state` value is computed once per evaluation by `problem.cached` and shared by every term that names it, so the
constraint `under` reads the same forward-kinematics pass as `reach`. Run the full example to compare against a
penalty-only solve:

```bash
uv run python examples/05_hard_constraints.py
```

## A clearance inequality keeps a whole trajectory out of an obstacle

The same call scales to time variables. In `examples/06_obstacle_trajectory.py` the Panda is a set of spheres and
the obstacle is one sphere. The straight joint-space path sweeps the hand through it; the clearance
$d(q_t) \ge \text{margin}$, written as the inequality $\text{margin} - d(q_t) \le 0$ for every knot $t$, pushes the
path around it. The example solves it twice, once with the clearance as a weighted hinge penalty and once as a hard
inequality, so you can compare them.

<figure markdown="span"><video autoplay loop muted playsinline preload="metadata" poster="../../clips/obstacle_trajectory.png" width="960"><source src="../../clips/obstacle_trajectory.webm" type="video/webm"></video>
<figcaption>The Panda's trajectory around a spherical obstacle (examples/06).</figcaption></figure>

```bash
uv run python examples/06_obstacle_trajectory.py
```

## Use a weight instead when the term is a preference

- The target is a preference, not a requirement: use a weighted term and `optim.solve`.
- The constraint is a box on a variable: pass `bounds=` to `Problem`; every solver step is clamped to it.
- The constraint is infeasible for some batch elements: those elements run all `outer` rounds and their $\rho$
  keeps growing. Check `result.history["violation"][-1]` rather than assuming success.

## Where next

- [Optimization](../concepts/optimization.md): `Problem`, terms and solvers.
- [Collision](../concepts/collision.md): the sphere distances behind the clearance term.
- [Write a term](write-a-term.md): the term signature.
