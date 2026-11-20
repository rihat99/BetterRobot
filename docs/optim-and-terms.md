# Optimization and terms

Inverse kinematics, trajectories and motion fits are all least-squares problems: find the variables that make a sum
of squared residuals small. `optim` solves such problems. It knows nothing about robots: a `Problem` holds named
tensors and residual functions, and a solver such as Levenberg-Marquardt, Gauss-Newton, L-BFGS or any `torch.optim`
optimizer changes the variables. `terms` adds the robot parts: pose targets, joint limits, smoothness, collision
clearance, and `solve_ik`, which puts them together for you.

The names you will use:

| Name | What it does |
|---|---|
| `br.solve_ik(model, targets, q0=)` | inverse kinematics for one or more frames, on a whole batch |
| `terms.ik_problem(model, targets, q0=)` | the same problem, returned so you can add your own residuals |
| `optim.Problem` | variables, data and residuals; `problem.residual(name, fn, weight=)` adds one |
| `optim.solve(problem, solver, iterations=)` | runs a solver and returns a `Result` with `x`, the solution |
| `optim.LevenbergMarquardt`, `GaussNewton`, `LBFGS`, `TorchOptim` | the solvers |
| `terms.frame_pose`, `terms.finite_difference`, `terms.RobotManifold` | robot residuals, smoothness, and how `q` moves |

## Inverse kinematics on a batch

`solve_ik` takes a target pose per frame and a start `q0`. With `seeds=8` it also tries 8 random starts per problem
and keeps the best.

```python
import torch
from robot_descriptions import panda_description

import better_robot as br
from better_robot import optim, terms

model = br.load(panda_description.URDF_PATH, dtype=torch.float64)
generator = torch.Generator().manual_seed(0)
q0 = model.q_neutral.expand(32, -1)
target = br.forward_kinematics(model, terms.random_start(model, q0, generator=generator)).frame_pose_of("panda_hand")

result = br.solve_ik(model, {"panda_hand": target}, q0=q0, seeds=8, generator=generator)
reached = br.forward_kinematics(model, result.x["q"]).frame_pose_of("panda_hand")
error = (reached[:, :3] - target[:, :3]).norm(dim=-1)
print(f"{int((error < 1e-4).sum())} of 32 reached, median error {float(error.median()):.1e} m")
```
```output
32 of 32 reached, median error 3.0e-13 m
```

## Add your own residual

`ik_problem` returns the problem before it is solved. A residual is a plain function of the variables; here one
asks the elbow joint to stay near -1.8 rad. `problem.losses` reports what each residual still costs at the
solution.

```python
problem = terms.ik_problem(model, {"panda_hand": target}, q0=q0)
problem.residual("elbow", lambda q: q[:, 3:4] + 1.8, weight=1e-3)

result = optim.solve(problem, optim.LevenbergMarquardt(), iterations=100)
for name, loss in problem.losses(result.x).items():
    print(f"{name}: {float(loss.mean()):.2e}")
```
```output
pose/panda_hand: 2.27e-02
elbow: 6.95e-04
```

## A smooth trajectory

A trajectory is a variable with a time axis. This one has 20 knots: the hand starts where it is, ends 25 cm away,
and velocity and acceleration residuals keep the motion smooth.

```python
knots, dt = 20, 0.05
q_start = torch.tensor([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785, 0.04], dtype=torch.float64)
start = br.forward_kinematics(model, q_start).frame_pose_of("panda_hand")
goal = start.clone()
goal[:3] += torch.tensor([0.10, 0.15, 0.15], dtype=torch.float64)

poses = start.repeat(1, knots, 1)  # (1, knots, 7)
poses[:, -1] = goal
ends = torch.zeros(knots, 1, dtype=torch.float64)
ends[0], ends[-1] = 1e6, 1e4  # only the first and last knot have a pose target

problem = optim.Problem(
    {"q": q_start.repeat(1, knots, 1)},
    manifolds={"q": terms.RobotManifold(model)},
    bounds={"q": (model.lower_position_limit, model.upper_position_limit)},
    data={"target": poses},
    time=("q",),
)
problem.cached(terms.kinematic_state(model), name="state")
problem.residual("ends", terms.frame_pose(model, frame="panda_hand", target="target"), weight=ends, window=1)
problem.residual("velocity", terms.finite_difference(model, order=1, dt=dt), weight=1e-3)
problem.residual("acceleration", terms.finite_difference(model, order=2, dt=dt), weight=1e-4)

result = optim.solve(problem, optim.LevenbergMarquardt(), iterations=200)
q = result.x["q"][0]  # (knots, nq)
reached = br.forward_kinematics(model, q[-1]).frame_pose_of("panda_hand")
print(f"goal error {float((reached[:3] - goal[:3]).norm()):.1e} m")
```
```output
goal error 1.4e-05 m
```

See also: [Collision](collision.md) for residuals that keep the robot clear of obstacles.
