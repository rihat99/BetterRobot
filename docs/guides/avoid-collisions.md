# Keep a trajectory clear of obstacles

A clearance term turns signed distances into residuals, so a trajectory problem bends a path around an obstacle.
This guide builds a two-link arm with three spheres, puts an obstacle on its straight path, and solves for a
path that keeps the tool sphere outside it; [Collision](../concepts/collision.md) explains the spheres and the
signs.

## Build the arm and its spheres

```python
import torch
from better_robot import ModelBuilder, collision, forward_kinematics, optim, terms
from better_robot.io import build_model

builder = ModelBuilder("arm", gravity=[0.0, 0.0, -9.81])
for body in ("base", "link1", "link2"):
    builder.add_body(body, mass=1.0)
builder.add_revolute("j1", axis=(0.0, 0.0, 1.0), parent="base", child="link1")
builder.add_revolute("j2", axis=(0.0, 1.0, 0.0), parent="link1", child="link2",
                       origin=torch.tensor([0.0, 0.0, 0.3, 0.0, 0.0, 0.0, 1.0]))
builder.add_frame("tool", parent="link2",
                  placement=torch.tensor([0.0, 0.0, 0.25, 0.0, 0.0, 0.0, 1.0]))
model = build_model(builder.finalize(), dtype=torch.float64)

spheres = collision.RobotSpheres.from_frames(model, {
    "base": [((0.0, 0.0, 0.05), 0.06)],
    "link2": [((0.0, 0.0, 0.12), 0.04)],
    "tool": [((0.0, 0.0, 0.0), 0.03)],
})
```

## Add a clearance term to a trajectory problem

`terms.sphere_clearance(spheres, distance_fn, margin=...)` builds a residual `margin - distance` for every sphere
and obstacle; `terms.self_clearance(spheres, pairs=pairs, margin=...)` does the same for sphere pairs of the robot
itself. Both are hinged at zero by default, so a clear configuration costs nothing, and both read the cached
kinematic state named `"state"`.

The example below swings `j1` from -0.9 to 0.9 rad. The straight path drives the tool through an
obstacle; a trajectory problem bends it around. [Optimization](../concepts/optimization.md) explains `Problem`,
`cached`, `window` and the solvers; `examples/06_obstacle_trajectory.py` is the full Panda version with
both soft and hard constraints.

```python
knots, dt, margin = 12, 0.1, 0.02
q_mid = torch.tensor([0.0, 0.5], dtype=torch.float64)
obstacle = forward_kinematics(model, q_mid).frame_translations[model.frame_id("tool")].clone()
obstacle_radius = torch.tensor(0.05, dtype=torch.float64)

def to_obstacle(centers, radii):
    return collision.sphere_sphere(centers, radii, center_b=obstacle, radius_b=obstacle_radius)

def clearance(q):
    return float(to_obstacle(spheres.centers(forward_kinematics(model, q)), spheres.radii).min())

alpha = torch.linspace(0, 1, knots, dtype=torch.float64)[:, None]
straight = (1 - alpha) * torch.tensor([-0.9, 0.5]) + alpha * torch.tensor([0.9, 0.5])
end_weight = torch.zeros(knots, 1, dtype=torch.float64)
end_weight[0] = end_weight[-1] = 1e3

problem = optim.Problem(
    {"q": straight[None].clone()},
    manifolds={"q": terms.RobotManifold(model)},
    data={"ends": straight.clone()[None]},
    time=("q",),
)
problem.cached(terms.kinematic_state(model), name="state")
problem.residual("ends", lambda q, ends: q - ends, weight=end_weight, window=1)
problem.residual("velocity", terms.finite_difference(model, order=1, dt=dt), weight=1e-2)
problem.residual("clearance", terms.sphere_clearance(spheres, to_obstacle, margin=margin),
                 weight=1e2, window=1)

result = optim.solve(problem, optim.LevenbergMarquardt(), iterations=100)
print(f"straight {clearance(straight):+.3f} m, optimized {clearance(result.x['q'][0]):+.3f} m")
# straight -0.070 m, optimized +0.019 m
assert clearance(result.x["q"][0]) > clearance(straight)
```
```output
straight -0.070 m, optimized +0.019 m
```

The clearance before optimization is negative: the tool sphere passes through the obstacle. After the
solve it sits just under the 0.02 m margin, because the soft weight trades clearance against smoothness. Raise the weight or switch to the
hard inequality to enforce it exactly.

<figure markdown="span"><video autoplay loop muted playsinline preload="metadata" poster="../../clips/obstacle_trajectory.png" width="960"><source src="../../clips/obstacle_trajectory.webm" type="video/webm"></video><figcaption>The Panda, modelled as spheres, follows a trajectory that a clearance term bends around a spherical obstacle (<code>examples/06_obstacle_trajectory.py</code>).</figcaption></figure>

## Enforce the margin exactly

With `hinge=False` the clearance residual is the inequality `margin - distance <= 0`, which
`optim.solve_constrained` enforces instead of trading it against smoothness; [Hard
constraints](hard-constraints.md) shows the call, and `examples/06_obstacle_trajectory.py` runs both forms on a
Panda.

## Where next

- [Collision](../concepts/collision.md): spheres, capsules, signed distances and the self-collision pairs.
- [Optimization](../concepts/optimization.md): `Problem`, `cached`, `window` and the solvers.
