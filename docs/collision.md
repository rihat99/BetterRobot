# Collision

`collision` covers a robot with spheres and measures signed distances: from the robot to itself, to the floor, and
to obstacles. A distance is positive when apart and negative when overlapping. The functions are plain tensor code,
so distances are batched and differentiable, and `terms.sphere_clearance` turns them into residuals for an
optimization.

The names you will use:

| Name | What it does |
|---|---|
| `collision.load_spheres(path, model)` | the spheres of a sphere file, and the frame pairs that may touch |
| `collision.SPHERE_SETS` | the folder of shipped sphere files (`panda.json`, `g1.json`) |
| `collision.RobotSpheres` | spheres fixed to frames; `spheres.centers(state)` puts them in the world |
| `collision.self_collision_pairs`, `self_distances` | which sphere pairs to check, and their distances |
| `collision.sphere_halfspace`, `sphere_sphere`, `sphere_box`, ... | distances to simple obstacles |
| `terms.sphere_clearance(spheres, distance_fn, margin=)` | a residual that pushes spheres out to `margin` |

## Self-collision

```python
import torch
from robot_descriptions import panda_description

import better_robot as br
from better_robot import collision

model = br.load(panda_description.URDF_PATH, dtype=torch.float64)
spheres, ignore = collision.load_spheres(collision.SPHERE_SETS / "panda.json", model)
pairs = collision.self_collision_pairs(model, spheres, ignore=ignore)
ready = torch.tensor([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785, 0.04], dtype=torch.float64)
distances = collision.self_distances(spheres, br.forward_kinematics(model, ready), pairs=pairs)
print(spheres, tuple(pairs.shape))
print(f"closest pair: {1e3 * float(distances.min()):.0f} mm")
```
```output
RobotSpheres(42 spheres, float64, cpu) (294, 2)
closest pair: 92 mm
```

## Distance to the floor

Leave out the spheres of the base, which stands on the floor, and measure the others against the plane `z = 0`.

```python
moving = spheres.frame_ids != model.frame_id("panda_link0")
arm = collision.RobotSpheres(frame_ids=spheres.frame_ids[moving], offsets=spheres.offsets[moving], radii=spheres.radii[moving])

def floor(centers, radii):
    up = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float64)
    return collision.sphere_halfspace(centers, radii, normal=up, offset=0.0)

print(f"lowest sphere: {float(floor(arm.centers(br.forward_kinematics(model, ready)), arm.radii).min()):.3f} m")
```
```output
lowest sphere: 0.116 m
```

## Keep inverse kinematics off the floor

Ask the hand to go 5 cm below the floor. With a clearance residual, the solver stops it just above.

```python
from better_robot import optim, terms

target = br.forward_kinematics(model, ready).frame_pose_of("panda_hand").clone()
target[2] = -0.05
problem = terms.ik_problem(model, {"panda_hand": target[None]}, q0=ready[None])
problem.residual("floor", terms.sphere_clearance(arm, floor, margin=0.02), weight=100.0)
result = optim.solve(problem, optim.LevenbergMarquardt(), iterations=100)

lowest = floor(arm.centers(br.forward_kinematics(model, result.x["q"])), arm.radii).min()
print(f"lowest sphere after IK: {float(lowest):.3f} m")
```
```output
lowest sphere after IK: 0.009 m
```

See also: [Optimization and terms](optim-and-terms.md) for building problems.
