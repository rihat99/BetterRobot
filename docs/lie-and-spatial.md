# Lie and spatial

These two packages are the math under everything else. They work on plain tensors, with any leading batch axes.

`lie` handles rotations and rigid motions. A pose is a 7-vector `[tx, ty, tz, qx, qy, qz, qw]`, quaternion scalar
last. A small motion (a tangent, or twist) is a 6-vector `[linear, angular]`. `lie.se3` works on poses and
`lie.so3` on rotations alone.

`spatial` handles the 6D vectors of rigid-body dynamics: motions, forces and inertias. An inertia is stored as a
10-vector `[mass, com (3), rotational inertia (6)]`.

The names you will use:

| Name | What it does |
|---|---|
| `se3.compose`, `se3.inverse`, `se3.act` | chain two poses, invert one, move a point |
| `se3.exp`, `se3.log` | a twist to a pose and back |
| `se3.adjoint` | the 6x6 matrix that moves a twist from one frame to another |
| `so3.from_axis_angle`, `so3.to_matrix` | build and convert rotations |
| `spatial.inertia_of_sphere`, `inertia_of_box`, ... | the inertia of a simple shape |
| `spatial.inertia_matrix`, `transform_motion`, `transform_force` | 6x6 inertias, motions and forces between frames |

## Compose poses and move points

```python
import torch
from better_robot.lie import se3

turn = torch.tensor([1.0, 0.0, 0.0, 0.0, 0.0, 0.7071068, 0.7071068])  # 1 m along x, then 90 degrees about z
shift = torch.tensor([0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 1.0])               # 2 m along y
print(se3.compose(turn, shift)[:3])                  # the shift is turned first, then added
print(se3.act(turn, torch.tensor([1.0, 0.0, 0.0])))  # where the point (1, 0, 0) lands
```
```output
tensor([-1.0000e+00, -2.3842e-07,  0.0000e+00])
tensor([1.0000, 1.0000, 0.0000])
```

## Exp and log

`exp` turns a twist into a pose; `log` gives the twist back. Both take a batch.

```python
twists = 0.3 * torch.randn(5, 6)
poses = se3.exp(twists)
print(poses.shape, torch.allclose(se3.log(poses), twists, atol=1e-5))
```
```output
torch.Size([5, 7]) True
```

## Inertias and spatial vectors

```python
from better_robot import spatial

ball = spatial.inertia_of_sphere(torch.tensor(2.0), radius=torch.tensor(0.1))  # 2 kg, 10 cm radius
print(ball)
twist = torch.tensor([0.1, 0.2, 0.3, 0.0, 0.0, 1.0])
print(spatial.inertia_matrix(ball) @ twist)  # the momentum of the ball moving with this twist
```
```output
tensor([2.0000, 0.0000, 0.0000, 0.0000, 0.0080, 0.0080, 0.0080, 0.0000, 0.0000,
        0.0000])
tensor([0.2000, 0.4000, 0.6000, 0.0000, 0.0000, 0.0080])
```

See also: [Kinematics](kinematics.md), which uses these functions to place every frame of a robot.
