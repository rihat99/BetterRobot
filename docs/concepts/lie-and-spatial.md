# Poses and twists are plain tensors with one layout

After this page you can compose, invert and apply poses, move between a twist and a pose with `exp` and
`log`, pick the right Jacobian of `exp`, and use the spatial value types `Motion`, `Force` and
`Inertia`. Every convention below is stated in the `better_robot.lie` and `better_robot.spatial`
module docstrings and checked by a snippet you can run.

![Poses are [t, q] with a scalar-last quaternion; twists are [v, ω], expressed in one of three frames.](../figures/lie_spatial.svg)

## One layout for poses, rotations and twists

| Object | Shape | Layout |
|--------|-------|--------|
| pose (SE(3) element) | `(..., 7)` | `[tx, ty, tz, qx, qy, qz, qw]`, quaternion scalar last |
| rotation (SO(3) element) | `(..., 4)` | `[qx, qy, qz, qw]` |
| twist (SE(3) tangent) | `(..., 6)` | `[vx, vy, vz, wx, wy, wz]`, linear first |
| SO(3) tangent | `(..., 3)` | `[wx, wy, wz]` |
| wrench (`Force`) | `(..., 6)` | `[fx, fy, fz, tx, ty, tz]`, linear first |
| spatial inertia (`Inertia`) | `(..., 10)` | `[mass, cx, cy, cz, Ixx, Iyy, Izz, Ixy, Ixz, Iyz]` |

All functions on tensors broadcast over leading batch axes and keep dtype and device. They are direct torch
formulas, so autograd flows through every one. The shape constructors `Inertia.from_sphere` / `from_box` /
`from_capsule` / `from_ellipsoid` take Python floats and build a float32 tensor unless you pass `dtype=`.

## `compose(a, b)` applies `b` first

`better_robot.lie.se3` and `better_robot.lie.so3` are modules of plain functions. `compose(a, b)` is
the matrix product $T_a T_b$: a point is moved by `b`, then by `a`. `act(T, p)` applies a pose to
points `p (..., 3)`, and `inverse(T)` undoes it.

```python
import torch
from better_robot.lie import se3, so3, tangents

yaw90 = torch.tensor([1.0, 0.0, 0.0, 0.0, 0.0, 0.7071068, 0.7071068])  # move +x, turn 90 deg about z
shift = torch.tensor([0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 1.0])              # move +y
print(se3.compose(yaw90, shift)[:3])    # tensor([-1., 0., 0.]): the shift is rotated, then offset
print(se3.act(yaw90, torch.tensor([1.0, 0.0, 0.0])))   # tensor([1., 1., 0.])

identity = se3.compose(yaw90, se3.inverse(yaw90))
assert torch.allclose(identity, se3.identity(), atol=1e-6)
```
```output
tensor([-1.0000e+00, -2.3842e-07,  0.0000e+00])
tensor([1.0000, 1.0000, 0.0000])
```

`se3` also provides `from_matrix`, `to_matrix`, `from_axis_angle`, `from_translation`, `normalize` and
`sclerp`; `so3` adds `from_euler`, `to_euler` and `slerp`. The value classes `SE3`, `SO3` and `Pose` in
`better_robot.lie` wrap one tensor and forward to these functions.

## `exp` and `log` are the true group maps

`se3.exp(xi)` maps a twist `xi (..., 6)` to a pose; `se3.log(T)` is its inverse. They are the real SE(3)
exponential and logarithm, not a separate translation plus rotation map: a twist with both linear and
angular parts traces a screw motion. `model.integrate` uses the same map for a free-flyer (see
[Joints, bodies and frames](joints-bodies-frames.md)).

```python
xi = torch.tensor([0.1, 0.2, 0.3, 0.4, -0.2, 0.1])
assert torch.allclose(se3.log(se3.exp(xi)), xi, atol=1e-6)
```

## Right Jacobians perturb on the right, left Jacobians on the left

`better_robot.lie.tangents` holds `hat_so3`/`vee_so3`, `hat_se3`/`vee_se3` and the Jacobians of `exp` with their inverses:
`right_jacobian_se3`, `right_jacobian_inv_se3`, `left_jacobian_se3`, `left_jacobian_inv_se3`, and the
`_so3` versions. For a small twist $\delta$:

$$
\log\!\big(\exp(\xi)\exp(\delta)\big) \approx \xi + J_r^{-1}(\xi)\,\delta,
\qquad
\log\!\big(\exp(\delta)\exp(\xi)\big) \approx \xi + J_l^{-1}(\xi)\,\delta,
$$

and $J_l(\xi) = J_r(-\xi)$. Use the right Jacobian when the perturbation is expressed in the local
(body) frame, the left one when it is expressed in the world frame.

```python
delta = 1e-6 * torch.tensor([1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
right = se3.log(se3.compose(se3.exp(xi), se3.exp(delta)))
left = se3.log(se3.compose(se3.exp(delta), se3.exp(xi)))
assert torch.allclose(right - xi, tangents.right_jacobian_inv_se3(xi) @ delta, atol=1e-7)
assert torch.allclose(left - xi, tangents.left_jacobian_inv_se3(xi) @ delta, atol=1e-7)
```

## The adjoint moves a twist to another frame

`se3.adjoint(T)` returns the `(..., 6, 6)` matrix

$$
\mathrm{Ad}(T) = \begin{bmatrix} R & \hat{p} R \\ 0 & R \end{bmatrix}
$$

for the linear-first twist order, where $R$ and $p$ are the rotation and translation of `T`. It
satisfies $T \exp(\xi) T^{-1} = \exp(\mathrm{Ad}(T)\,\xi)$. `se3.adjoint_inv(T)` is its inverse.

```python
lhs = se3.compose(se3.compose(yaw90, se3.exp(xi)), se3.inverse(yaw90))
assert torch.allclose(lhs, se3.exp(se3.adjoint(yaw90) @ xi), atol=1e-5)
```

!!! note "Frames of reference"
    A twist is always expressed in some frame. BetterRobot names three: `LOCAL` (the moving frame's own
    axes and origin), `WORLD` (world axes, world origin) and `LOCAL_WORLD_ALIGNED` (world axes, origin at
    the moving frame). Frame Jacobians default to `LOCAL_WORLD_ALIGNED`. Going from that to `LOCAL`
    needs only a rotation of both halves, not a full adjoint. See
    [Kinematics and Jacobians](kinematics-and-jacobians.md).

## Spatial value types wrap one tensor

`better_robot.spatial` provides `Motion` (a twist), `Force` (a wrench), `Inertia` and `Symmetric3`.
Each is a frozen dataclass around `.data`; every method returns a new value.

- `Motion.se3_action(T)` and `Force.se3_action(T)` change the frame of a twist or wrench; `T` is a
  `(..., 7)` tensor or an `SE3`.
- `Motion.cross_motion` and `Motion.cross_force` are the spatial cross products used by dynamics.
- `Inertia` stores mass, centre of mass `c` and the rotational inertia about `c`, both in the body
  frame. Build one with `from_sphere`, `from_box`, `from_capsule`, `from_ellipsoid`,
  `from_mass_com_sym3`, `from_mass_com_matrix` or `from_mesh`.
- `Inertia.apply(v)` returns the spatial momentum $I v$ as a `Force`.

```python
from better_robot.spatial import Force, Inertia, Motion

ball = Inertia.from_sphere(2.0, 0.1)   # 2 kg, 10 cm radius
print(ball.data)       # [2, 0, 0, 0, 0.008, 0.008, 0.008, 0, 0, 0]: I = 2/5 m r^2
momentum = ball.apply(Motion(xi))
print(type(momentum).__name__, momentum.linear)   # Force tensor([0.2000, 0.4000, 0.6000])
moved = Motion(xi).se3_action(yaw90)
assert torch.allclose(moved.data, se3.adjoint(yaw90) @ xi)
```
```output
tensor([2.0000, 0.0000, 0.0000, 0.0000, 0.0080, 0.0080, 0.0080, 0.0000, 0.0000,
        0.0000])
Force tensor([0.2000, 0.4000, 0.6000])
```

Model inertias use the same 10-vector: `model.body_inertias (nbodies, 10)` (see
[Model and state](model-and-state.md)). [Dynamics](dynamics.md) builds on these types.

## Where next

- [Kinematics and Jacobians](kinematics-and-jacobians.md): these types applied to a whole robot.
- [Dynamics](dynamics.md): spatial inertia and force in RNEA and CRBA.
