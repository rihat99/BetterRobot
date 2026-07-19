# Rotations, Poses, and Spatial Motion

Robot motion is not ordinary vector addition. A position can be shifted by a
three-dimensional vector, but an orientation lives on a curved space: adding
two sets of angles does not, in general, compose two rotations. BetterRobot
uses Lie groups to represent that geometry while keeping the implementation in
ordinary Torch tensors.

This chapter starts with the picture behind the terms. The storage and API
details come afterward.

## A rotation quaternion in plain words

A three-dimensional rotation has three degrees of freedom. BetterRobot stores
it as a **unit quaternion** with four numbers:

```text
[qx, qy, qz, qw]
```

The extra number is constrained by `qx² + qy² + qz² + qw² = 1`, so it does
not add a fourth degree of freedom. Quaternions avoid the gimbal lock of Euler
angles and compose rotations efficiently. They also have a double cover:
`q` and `-q` describe the same physical rotation.

The set of all three-dimensional rotations is called **SO(3)**. The name is
less important than the rule: values on SO(3) must be composed and compared as
rotations, not treated as unconstrained four-vectors.

For a gentle introduction, see Lynch and Park's
[Modern Robotics](https://hades.mech.northwestern.edu/index.php/Modern_Robotics)
and Joan Solà's
[quaternion notes](https://arxiv.org/abs/1711.02508).

## A pose and SE(3)

A rigid-body pose combines a translation and a rotation. The set of all such
poses is called **SE(3)**. BetterRobot stores one pose as seven numbers:

```text
[tx, ty, tz, qx, qy, qz, qw]
```

If `T_world_tool` is a pose, it answers both “where is the tool origin?” and
“how are the tool axes oriented?” Pose composition follows frames: composing
`T_world_parent` with `T_parent_child` produces `T_world_child`.

SO(3) and SE(3) are Lie groups. In practical robotics language, that means
they have two useful views:

- a curved value space for complete rotations or poses; and
- a flat local space for small changes around one value.

Solà, Deray, and Atchuthan's
[micro Lie theory](https://arxiv.org/abs/1812.01537) develops this connection
with robotics examples and formula tables.

## Tangents and twists

A **tangent** is a small local change expressed in a flat vector space. An
SO(3) tangent has three angular components. An SE(3) tangent has six
components: three linear and three angular. A motion tangent is often called a
**twist**.

BetterRobot stores an SE(3) tangent as:

```text
[vx, vy, vz, wx, wy, wz]
```

The exponential map `exp` turns a tangent into a rotation or pose change. The
logarithm `log` turns a nearby rotation or pose back into a tangent. An
optimizer uses the same idea through a retraction: compute a step in the flat
tangent space, then move the configuration back onto its manifold.

This explains why a free-flyer has `nq=7` but `nv=6`. Its pose needs seven
stored numbers because rotation uses a quaternion. Its local motion still has
only six degrees of freedom. BetterRobot uses `nq` for stored configuration
width and `nv` for tangent or velocity width throughout the library.

## Storage conventions

The layouts are fixed:

| Object | Shape | Last-axis order |
|---|---|---|
| SO(3) quaternion | `(..., 4)` | `[qx, qy, qz, qw]` |
| SO(3) tangent | `(..., 3)` | `[wx, wy, wz]` |
| SE(3) pose | `(..., 7)` | `[tx, ty, tz, qx, qy, qz, qw]` |
| SE(3) tangent or twist | `(..., 6)` | `[vx, vy, vz, wx, wy, wz]` |
| Spatial Jacobian | `(..., 6, nv)` | linear rows, then angular rows |

Quaternion scalars are last; spatial linear components come first. These are
ecosystem choices rather than mathematical necessities. The alternatives and
interchange argument are recorded in {ref}`decision-pose-layout`.

Euler angles are an interchange format, not internal storage.
`so3.from_euler` and `so3.to_euler` use active roll, pitch, yaw rotations with
`R = Rz(yaw) @ Ry(pitch) @ Rx(roll)`. Euler angles necessarily become
ambiguous at pitch `+/- pi/2`.

## Functional Lie operations

The `better_robot.lie.se3` and `better_robot.lie.so3` modules contain free
functions over `torch.Tensor` values:

```{testcode}
import torch
from better_robot.lie import se3, so3

T_a = se3.identity(dtype=torch.float64)
T_b = se3.exp(torch.tensor([0.1, 0.0, 0.0, 0.0, 0.2, 0.0], dtype=torch.float64))
T_ab = se3.compose(T_a, T_b)
T_ba = se3.inverse(T_ab)
points_b = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64)
points_a = se3.act(T_ab, points_b)

xi = se3.log(T_ab)
T_again = se3.exp(xi)

R = so3.to_matrix(T_ab[..., 3:])
q = so3.from_matrix(R)
```

The main operations are `identity`, `compose`, `inverse`, `act`, `exp`,
`log`, `normalize`, `from_matrix`, and `to_matrix`. They broadcast leading
batch axes and preserve dtype and device.

The implementation does not subclass `torch.Tensor`. Optional typed wrappers
such as `SE3`, `SO3`, and `Pose` contain a tensor and delegate to the same
functions. Core algorithms store raw tensors on `Model` and `Data`, which
keeps autograd, `torch.func`, and `torch.compile` on a direct Torch graph.
See {ref}`decision-functional-lie` for why this was chosen over tensor
subclasses and per-operation dispatch.

## Exponential and logarithm Jacobians

Differentiating `exp` and `log` on a curved group introduces left and right
Jacobians. BetterRobot exposes these in `better_robot.lie.tangents`:

```{testcode}
import torch
from better_robot.lie import tangents

xi = torch.tensor([0.1, 0.0, 0.0, 0.0, 0.2, 0.0], dtype=torch.float64)
Jr = tangents.right_jacobian_se3(xi)
Jr_inv = tangents.right_jacobian_inv_se3(xi)
Jl = tangents.left_jacobian_se3(xi)
```

These are not robot frame Jacobians. They describe how a perturbation passes
through a Lie-group exponential or logarithm. A pose residual composes them
with a frame Jacobian; {doc}`kinematics_and_jacobians` shows that example.

## Singularities and numerical branches

Closed-form SO(3) and SE(3) formulas contain ratios that appear to divide by
zero near a zero rotation. The mathematical limit is well defined, so the
implementation uses Taylor series in that region. Both branches receive safe
denominators because `torch.where` may evaluate their gradients even when one
branch is not selected.

Near a rotation of pi, the principal logarithm has a branch boundary. The two
quaternion signs are first folded to a common hemisphere, but no representation
can make the principal logarithm smooth across every possible rotation.
Optimization code should keep adjacent trajectory quaternions on a consistent
hemisphere and avoid comparing quaternion storage directly.

For equality, compare rotation matrices or use the norm of a relative
SO(3) logarithm. `torch.testing.assert_close(q1, q2)` is not a physical
rotation comparison because `q` and `-q` are equivalent.

## Spatial motion, force, and inertia

Spatial algebra packages a rigid body's linear and angular quantities into
six-dimensional objects:

- `Motion` stores linear and angular velocity or acceleration.
- `Force` stores force and torque.
- `Inertia` stores mass, center of mass, and rotational inertia as one
  operator from motion to force.

The wrappers validate their event shapes and provide operations whose meaning
is unambiguous. A motion can act on another motion through the spatial cross
product; its dual action on a force uses the corresponding force cross
operator. An inertia multiplies a motion to produce a force.

The convention is always linear first. This matches the rest of BetterRobot's
Jacobians and avoids a reorder at every dynamics boundary. Roy Featherstone's
[spatial-vector resources](https://royfeatherstone.org/) are the primary
reference for the six-dimensional algebra and the dynamics algorithms built
on it.

## Matrices, alignment, and typed views

`se3.from_matrix` and `se3.to_matrix` convert homogeneous `(..., 4, 4)`
matrices without changing the scalar-last pose convention. The Umeyama helper
fits a batched similarity transform between point sets:

```{testcode}
import torch
from better_robot.lie import umeyama

source = torch.tensor(
    [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
    dtype=torch.float64,
)
target = 2.0 * source + torch.tensor([0.5, -0.2, 0.3], dtype=torch.float64)
weights = torch.ones(3, dtype=torch.float64)
scale, rotation, translation = umeyama(source, target, weights)
```

The fitted rotation is proper: its determinant is `+1`. This is a data
alignment helper, not a replacement for pose composition.

Typed wrappers are useful at application boundaries where `SE3` conveys more
meaning than “tensor ending in seven.” Raw algorithm storage remains tensor
based so thousands of poses do not become thousands of Python objects.

## Practical rules

- Normalize quaternions before they enter a long computation. Public FK can
  optionally diagnose clearly invalid free-flyer norms, but it does not
  normalize every call.
- Do not add or linearly interpolate quaternion components. Use SO(3)
  composition, logarithms, or spherical interpolation.
- Keep frame direction in the variable name: `T_world_tool` is easier to use
  correctly than `pose`.
- Remember that `nq` is storage width and `nv` is tangent width.
- Do not confuse Lie exponential Jacobians with robot frame Jacobians.

## Where to continue

- {doc}`joints_bodies_frames` applies these manifolds to every joint kind.
- {doc}`kinematics_and_jacobians` composes joint transforms and explains robot
  Jacobians.
- {doc}`dynamics` uses spatial motion, force, and inertia.
- {doc}`the_compute_seam` explains how the tensor functions participate in a
  whole-pass implementation.
