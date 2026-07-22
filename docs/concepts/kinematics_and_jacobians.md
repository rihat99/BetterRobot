# Kinematics and Jacobians

Kinematics describes motion without asking which forces caused it. Forward
kinematics starts with joint configuration values and computes poses. A
Jacobian describes how those poses change under a small joint motion.

These two operations appear throughout robotics. A viewer needs poses. An IK
solver needs poses and their Jacobians. A dynamics pass needs the same joint
transforms before it can propagate velocities and forces.

## Forward kinematics: joints in, poses out

A configuration `q` stores the position of every independent joint. Forward
kinematics, usually shortened to **FK**, walks from each parent joint to its
children and composes transforms along the tree.

```{testcode}
import torch
from better_robot import forward_kinematics
from better_robot.io import ModelBuilder, build_model

builder = ModelBuilder("one_joint_kinematics")
builder.add_body("base")
builder.add_body("link")
builder.add_revolute_z("joint", parent="base", child="link", lower=-3.14, upper=3.14)
model = build_model(builder.finalize(), dtype=torch.float64)
q = model.q_neutral
data = forward_kinematics(model, q, compute_frames=True)
print("joint poses:", tuple(data.joint_pose_world.shape))
print("frame poses:", tuple(data.frame_pose_world.shape))
```

```{testoutput}
joint poses: (3, 7)
frame poses: (3, 7)
```

The final axis of each pose is
`[tx, ty, tz, qx, qy, qz, qw]`. Leading axes are batch axes. FK does not move
the robot or search for a configuration; it answers where the model would be
for the `q` it was given.

Every joint supplies a transform from its parent at the current joint slice.
The local pose composes its fixed placement with the current joint transform;
the world pose then composes that result with its parent's world pose. Those
poses feed the frame-Jacobian pass directly:

```{testcode}
from better_robot.kinematics import compute_joint_jacobians, get_frame_jacobian

compute_joint_jacobians(model, data)
frame_id = model.frame_id("body_link")
frame_jacobian = get_frame_jacobian(model, data, frame_id)
print("frame Jacobian shape:", tuple(frame_jacobian.shape))
```

```{testoutput}
frame Jacobian shape: (6, 1)
```

A fixed-base and a floating-base model use the same recurrence. The latter
simply has a free-flyer near the root. That decision is explained in
{ref}`decision-unified-base`.

## Joints, bodies, and frames

A joint pose belongs to the kinematic tree. A frame is a named coordinate
system attached to a joint by a fixed transform: a tool tip, camera mount,
sensor, or body frame. Once joint poses are known, frame placement is a
batched gather and composition.

Pass `compute_frames=True` when frame poses are needed immediately. The same
work can be requested later with `update_frame_placements(model, data)`.
Frame metadata lives on `Model`; computed poses live on `Data`.

## What a Jacobian says

A Jacobian answers a local “what if?” question:

> If the input moves a tiny amount, which way and how far does the output
> move?

For a frame motion `x` and a small tangent step `delta_v`, the first-order
relationship is:

```{math}
\delta x \approx J(q)\,\delta v.
```

Each Jacobian column is the output motion caused by one input tangent
coordinate. Each row is one component of the output. A spatial frame Jacobian
has six rows: three linear and three angular. Its columns number `nv`, not
necessarily `nq`, because quaternion configurations take steps in a smaller
tangent space.

The approximation is local. A large motion changes both the configuration and
the Jacobian, so an optimizer repeatedly evaluates, steps, and evaluates
again.

Lynch and Park's
[Modern Robotics](https://hades.mech.northwestern.edu/index.php/Modern_Robotics)
develops the geometric Jacobian from screw motions.

## Three ways to obtain a Jacobian

There are three common methods. They answer the same derivative question but
make different engineering trades.

### An analytic formula

An analytic Jacobian is written from a mathematical derivative. It is usually
the fastest choice because it can reuse quantities already computed by the
forward pass. Exact formulas also avoid the step-size error of numerical
differences.

“Analytic” describes how the block was written, not an automatic guarantee of
exactness. BetterRobot's pose and frame Jacobians use full Lie-group formulas.
Velocity and higher-order smoothness expose constant blocks only for affine
all-scalar robot topologies; manifold models use dense AD. The reference
trajectory residual still uses a documented small-step identity-Jacobian
approximation. That approximation is useful near its reference, but it should
not be described as an exact derivative far from it.

### Automatic differentiation

Automatic differentiation, or **AD**, records elementary operations and
applies the chain rule. It is exact up to floating-point arithmetic and needs
a differentiable forward implementation. It is neither symbolic algebra nor
finite differences.

When a residual does not provide a Jacobian block, `Problem` uses
`torch.func.jacrev` or `jacfwd` according to the residual and tangent widths.
This is the default way for custom differentiable residuals to become usable
without a hand-derived formula. PyTorch's
[autograd documentation](https://docs.pytorch.org/docs/stable/autograd)
describes the underlying differentiation system.

### Finite differences

Finite differences evaluate the function after small positive and negative
steps and divide the change by the step size. They work even when no analytic
or AD path is available, but they are approximate, slow, and sensitive to the
chosen step.

Robot finite differences must perturb through the variable's retraction. A
quaternion component cannot be nudged as if it were an unconstrained scalar.
BetterRobot keeps finite differences as an explicit debugging cross-check,
not an automatic production fallback.

In short: use a verified analytic block when one exists, otherwise use AD for
a differentiable residual, and use finite differences to check the other two.

## The robot Jacobian API

The public functions share one underlying assembly:

- `compute_joint_jacobians(model, data)` populates every joint Jacobian with
  shape `(..., njoints, 6, nv)`.
- `get_joint_jacobian(model, data, joint_id, reference=...)` returns one
  joint's `(..., 6, nv)` block.
- `get_frame_jacobian(model, data, frame_id, reference=...)` shifts the
  result to an attached frame.

Joint poses must already be present in `data`, so call FK first. The cache
level on `Data` turns an out-of-order read into a direct error instead of
returning a stale tensor.

## World, local, and local-world-aligned

A spatial velocity needs both a point of reference and coordinate axes. The
`reference=` argument accepts:

- `"local_world_aligned"`: velocity at the frame origin, expressed in world
  axes;
- `"local"`: velocity at the frame origin, expressed in frame axes; or
- `"world"`: velocity expressed in world axes and referenced at the world
  origin.

`get_frame_jacobian` defaults to local-world-aligned because it directly says
how the frame origin and orientation move in world coordinates. This is the
usual quantity for end-effector position and pose errors. The full decision is
{ref}`decision-jacobian-frame`.

One conversion deserves special care. Local-world-aligned and local
Jacobians refer to the same frame origin, so converting between them only
rotates the linear and angular rows. A `world` Jacobian refers to a different
point; converting it to `local` uses the full SE(3) adjoint, including the
translation cross term. Applying that full adjoint to an already aligned
frame-origin Jacobian adds a false term.

## A pose residual combines two Jacobians

A pose residual compares a target pose with a frame pose:

```{math}
r(q) = \log\left(T_{target}^{-1} T_{frame}(q)\right).
```

Its derivative combines the frame Jacobian with the inverse right Jacobian of
the SE(3) logarithm:

```{math}
J_r(q) = J_{right}^{-1}(r(q))\,J_{frame,local}(q).
```

The first factor accounts for the curved pose error; the second tells how the
robot moves the frame. Omitting the logarithm Jacobian couples translation and
rotation incorrectly when the error is not tiny.

`PoseResidual`, `PositionResidual`, and `OrientationResidual` expose verified
analytic blocks built from this path. Other residuals can let the problem
differentiate their forward calls.

## Raw results and mutable Data

The tensor-only functions return frozen result records:

- `forward_kinematics_raw` returns world and local joint placements.
- `frame_placements_raw` returns world frame placements.
- `joint_jacobians_raw` returns all joint Jacobians.
- `frame_jacobian_raw` returns one frame Jacobian in the requested reference frame.

These functions do not mutate `Data`, which makes them convenient inside a
larger differentiable function. Public wrappers validate call inputs and fill
or return the familiar workspace. The mathematical computation is shared;
the difference is who owns the result storage.

## Batches, dtype, and device

FK and Jacobians preserve arbitrary compatible leading batch axes. The model
and query tensors must use the same device and floating dtype. Supported
working dtypes are `float32` and `float64`; lower precision is not part of the
contract because pose composition and Jacobian formulas accumulate error
along the tree.

The ordinary Torch pass is the reference implementation. FK also has an
optional fused Warp lane for eligible CUDA inputs. Both keep the same public
tensors and are compared for value and gradient parity. See
{doc}`the_compute_seam`.

## Common mistakes

- Passing degrees where a joint expects radians.
- Treating quaternion storage width `nq` as tangent width `nv`.
- Reading frame poses without requesting or updating frame placements.
- Comparing quaternion tensors without accounting for `q` and `-q`.
- Applying a full adjoint to a local-world-aligned Jacobian.
- Using finite differences as a silent fallback instead of a diagnostic.

## Where to continue

- {doc}`lie_and_spatial` explains SE(3), tangents, and logarithm Jacobians.
- {doc}`dynamics` adds velocities, accelerations, forces, and inertias.
- {doc}`residuals_costs_and_solvers` shows how Jacobians drive least-squares
  optimization.
- {doc}`/guides/differentiate_through_kinematics` gives the practical autograd
  recipe.
