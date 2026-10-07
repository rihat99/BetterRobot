# Load your first robot model

You load the Franka Panda arm from its URDF, read its structure, and build a small robot in code.
After this page you know what a `Model` holds and how its configuration vector is laid out.

## Load the Panda from a URDF

`br.load` reads a URDF or MJCF file (or a `ModelBuilder`) and returns a `Model`. The tutorials use
the Panda that ships with `robot_descriptions`, in `float64` so that solver errors print cleanly:

```python
import torch
import better_robot as br
from robot_descriptions import panda_description

model = br.load(panda_description.URDF_PATH, dtype=torch.float64)
print(model.name, "nq =", model.nq, "nv =", model.nv)
```
```output
 nq = 8 nv = 8
```

A `Model` is immutable. Its topology (joint tree, names, index tables) is static Python data; its numbers
(placements, inertias, limits) are tensors on one device and dtype. See
[Model and state](../concepts/model-and-state.md) for the split.

![Model, values and state: ModelStructure holds the static topology, ModelValues the differentiable tensors, and each fk query returns a frozen KinematicState.](../figures/model_values_state.svg){ width="860" }

*What `br.load` builds (left, middle) and what every query returns (right).*

## Read the joints and frames

Joint 0 is always `universe`, the fixed world. The URDF parser adds a fixed `root_joint` that attaches the
robot to it. Every body gets a frame named `body_<link>`; you address frames by name:

```python
print(model.njoints, model.joint_names[:4])
print(model.nframes, model.frame_names[-3:])
hand = model.frame_id("body_panda_hand")
print("hand frame id:", hand)
```
```output
14 ('universe', 'root_joint', 'panda_joint1', 'panda_joint2')
14 ('body_panda_hand_tcp', 'body_panda_leftfinger', 'body_panda_rightfinger')
hand frame id: 10
```

Frame names are prefixed with their kind, so the hand link becomes `body_panda_hand`.

## Read the configuration vector

The configuration `q (nq,)` stacks one slice per joint. The Panda has seven revolute arm joints and one
finger joint; the second finger mimics the first, so it does not get its own entry and `nq = 8`.

```python
print(model.q_neutral)
print(model.lower_pos_limit)
print(model.upper_pos_limit)
q = model.q_neutral.clamp(model.lower_pos_limit, model.upper_pos_limit)
```
```output
tensor([0., 0., 0., 0., 0., 0., 0., 0.], dtype=torch.float64)
tensor([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973,  0.0000],
       dtype=torch.float64)
tensor([ 2.8973,  1.7628,  2.8973, -0.0698,  2.8973,  3.7525,  2.8973,  0.0400],
       dtype=torch.float64)
```

`q_neutral` is all zeros here, which lies outside the limits of joint 4, so you clamp it before use.
For a fixed-base arm `nq == nv`. A floating base breaks that equality; [Floating-base robots](floating-base.md)
shows why.

## Build a robot in code

When you have no URDF, `ModelBuilder` describes bodies and joints directly. This is a one-joint pendulum:

```python
builder = br.ModelBuilder("pendulum")
builder.add_body("base", mass=0.0)
builder.add_body(
    "link",
    mass=2.0,
    com=torch.tensor([0.3, 0.0, 0.0]),
    inertia=torch.diag(torch.tensor([0.2, 0.3, 0.4])),
)
builder.add_revolute_y(
    "shoulder",
    parent="base",
    child="link",
    origin=torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]),
    lower=-2.0,
    upper=2.0,
)
pendulum = br.io.build_model(builder.finalize(), dtype=torch.float64)
print(pendulum.nq, pendulum.joint_names)
```
```output
1 ('universe', 'root_joint', 'shoulder')
```

The `origin` is a pose `[tx, ty, tz, qx, qy, qz, qw]`: the quaternion scalar comes last everywhere in
BetterRobot. [Load a robot](../guides/load-a-robot.md) covers MJCF, meshes and joint-order options.

## Where next

- [Compute forward kinematics](forward-kinematics.md) — turn a configuration into frame poses.
- [Load a robot](../guides/load-a-robot.md) — MJCF, meshes and joint-order options.
- [Joints, bodies and frames](../concepts/joints-bodies-frames.md) — every joint type and how frames attach.
