# Coupling standard

The shapes, layouts, identifiers and file formats through which BetterHuman, BetterVideoReconstruction and
BetterRetarget read BetterRobot. The standard changes only by addition: removing or changing a line needs a decision
record that shows each consumer's migration (record [0026](decisions/0026-coupling-standard-page.md)). It names no
function, so a function can be renamed without touching it. Units are SI (metres, radians, kilograms, seconds).
`B...` are the leading batch axes of an input, `I...` those of the model's values.

Every line of a `text` block below is part of the standard. `tests/contract/test_coupling_standard.py` checks each one
against the library, and holds the lines to their frozen copy `tests/contract/coupling_standard_lines.txt`: a line
added here is added there in the same commit, and a line there never leaves this page. Each consumer runs its own
coupling test against this page; BetterHuman's page adds the `HumanState` section.

## Configuration and tangent

```text
q        (B..., nq)   one slice per joint, in joint order; a joint without coordinates has an empty slice
v        (B..., nv)   the tangent of q, one slice per joint; never added to q
```

Per joint kind, the width and layout of its slices. The axis is the joint's unit axis, the child frame the frame the
joint moves.

```text
kind         nq  nv  q                                      v
fixed        0   0   []                                     []
revolute     1   1   [angle] about the axis                 [angular velocity]
prismatic    1   1   [displacement] along the axis          [linear velocity]
helical      1   1   [angle] about the axis, with pitch * angle along it   [angular velocity]
continuous   2   1   [cos, sin] of the angle about the axis   [angular velocity]
spherical    4   3   [qx, qy, qz, qw]                       [wx, wy, wz] in the child frame
planar       4   3   [x, y, cos, sin]: x, y in the plane, the angle about z   [vx, vy, wz] in the child frame
translation  3   3   [x, y, z]                              [vx, vy, vz]
free flyer   7   6   [tx, ty, tz, qx, qy, qz, qw]           [vx, vy, vz, wx, wy, wz] in the child frame
```

## Poses, twists, wrenches, inertias, Euler angles

```text
pose       (..., 7)     [tx, ty, tz, qx, qy, qz, qw]: quaternion scalar last
rotation   (..., 4)     [qx, qy, qz, qw]; a quaternion and its negative are the same rotation
transform  (..., 3, 4)  [R | p]: the same pose as a matrix
twist      (..., 6)     [linear(3), angular(3)]
wrench     (..., 6)     [force(3), torque(3)]
inertia    (..., 10)    [m, cx, cy, cz, Ixx, Iyy, Izz, Ixy, Ixz, Iyz]: mass, centre of mass, rotational inertia
                        about the centre of mass, in the body's axes (MuJoCo's fullinertia order)
euler      (..., 3)     [roll, pitch, yaw], the rotation Rz(yaw) Ry(pitch) Rx(roll)
```

## Jacobians

```text
jacobian   (B..., 6, nv)   rows [linear, angular], one column per coordinate of v
reference  default local_world_aligned: the velocity of the frame's origin and the angular velocity, in world axes
```

## Forces and gravity

```text
external force  (B..., njoints, 6)  [force(3), torque(3)] on the body of each joint, in that joint's frame
gravity         (6,)                [linear(3), angular(3)]: the world's acceleration of gravity, about the world origin
```

## Joints and identifiers

```text
joint 0    "universe": fixed, identity placement, parent -1, no coordinates; body 0 and frame 0 carry its name
joint 1    the root: the joint that holds the root body under the universe; a free flyer there floats the base
order      parents first; among joints whose parent is placed, the one the description lists first
mimic      a mimic joint has no slice of q or v; it appears only in the full layout (the *_full fields)
ids        joint, frame and body: an int id or a str name at every public boundary
           an int outside [0, n) or a name the model does not have raises
names      exactly as the description gives them; every body has a frame of its own name
```

Joint order is record [0012](decisions/0012-one-joint-order.md); identifiers are record
[0029](decisions/0029-frames-and-identifiers.md).

## Batch axes

```text
batch      a state's batch is broadcast(B..., I...); shapes that do not broadcast raise ShapeError
values     joint_placements (I..., njoints, 7), body_inertias (I..., nbodies, 10), frame_placements (I..., nframes, 7)
no batch   every other model value (limits, armature, damping, gravity, q_neutral) has exactly its own shape
optim      a variable is (B, *event): one leading batch axis, the same B for every variable of a problem
```

Batch axes are record [0028](decisions/0028-batch-axes.md): forward kinematics, Jacobians and inverse dynamics
broadcast an input batch against an identity batch, and `optim` keeps one batch axis as the one exception.

## Files

```text
format                  direction    owner         promise
URDF                    read         BetterRobot   the upstream format, within the loader's documented scope
MJCF                    read         BetterRobot   compiled by MuJoCo; the passive rigid tree
.viser                  write        BetterRobot   viser's scene recording, played by viser's static client
better_human.mhr        read         BetterHuman   format and format_version fields; every version written stays readable
better_human.soma       read         BetterHuman   format and format_version fields; every version written stays readable
better_human.frame_set  read, write  BetterHuman   format and format_version fields; every version written stays readable
no BetterRobot model file: a model is rebuilt from its description
```

The formats are record [0030](decisions/0030-formats-in-the-standard.md); the MJCF scope is record
[0036](decisions/0036-mjcf-compile-through-mujoco.md) and the missing model file record
[0032](decisions/0032-br-serialization-non-goal.md). BetterHuman checks its three formats in
`BetterHuman/tests/contract/test_formats.py`; each one's spec is a page of BetterHuman's documentation.
