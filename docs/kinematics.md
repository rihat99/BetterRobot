# Kinematics

Kinematics answers "where is each part of the robot?" and "how does it move when the joints move?".
`br.forward_kinematics(model, q)` takes a configuration and returns a `KinematicState`: the pose of every joint,
body and frame. Jacobians are functions of that state. The state is frozen, and anything derived from it is
computed once and kept.

The names you will use:

| Name | What it does |
|---|---|
| `br.forward_kinematics(model, q)` | the `KinematicState` of `q`, with any leading batch axes |
| `KinematicState` | `frame_poses`, `frame_translations`, `frame_rotations`, `frame_pose_of(name)`, `joint_transforms` |
| `br.frame_jacobian(model, state, frame=...)` | the `(6, nv)` Jacobian of a frame; `br.joint_jacobian` for a joint |
| `kinematics.integrate(model, q, v)` | moves `q` along the velocity `v`; use it instead of `q + v` |
| `kinematics.difference(model, q0, q1)` | the velocity that takes `q0` to `q1`; use it instead of `q1 - q0` |
| `kinematics.compiled(fn)` | a `torch.compile`d version of a kinematics or dynamics call, for repeated calls |

## Where is the hand?

```python
import torch
from robot_descriptions import panda_description

import better_robot as br

model = br.load(panda_description.URDF_PATH, dtype=torch.float64)
state = br.forward_kinematics(model, model.q_neutral)
print(state.frame_pose_of("panda_hand"))
print(state.frame_poses.shape)  # (nframes, 7)
```
```output
tensor([ 1.0731e-01, -3.2653e-17,  9.2494e-01,  9.2332e-01,  3.8245e-01,
        -3.2237e-02, -1.3353e-02], dtype=torch.float64)
torch.Size([14, 7])
```

## A batch of configurations

Give `q` any leading batch axes and every result keeps them. Here are 4 x 10 random configurations inside the joint
limits:

```python
generator = torch.Generator().manual_seed(0)
low, high = model.lower_position_limit, model.upper_position_limit
q = low + (high - low) * torch.rand(4, 10, model.nq, generator=generator, dtype=torch.float64)

batch = br.forward_kinematics(model, q)
print(batch.frame_poses.shape)
print(batch.frame_pose_of("panda_hand").shape)
```
```output
torch.Size([4, 10, 14, 7])
torch.Size([4, 10, 7])
```

## Jacobians and gradients

The frame Jacobian maps joint velocities to the frame's twist, `[linear, angular]`, expressed in world axes at the
frame's origin. Autograd also flows through every call, so you can differentiate anything you compute.

```python
J = br.frame_jacobian(model, state, frame="panda_hand")
print(J.shape)

q = model.q_neutral.clone().requires_grad_(True)
height = br.forward_kinematics(model, q).frame_pose_of("panda_hand")[2]
height.backward()
print(torch.allclose(q.grad, J[2]))  # the gradient of the hand's height is the Jacobian's z row
```
```output
torch.Size([6, 8])
True
```

See also: [Dynamics](dynamics.md), which takes the same state.
