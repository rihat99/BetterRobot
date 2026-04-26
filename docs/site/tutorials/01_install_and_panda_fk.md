# Install and run forward kinematics on Panda

This tutorial gets you to a working forward-kinematics call on the
Franka Panda in five minutes.

## Install

```bash
pip install better-robot[urdf,examples]
```

The `examples` extra pulls in `robot_descriptions`, which provides
URDFs for Panda, G1, and friends without manual downloads.

## Load the robot

```python
import torch
import better_robot as br
from robot_descriptions import panda_description

model = br.load(panda_description.URDF_PATH, dtype=torch.float64)
print(model.nq, model.nv, model.njoints)
```

`Model` is a frozen dataclass — it owns the kinematic tree, joint
limits, and per-body inertial properties. It does not own a workspace.

## Compute forward kinematics

```python
q = model.q_neutral.clamp(model.lower_pos_limit, model.upper_pos_limit)
data = br.forward_kinematics(model, q, compute_frames=True)

print(data.joint_pose_world.shape)   # (njoints, 7)
print(data.frame_pose_world.shape)   # (nframes, 7)
print(br.get_frame_jacobian(model, data, model.frame_id("panda_hand")).shape)
# (6, nv)
```

`Data` is a mutable per-query workspace. Every kinematics / dynamics
call writes into it; the cache-invariants framework tracks how far the
recursion has been advanced (placements, velocities, accelerations).

## Batched FK

Pass a batched `q` and BetterRobot returns batched poses:

```python
q_batch = q.unsqueeze(0).expand(8, -1).contiguous()
data_batch = br.forward_kinematics(model, q_batch)
print(data_batch.joint_pose_world.shape)  # (8, njoints, 7)
```

That's it — you can now move on to `02_basic_ik.md`.
