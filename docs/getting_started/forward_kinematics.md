# Forward kinematics on Panda

A working forward-kinematics call on the Franka Panda in under five
minutes.

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
See {doc}`/concepts/model_and_data` for the *why*.

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
recursion has been advanced (placements → velocities → accelerations).

## Batched FK

Pass a batched `q` and BetterRobot returns batched poses:

```python
q_batch = q.unsqueeze(0).expand(8, -1).contiguous()
data_batch = br.forward_kinematics(model, q_batch)
print(data_batch.joint_pose_world.shape)  # (8, njoints, 7)
```

There is no "unbatched mode" — single configurations are just `B=1`.
See {doc}`/concepts/batched_by_default`.

## Next

Move on to {doc}`inverse_kinematics`.
