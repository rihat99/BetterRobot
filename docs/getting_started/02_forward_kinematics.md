# Forward kinematics

Forward kinematics answers a direct question: **given the joint
configuration, where is every part of the robot?** Joint values go in; poses
come out. There is no optimization step.

BetterRobot returns a mutable {py:class}`better_robot.Data` workspace. Joint
poses are always computed. Pass `compute_frames=True` when you also need named
frame poses.

```{testcode}
import better_robot as br
from robot_descriptions import panda_description

model = br.load(panda_description.URDF_PATH)
q = model.q_neutral
data = br.forward_kinematics(model, q, compute_frames=True)

hand_id = model.frame_id("body_panda_hand")
hand_pose = data.frame_pose_world[hand_id]

print(data.joint_pose_world.shape)
print(data.frame_pose_world.shape)
print(hand_pose.shape)
```

```{testoutput}
torch.Size([14, 7])
torch.Size([14, 7])
torch.Size([7])
```

Each pose stores translation followed by a scalar-last quaternion:
`[x, y, z, qx, qy, qz, qw]`. The suffix `_world` means that the pose is
expressed in world coordinates.

`Data` belongs to one query and may be updated by later kinematics or dynamics
calls. Keep `Model` for the robot description; keep `Data` only as long as you
need the computed values.

The reverse question—finding joints that achieve a desired pose—is
{doc}`03_inverse_kinematics`.
