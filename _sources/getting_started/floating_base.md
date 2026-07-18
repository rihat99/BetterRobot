# Floating-base IK on the G1 humanoid

The G1 is a free-flyer humanoid. Loading it is identical to fixed-base
robots — pass `free_flyer=True`:

```bash
python -m pip install '.[demos]'  # from the BetterRobot source checkout
```

```python
import torch
import better_robot as br
from robot_descriptions import g1_description

model = br.load(g1_description.URDF_PATH, free_flyer=True, dtype=torch.float64)

# First 7 dimensions of q are [tx, ty, tz, qx, qy, qz, qw] (the base pose).
print(model.nq, model.nv)  # 36 35 for the currently locked G1 description
```

The base joint is `JointFreeFlyer` — its `nq=7` (quaternion-augmented
position) but `nv=6` (twist).

## Solving for a foot pose

```python
q0 = model.q_neutral.clone()
q0[7:] = q0[7:].clamp(model.lower_pos_limit[7:], model.upper_pos_limit[7:])

foot_frame = "body_left_ankle_roll_link"
target_left_ankle = br.forward_kinematics(
    model, q0, compute_frames=True
).frame_pose_world[model.frame_id(foot_frame)].clone()
target_left_ankle[2] += 0.01  # a nearby, reachable 1 cm lift

result = br.solve_ik(
    model,
    {foot_frame: target_left_ankle},
    initial_q=q0,
)
```

Frame names come from the loaded description. In the currently locked G1 URDF,
the ankle frames include `body_left_ankle_pitch_link` and
`body_left_ankle_roll_link`; a generic `left_ankle` frame does not exist. Inspect
`model.frame_names` when using another description version.

`solve_ik` handles this transparently — no `initial_base_pose` argument
needed. The retraction inside the LM step is SE(3)-aware on the base
joint and linear on revolutes — this is what makes "one code path"
possible for fixed and floating-base alike.

```{seealso}
{doc}`/concepts/lie_and_spatial` for the Lie-algebra background, and
{doc}`/concepts/dynamics` for the full per-joint integration story.
```
