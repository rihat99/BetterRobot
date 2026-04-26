# Floating-base IK on the G1 humanoid

The G1 is a free-flyer humanoid. Loading it is identical to fixed-base
robots — pass `free_flyer=True`:

```python
import torch
import better_robot as br
from robot_descriptions import g1_description

model = br.load(g1_description.URDF_PATH, free_flyer=True, dtype=torch.float64)

# First 7 dimensions of q are [tx, ty, tz, qx, qy, qz, qw] (the base pose).
print(model.nq, model.nv)  # nq = 7 + n_actuated, nv = 6 + n_actuated
```

The base joint is `JointFreeFlyer` — its `nq=7` (quaternion-augmented
position) but `nv=6` (twist).

## Solving for a foot pose

```python
q0 = model.q_neutral.clone()
target_left_ankle = torch.tensor(
    [0.05, 0.10, 0.10, 0.0, 0.0, 0.0, 1.0],
    dtype=torch.float64,
)
result = br.solve_ik(
    model,
    {"left_ankle": target_left_ankle},
    initial_q=q0,
)
```

`solve_ik` handles this transparently — no `initial_base_pose` argument
needed. The retraction inside the LM step is SE(3)-aware on the base
joint and linear on revolutes — this is what makes "one code path"
possible for fixed and floating-base alike.

```{seealso}
{doc}`/concepts/lie_and_spatial` for the Lie-algebra background, and
{doc}`/concepts/dynamics` for the full per-joint integration story.
```
