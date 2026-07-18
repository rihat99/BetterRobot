# Floating-base robots

A fixed-base robot has a root that cannot move. A floating-base robot has six
root motion directions: three translations and three rotations. BetterRobot
represents that root as an ordinary `JointFreeFlyer`, so FK, IK, and dynamics
do not need a second robot API.

Pass `free_flyer=True` while loading a URDF. The first seven entries of `q` are
the base pose `[x, y, z, qx, qy, qz, qw]`, while the base contributes six
entries to velocity. That is why `nq` is one larger than `nv` for this model.

```{testcode}
import better_robot as br
from robot_descriptions import g1_description

model = br.load(g1_description.URDF_PATH, free_flyer=True)
q = model.q_neutral.clone()
q[2] += 0.05

data = br.forward_kinematics(model, q, compute_frames=True)
left_ankle = model.frame_id("body_left_ankle_roll_link")

assert model.nq == model.nv + 1
assert q[:7].shape == (7,)
assert data.frame_pose_world[left_ankle].shape == (7,)
```

Do not add or normalize quaternion components by hand. Use
`model.integrate(q, tangent_step)` when applying an update in the model's
`nv`-dimensional tangent space. `solve_ik` already uses that operation, so a
floating-base IK call has exactly the same shape as the fixed-base call in the
previous tutorial.

Frame names come from the robot description. Inspect `model.frame_names`
instead of assuming that two URDF versions use the same names.

Next, {doc}`05_batched_gpu` evaluates many independent configurations at once.
