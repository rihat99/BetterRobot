# Meet the robot model

Loading a robot gives you a {py:class}`better_robot.Model`: a read-only
description of its topology and numerical properties. It does not contain the
answer to a kinematics query. Those answers live in a separate
{py:class}`better_robot.Data` object created by each computation.

Four words appear throughout the library:

- A **body** is a rigid physical part with mass and inertia.
- A **joint** connects bodies and describes their allowed relative motion.
- A **frame** is a named coordinate system attached to a body or joint. Tool
  tips and sensor mounts are usually frames.
- **`q`** is the configuration vector. It stores the position of every joint.
  Its layout comes from the joint types, so it is not always just a list of
  angles.

Load the Panda model and inspect those pieces:

```{testcode}
import better_robot as br
from robot_descriptions import panda_description

model = br.load(panda_description.URDF_PATH)
q = model.q_neutral
hand_frame = model.frame_id("body_panda_hand")

assert q.shape == (model.nq,)
assert len(model.joint_names) == model.njoints
assert len(model.body_names) == model.nbodies
assert len(model.frame_names) == model.nframes
assert model.frame_names[hand_frame] == "body_panda_hand"
```

`model.q_neutral` is a valid resting configuration on the model's device and
with its dtype. `model.lower_pos_limit` and `model.upper_pos_limit` describe
the bounded coordinates. A floating base and a spherical joint use
quaternions, so `nq` (stored numbers) can differ from `nv` (independent motion
directions).

Treat the tensors and dictionaries inside `Model` as read-only. If you need a
different device, dtype, placement, or inertial value, create another model
with `model.to(...)` or `model.with_values(...)`.

Next, {doc}`02_forward_kinematics` turns `q` into world poses.
