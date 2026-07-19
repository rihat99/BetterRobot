# Differentiate through kinematics

PyTorch can differentiate a kinematics result with respect to `q`. Use the raw
functional passes inside `torch.func` transforms: they return tensors in frozen
named results and do not mutate a {py:class}`better_robot.Data` workspace.

This example computes the derivative of the Panda hand position with respect
to every stored configuration coordinate.

```{testcode}
import torch
import better_robot as br
from better_robot.kinematics import (
    forward_kinematics_raw,
    frame_placements_raw,
)
from robot_descriptions import panda_description

model = br.load(panda_description.URDF_PATH, dtype=torch.float64)
frame_id = model.frame_id("body_panda_hand")


def hand_position(q):
    fk = forward_kinematics_raw(model.structure, model.values, q)
    frames = frame_placements_raw(
        model.structure,
        model.values,
        fk.joint_pose_world,
    )
    return frames.frame_pose_world[frame_id, :3]


q = model.q_neutral.clone()
position_jacobian = torch.func.jacrev(hand_position)(q)

assert position_jacobian.shape == (3, model.nq)
assert torch.isfinite(position_jacobian).all()
```

The raw passes trust their caller. Keep `q`, `ModelValues`, dtype, device, and
trailing dimensions consistent with the model, just as the example does.

This representation Jacobian has `nq` columns because it differentiates the
stored coordinates. A robot's geometric frame Jacobian has `nv` columns and
maps a tangent velocity to a spatial twist. Use
`better_robot.get_frame_jacobian` when that physical mapping is what you need;
use `torch.func` when you need the derivative of a particular tensor-valued
calculation.

Optimization uses the tangent interpretation. A `RobotVariable` owns `q` and
its `Model.integrate`/`Model.difference` geometry; residuals reference that
variable (or a shared `RobotState` node), and `Problem` Jacobian columns have
width `nv`. Graph-carrying targets belong in static `Variable` objects when an
implicit optimizer backward should differentiate the solved configuration
with respect to them.
