# Inverse kinematics

Inverse kinematics asks the reverse of FK: **which joint configuration places
a frame at a target pose?** The target pose goes in and a candidate `q` comes
out. BetterRobot solves this as an optimization problem because several joints
can affect the same frame and several targets can compete.

The most reliable first example creates a reachable target with FK, then asks
IK to recover it:

```{testcode}
import better_robot as br
from better_robot.tasks import IKCostConfig, OptimizerConfig
from robot_descriptions import panda_description

model = br.load(panda_description.URDF_PATH)
q0 = model.q_neutral.clamp(model.lower_pos_limit, model.upper_pos_limit)
q_goal = q0.clone()
q_goal[0] = 0.25

frame_name = "body_panda_hand"
target_pose = br.forward_kinematics(
    model,
    q_goal,
    compute_frames=True,
).frame_pose_world[model.frame_id(frame_name)].clone()

result = br.solve_ik(
    model,
    {frame_name: target_pose},
    initial_q=q0,
    cost_cfg=IKCostConfig(limit_weight=0.0, rest_weight=0.0),
    optimizer_cfg=OptimizerConfig(max_iter=100),
)

assert result.converged
assert result.q.shape == (model.nq,)
assert result.frame_pose(frame_name).shape == (7,)
```

Targets use the same `[x, y, z, qx, qy, qz, qw]` layout as FK. The dictionary
may contain more than one frame. `IKResult` always returns the final candidate;
check `result.converged` before treating it as a solution. `result.iters` and
`result.residual` help diagnose an unreachable or over-constrained request.

The optional configuration objects tune weights and stopping behavior. You do
not need them for a first solve; they are shown above only to make this small
generated target especially easy to recover.

Floating-base robots use the same call, as shown in {doc}`04_floating_base`.
