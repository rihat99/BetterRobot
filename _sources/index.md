# BetterRobot

BetterRobot is a PyTorch library for robot kinematics, dynamics, and
optimization. It works with ordinary tensors, accepts leading batch axes, and
keeps gradients through the robot calculations. A fixed-base arm and a
floating-base humanoid use the same functions.

Use it when you want to:

- ask where a robot's links and frames are;
- compute rigid-body forces, accelerations, or inertias;
- fit joint configurations and trajectories to observations; or
- put robot calculations inside a PyTorch model or training loop.

BetterRobot analyzes and fits articulated systems. It is not a physics
simulator. Pair it with MuJoCo, Drake, or another simulator when you need to
advance a world through contact and time. The reasoning behind that boundary,
and the costs of the other major choices, is in
{doc}`concepts/design_decisions`.

## Try inverse kinematics

Install the repository with its example robot descriptions:

```bash
git clone --branch dev https://github.com/rihat99/BetterRobot.git
cd BetterRobot
python -m pip install '.[demos]'
```

This example loads a Franka Panda, makes a reachable target from a known joint
configuration, and optimizes a robot configuration to recover it.

<!-- front-page-example:start -->
```{testcode}
import better_robot as br
from better_robot.optim import LevenbergMarquardt, Problem, RobotVariable
from better_robot.residuals import PoseResidual
from robot_descriptions import panda_description

model = br.load(panda_description.URDF_PATH)
q0 = model.q_neutral.clamp(model.lower_pos_limit, model.upper_pos_limit)
q_goal = q0.clone()
q_goal[0] = 0.25
target_pose = br.forward_kinematics(
    model, q_goal, compute_frames=True
).frame_pose_world[model.frame_id("body_panda_hand")].clone()

q = RobotVariable(model, q0, bounds=True)
reach = PoseResidual(
    q, frame="body_panda_hand", target=target_pose
)
problem = Problem([reach])
optimizer = LevenbergMarquardt(problem, max_iterations=100)
info = optimizer.optimize()

solution = br.forward_kinematics(model, q.tensor, compute_frames=True)
solution_pose = solution.frame_pose_world[model.frame_id("body_panda_hand")]

print(bool(info.converged))
print(q.tensor.shape)
print(solution_pose.shape)
```

```{testoutput}
True
torch.Size([8])
torch.Size([7])
```
<!-- front-page-example:end -->

`info.converged` tells you whether the optimizer met its stopping rule. The
variable's `q.tensor` still contains the final candidate when it does not
converge.

## Learn in the order you need

- Start with {doc}`getting_started/index` if robots and their tensor layouts
  are new to you.
- Use {doc}`guides/index` when you already know the outcome you want.
- Read {doc}`concepts/index` for the mathematics and design choices.
- Consult {doc}`reference/index` for exact names, signatures, and terms.
- Read {doc}`conventions/index` before changing the library itself.

## Documentation

```{toctree}
:maxdepth: 2
:caption: Get started

getting_started/index
```

```{toctree}
:maxdepth: 2
:caption: How-to guides

guides/index
```

```{toctree}
:maxdepth: 2
:caption: Concepts

concepts/index
```

```{toctree}
:maxdepth: 1
:caption: Conventions

conventions/index
```

```{toctree}
:maxdepth: 1
:caption: Reference

reference/index
```

```{toctree}
:hidden:

GitHub <https://github.com/rihat99/BetterRobot>
```

## License

BetterRobot is licensed under Apache-2.0. See
{doc}`conventions/source_and_license` for third-party provenance.

- {ref}`genindex`
- {ref}`modindex`
- {ref}`search`
