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
configuration, and asks inverse kinematics to recover it.

<!-- front-page-example:start -->
```python
import better_robot as br
from better_robot.tasks.ik import IKCostConfig, OptimizerConfig
from robot_descriptions import panda_description

model = br.load(panda_description.URDF_PATH)
q0 = model.q_neutral.clamp(model.lower_pos_limit, model.upper_pos_limit)
q_goal = q0.clone()
q_goal[0] = 0.25
target_pose = br.forward_kinematics(
    model, q_goal, compute_frames=True
).frame_pose_world[model.frame_id("body_panda_hand")].clone()

result = br.solve_ik(
    model,
    {"body_panda_hand": target_pose},
    initial_q=q0,
    cost_cfg=IKCostConfig(limit_weight=0.0, rest_weight=0.0),
    optimizer_cfg=OptimizerConfig(max_iter=100),
)

result.q  # (nq,) joint solution
result.frame_pose("body_panda_hand")  # (7,) pose at the solution
```
<!-- front-page-example:end -->

`result.converged` tells you whether the solver met its stopping rule. The
result still contains the final candidate when it does not converge.

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
