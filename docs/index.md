# BetterRobot

BetterRobot is a PyTorch-native robotics library. It loads robots from URDF, MJCF or code into a
`Model`, and computes forward kinematics, Jacobians and rigid-body dynamics on batches of
configurations, on CPU or GPU, with autograd flowing through every step. On top of that sits an
optimisation layer (`optim`) that solves inverse kinematics, trajectories and motion fits as
problems over named tensors, and a browser viewer for whole clips.

```python
import torch
from robot_descriptions import panda_description

import better_robot as br
from better_robot import terms
from better_robot.kinematics import forward_kinematics_state

model = br.load(panda_description.URDF_PATH, dtype=torch.float64)
hand = model.frame_id("body_panda_hand")

# A batch of 64 reachable hand poses, made from random configurations.
generator = torch.Generator().manual_seed(0)
q0 = model.q_neutral.clamp(model.lower_pos_limit, model.upper_pos_limit).expand(64, -1)
targets = forward_kinematics_state(model, terms.random_start(model, q0, generator=generator)).frame_pose[:, hand]

# Solve all 64 inverse-kinematics problems at once, then check them with forward kinematics.
result = terms.solve_ik(model, {"body_panda_hand": targets}, q0=q0, restarts=4, generator=generator)
reached = forward_kinematics_state(model, result.x["q"]).frame_pose[:, hand]
error = (reached[:, :3] - targets[:, :3]).norm(dim=-1)
print(f"median hand position error: {float(error.median()):.1e} m")
```

<figure markdown>
  ![The Panda at its neutral configuration, hand frame shown.](images/panda_rest.png){ width="560" }
  <figcaption>The Panda at its neutral configuration, hand frame shown.</figcaption>
</figure>

<div class="grid cards" markdown>

-   **Getting started**

    ---

    Install, load a robot, run forward and inverse kinematics, and optimise a trajectory.

    [Start here](getting-started/installation.md)

-   **Concepts**

    ---

    How the model, kinematic state, dynamics and optimisation layers fit together.

    [Read the concepts](concepts/architecture.md)

-   **Reference**

    ---

    Every public function and class, the glossary, the changelog and the roadmap.

    [Open the API](reference/api/better_robot.md)

</div>
