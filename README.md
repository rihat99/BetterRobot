# BetterRobot

PyTorch-native robotics: batched kinematics, Jacobians and rigid-body dynamics with autograd through every
step, a least-squares optimiser over named tensors (inverse kinematics, trajectories, motion fits), and a
browser viewer for whole clips. Fixed- and floating-base robots share one API.

Documentation: [rihat99.github.io/BetterRobot](https://rihat99.github.io/BetterRobot/) — or build it locally
with `uv run mkdocs serve`.

## Installation

Requires [uv](https://docs.astral.sh/uv/):

```bash
git clone https://github.com/rihat99/BetterRobot
cd BetterRobot
uv sync                                               # torch, numpy, URDF loading
uv sync --extra demos --extra viewer --extra meshes   # example robots, browser viewer, mesh rendering
uv sync --extra io-mjcf                               # MJCF loading
uv sync --extra dev                                   # tests, benchmarks, docs, lint
```

## Quick start

Load a robot, run forward kinematics on a batch, solve 64 inverse-kinematics problems at once:

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
targets = forward_kinematics_state(model, terms.random_start(model, q0, generator)).frame_pose[:, hand]

# Solve all 64 problems, then check them with forward kinematics.
result = terms.solve_ik(model, {"body_panda_hand": targets}, q0, restarts=4, generator=generator)
reached = forward_kinematics_state(model, result.x["q"]).frame_pose[:, hand]
print(float((reached[:, :3] - targets[:, :3]).norm(dim=-1).median()))
```

`forward_kinematics_state(model, q)` returns a frozen `KinematicState`; Jacobians
(`kinematics.frame_jacobian`) and dynamics (`dynamics.inverse_dynamics`, `mass_matrix`, `forward_dynamics`,
`centroidal_map`) are functions of that state. Poses are `[tx, ty, tz, qx, qy, qz, qw]`; a floating base is
`br.load(path, free_flyer=True)`, whose first seven configuration entries are the base pose.

Show a clip in the browser (`viewer` and `meshes` extras):

```python
from better_robot.viewer import Viewer, robot_layers

clip = q0[0].repeat(30, 1)                       # (T, nq)
clip[:, 0] = torch.linspace(-1.0, 1.0, 30, dtype=torch.float64)
viewer = Viewer()                                # http://127.0.0.1:8080
for layer in robot_layers(model, clip, frames=["body_panda_hand"]):
    viewer.add(layer)
viewer.show(0)
```

## Examples

```bash
uv run python examples/01_ik.py             # drag a gizmo, the Panda follows
uv run python examples/02_trajectory.py     # banded Levenberg-Marquardt over 20 knots
uv run python examples/08_dynamics.py       # inverse and forward dynamics round trip
```

`examples/01` to `08` go from IK to hard constraints, obstacle avoidance and dynamics; `examples/viewer/` shows the
viewer layers.

## Architecture

```text
src/better_robot/
  lie/          SO(3) and SE(3) operations on tensors
  spatial/      6D spatial algebra: Motion, Force, Inertia
  data_model/   Model (frozen structure + differentiable values), joint models
  kinematics/   forward_kinematics_state, KinematicState, Jacobians
  dynamics/     inverse/forward dynamics, mass matrix, centroidal dynamics
  collision/    primitive distances for collision terms
  optim/        Problem over named tensors, LM / Gauss-Newton / L-BFGS / torch.optim, banded structure
  terms/        robot terms for optim: poses, limits, collision, solve_ik, RobotManifold
  viewer/       layers, timeline, tools, offline rendering on viser
  io/           URDF / MJCF loading, programmatic ModelBuilder
```

Imports point down this list; `optim` imports only torch. See `docs/concepts/architecture.md`.

## Development

```bash
uv run pytest tests -q -m "not bench and not cuda"
uv run ruff check src tests
uv run mkdocs build --strict
```

Pinocchio parity tests (`tests/test_pinocchio/`) are the numerical oracle.

## License

BetterRobot is licensed under the [Apache License 2.0](LICENSE). Copyright 2026 BetterRobot contributors.
