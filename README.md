# BetterRobot

PyTorch-native robotics. Load a URDF or MJCF robot, run batched forward kinematics, Jacobians and rigid-body
dynamics with autograd through every step, and solve inverse kinematics, trajectories and motion fits as
least-squares problems over named tensors. Fixed- and floating-base robots share one API, and a browser viewer
plays whole clips.

![64 Panda IK problems solved in one batch](docs/images/readme_ik.gif)

Documentation: [docs/](docs/index.md), served as a site by `uv run mkdocs serve`.

## Install with [uv](https://docs.astral.sh/uv/)

```bash
git clone https://github.com/rihat99/BetterRobot && cd BetterRobot
uv sync --all-extras --all-groups  # the package, every extra, and the dev and test-oracles groups
```

## Solve 64 IK problems at once

```python
import torch
from robot_descriptions import panda_description

import better_robot as br
from better_robot import terms

model = br.load(panda_description.URDF_PATH, dtype=torch.float64)
generator = torch.Generator().manual_seed(0)
q0 = model.q_neutral.expand(64, -1)
# 64 reachable hand poses from random configurations, one batched solve, and forward kinematics checks every answer.
targets = br.forward_kinematics(model, terms.random_start(model, q0, generator=generator)).frame_pose_of("panda_hand")
result = terms.solve_ik(model, {"panda_hand": targets}, q0=q0, restarts=4, generator=generator)
reached = br.forward_kinematics(model, result.x["q"]).frame_pose_of("panda_hand")
print(float((reached[:, :3] - targets[:, :3]).norm(dim=-1).median()))
```
```output
3.1075885246701783e-12
```

The printed number is the median distance, in metres, between the hand and its target over the 64 solves.
`br.forward_kinematics(model, q)` returns a frozen `KinematicState`; Jacobians and dynamics are functions of that
state. Poses are `[tx, ty, tz, qx, qy, qz, qw]` (scalar last).

## Where next

- [Getting started](docs/getting-started/first-model.md): from loading a robot to batched IK, trajectories and the
  viewer, one step at a time.
- [Architecture](docs/concepts/architecture.md): the layers and how one call flows through them.
- [Scope](docs/contracts/scope.md): what BetterRobot computes, what it leaves out, and the bar for a new feature.
- [Examples](examples/): short programs from batched IK (`uv run python examples/01_ik.py`) to dynamics
  (`examples/08_dynamics.py`); the scripts in `examples/viewer/` show, play and record scenes with the viewer.
- Upgrading from 0.2: [Migrating to 0.3.0](docs/reference/migration-0.3.0.md) maps every public name that 0.3.0
  renames or removes, and the [changelog](docs/reference/changelog.md) says what changed.

## Development

```bash
uv run pytest tests/contract -q -m "not slow and not cuda"   # fast tier, about 16 s
uv run pytest tests -q -m "not cuda"                         # full tier, about 4.5 min
uv run ruff check . && uv run ruff format --check .
uv run mkdocs build --strict
```

[Run the tests](docs/guides/run-the-tests.md) gives every tier and what it needs. Pinocchio parity tests
(`tests/parity/pinocchio/`) are the numerical oracle.

## License

Apache License 2.0 ([LICENSE](LICENSE)). Copyright 2026 BetterRobot contributors.
