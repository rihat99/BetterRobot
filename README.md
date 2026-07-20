# BetterRobot

PyTorch-native library for robot kinematics, optimization, and visualization.

Read the browser documentation at
[rihat99.github.io/BetterRobot](https://rihat99.github.io/BetterRobot/).

## Installation

Requires [uv](https://docs.astral.sh/uv/):

```bash
git clone https://github.com/rihat99/BetterRobot
cd BetterRobot
uv sync
```

The core install contains only BetterRobot's tensor, array, and URDF-loading
requirements. Install extras for the surfaces you use:

```bash
uv sync --extra demos --extra viewer  # bundled examples and browser viewer
uv sync --extra io-mjcf               # MJCF loading
uv sync --extra warp                  # CUDA-validated opt-in Warp FK lane
uv sync --extra dev                   # tests, benchmarks, docs, and lint tools
```

## Examples

```bash
uv sync --extra demos --extra viewer
uv run python examples/01_basic_ik.py   # Franka Panda — open http://localhost:8080
uv run python examples/02_g1_ik.py      # Unitree G1 whole-body IK — open http://localhost:8081
```

Drag the SE(3) gizmos in the browser to move the robot in real time.

## Quick Start

```python
import better_robot as br
from robot_descriptions import panda_description

model = br.load(panda_description.URDF_PATH)

# Forward kinematics
data = br.forward_kinematics(model, model.q_neutral, compute_frames=True)
T_hand = data.frame_pose_world[model.frame_id("body_panda_hand")]

# Inverse kinematics
from better_robot.tasks.ik import solve_ik
result = solve_ik(model, targets={"body_panda_hand": T_hand})
result.q           # solution configuration
result.converged   # bool
result.fk()        # Data with FK at solution
```

Floating-base robots (e.g. humanoids):

```python
from robot_descriptions import g1_description

g1_model = br.load(g1_description.URDF_PATH, free_flyer=True)
# First 7 DOF of q are the base pose [tx, ty, tz, qx, qy, qz, qw]
```

Poses use `[tx, ty, tz, qx, qy, qz, qw]` (scalar-last quaternion) throughout.

## Viewer

Interactive browser-based 3D visualization via [viser](https://github.com/nerfstudio-project/viser):

```python
from better_robot.viewer import Visualizer

viewer = Visualizer(model, port=8080)
viewer.update(result.q)

# Draggable IK targets — returns a TargetsOverlay for polling
overlay = viewer.add_ik_targets({"body_panda_hand": T_hand}, scale=0.15)
viewer.show(block=False)

# Main-loop interactive IK
while True:
    targets = overlay.live_targets()
    r = solve_ik(model, targets=targets, initial_q=viewer.last_q)
    viewer.update(r.q)
```

The viewer renders URDF meshes with embedded colors (DAE) or URDF material colors (STL), falls back to a skeleton mode for programmatic models, and includes a ground grid overlay.

## Architecture

```
src/better_robot/
  lie/           SE3/SO3 group operations
  spatial/       6D spatial algebra (Motion, Force, Inertia)
  data_model/    Model (frozen) and Data (workspace)
  kinematics/    forward_kinematics, Jacobians
  dynamics/      RNEA, ABA, CRBA, centroidal dynamics
  residuals/     Pose, Position, Orientation, Limits, Rest
  costs/         CostStack
  optim/         named-block Problem evaluation plus legacy LM/GN/Adam/L-BFGS
  tasks/         solve_ik, solve_trajopt, solve_contact_forces, Trajectory
  viewer/        Visualizer, Scene, render modes, overlays
  io/            URDF/MJCF loading
```

Named variables, residuals, and least-squares solvers are public under
`better_robot.optim`. Torch/LM/GN optimizers and the `solve_ik` and
knot-based `solve_trajopt` task facades preserve leading batch axes with
per-element solver state. See
[Write a custom residual](docs/guides/custom_residual.md).

## Dependencies

| Package | Purpose |
|---------|---------|
| `torch` | core tensors, autograd, and the eager PyTorch compute lane |
| `numpy` | core array interchange |
| `yourdfpy` | core URDF parsing (currently brings `trimesh` transitively) |
| `viser` | optional `viewer` extra for browser-based visualization |
| `mujoco` | optional `io-mjcf` extra for MJCF loading |
| `trimesh` | optional direct `meshes` extra for mesh APIs |
| `robot_descriptions` | optional `demos` extra for Panda, G1, and other examples |
| `warp-lang` | optional `warp` extra for the CUDA-validated, opt-in fused FK and RNEA lanes |

## License

BetterRobot is licensed under the [Apache License 2.0](LICENSE).
Copyright 2026 BetterRobot contributors.
