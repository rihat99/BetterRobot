# BetterRobot

PyTorch-native library for robot kinematics, optimization, and visualization.

## Installation

Requires [uv](https://docs.astral.sh/uv/):

```bash
git clone https://github.com/rihat99/BetterRobot
cd BetterRobot
uv sync
```

## Examples

```bash
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
T_hand = data.oMf[model.frame_id("body_panda_hand")]

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

model = br.load(g1_description.URDF_PATH, free_flyer=True)
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
overlay = viewer.add_ik_targets({"body_panda_hand": T_target}, scale=0.15)
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
  dynamics/      RNEA/ABA/CRBA stubs
  residuals/     Pose, Position, Orientation, Limits, Rest
  costs/         CostStack
  optim/         LM, Gauss-Newton, Adam, L-BFGS
  tasks/         solve_ik, IKCostConfig, OptimizerConfig
  viewer/        Visualizer, Scene, render modes, overlays
  io/            URDF/MJCF loading
```

## Dependencies

| Package | Purpose |
|---------|---------|
| `torch` | tensors, autograd |
| `pypose` | SE3/SO3 Lie group ops |
| `yourdfpy` | URDF parsing |
| `viser` | browser-based 3D viewer |
| `trimesh` | mesh loading (DAE/OBJ/STL) |
| `robot_descriptions` | example URDFs (Panda, G1, etc.) |
