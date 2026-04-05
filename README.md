# BetterRobot

PyTorch-native library for robot kinematics, inverse kinematics, trajectory optimization, and motion retargeting.

Designed around three principles from production robotics toolkits:
- **Model/Data separation** (Pinocchio): `RobotModel` is immutable; algorithms are free functions.
- **Registry-driven solvers** (IsaacLab): swap solvers via a string key, no code changes.
- **Functional cost terms** (PyRoki): residuals compose cleanly into optimization problems.

## Installation

Install from source using [uv](https://docs.astral.sh/uv/):

```bash
git clone <repo-url>
cd BetterRobot
uv sync
```

Or as a local dependency in an existing project:

```bash
uv add --editable /path/to/BetterRobot
# or
pip install -e /path/to/BetterRobot
```

## Quick Start

```python
import better_robot as br
from robot_descriptions.loaders.yourdfpy import load_robot_description

urdf = load_robot_description("panda_description")
model = br.load_urdf(urdf)

# Fixed-base IK — returns (num_joints,) tensor
cfg = br.solve_ik(model, targets={"panda_hand": target_pose})

# Floating-base IK — returns (base_pose (7,), cfg (n,))
base_pose, cfg = br.solve_ik(
    model,
    targets={"left_rubber_hand": p_lh, "right_rubber_hand": p_rh},
    initial_base_pose=torch.tensor([0., 0., 0.78, 0., 0., 0., 1.]),
)
```

Poses use PyPose convention: `[tx, ty, tz, qx, qy, qz, qw]` (scalar-last quaternion).

## Examples

```bash
uv run python examples/01_basic_ik.py   # Franka Panda IK — open http://localhost:8080
uv run python examples/02_g1_ik.py     # Unitree G1 whole-body IK (floating base)
```

Drag the transform handles in the browser to move IK targets in real time.

## Architecture

```
src/better_robot/
  math/           SE3 ops, adjoint, quaternion utils
  models/         RobotModel, load_urdf(), JointInfo, LinkInfo
  algorithms/
    kinematics/   forward_kinematics(), compute_jacobian()
    geometry/     collision primitives (Sphere, Capsule, Box, HalfSpace)
  costs/          CostTerm, pose/limit/rest residual functions + factories
  solvers/        Registry, Problem, LM, PyPose LM (benchmarking), stubs
  tasks/
    ik/           solve_ik(), IKConfig
    trajopt/      stub
    retarget/     stub
  viewer/         Visualizer (viser), quat conversion helpers
```

Dependency rule: `math → models → algorithms → costs → solvers → tasks → viewer`.
No layer imports from a layer above it.

## API Overview

```python
import better_robot as br

# Load
model = br.load_urdf(urdf)                        # yourdfpy.URDF → RobotModel

# Forward kinematics
fk = br.forward_kinematics(model, cfg)            # (num_links, 7) SE3 poses
fk = model.forward_kinematics(cfg)                # same, convenience method
idx = model.link_index("panda_hand")              # link name → index

# Jacobian
J = br.compute_jacobian(model, cfg, link_idx, target_pose, pos_weight, ori_weight)

# IK config
cfg_ik = br.IKConfig(
    pos_weight=1.0,
    ori_weight=0.1,       # keep low — Panda default is ~173° from identity
    limit_weight=0.1,
    rest_weight=0.01,
    jacobian="analytic",  # "autodiff" (default) or "analytic"
    solver="lm",          # any key from SOLVERS registry
)

# Solver registry
from better_robot.solvers import SOLVERS
print(SOLVERS.list())     # ["lm", "lm_pypose", "gn", "adam", "lbfgs"]
solver = SOLVERS.get("lm")(damping=1e-3)
```

## Testing

```bash
uv run pytest tests/ -v                          # 57 tests, all use real Panda URDF
uv run pytest tests/test_ik.py                   # IK tests only
uv run pytest tests/test_lm_benchmark.py -v -s   # timing: autodiff vs analytic vs PyPose LM
```

## Dependencies

| Package | Purpose |
|---------|---------|
| `torch` | tensors, autograd, `torch.func.jacrev` |
| `pypose` | SE3/SO3 Lie group ops, PyPose LM solver |
| `yourdfpy` | URDF parsing |
| `trimesh` | mesh loading for collision geometry |
| `viser` | browser-based 3D visualizer |
| `robot_descriptions` | pre-packaged robot URDFs (Panda, G1, …) |

## Status

| Feature | Status |
|---------|--------|
| Forward kinematics | Done |
| Fixed-base IK (autodiff + analytic J) | Done |
| Floating-base IK (autodiff + analytic J) | Done |
| LM solver | Done |
| Browser visualizer | Done |
| Trajectory optimization | Stub |
| Motion retargeting | Stub |
| GN / Adam / LBFGS solvers | Stubs |
| Collision distances | Stubs |
