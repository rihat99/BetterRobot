# BetterRobot

PyTorch-native library for robot kinematics and inverse kinematics.

## Installation

```bash
git clone https://github.com/rikhat/BetterRobot
cd BetterRobot
uv sync
```

Requires [uv](https://docs.astral.sh/uv/). As a local editable dependency:

```bash
uv add --editable /Users/rikhat.akizhanov/Desktop/cources/PhD/ROB803/assignment_3/BetterRobot
```

## Quick Start

```python
import better_robot as br
from robot_descriptions.loaders.yourdfpy import load_robot_description

urdf = load_robot_description("panda_description")
model = br.load_urdf(urdf)

# Fixed-base IK
cfg = br.solve_ik(model, targets={"panda_hand": target_pose})

# Floating-base IK
base_pose, cfg = br.solve_ik(
    model,
    targets={"left_rubber_hand": p_lh, "right_rubber_hand": p_rh},
    initial_base_pose=torch.tensor([0., 0., 0.78, 0., 0., 0., 1.]),
)
```

Poses: `[tx, ty, tz, qx, qy, qz, qw]` (scalar-last quaternion).

## Examples

```bash
uv run python examples/01_basic_ik.py   # Franka Panda — open http://localhost:8080
uv run python examples/02_g1_ik.py     # Unitree G1 whole-body IK (floating base)
```

## Dependencies

| Package | Purpose |
|---------|---------|
| `torch` | tensors, autograd |
| `pypose` | SE3/SO3 Lie group ops |
| `yourdfpy` | URDF parsing |
| `viser` | browser-based 3D visualizer |
| `robot_descriptions` | Panda, G1, and other URDFs |
