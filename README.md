# BetterRobot

PyTorch-native library for robot kinematics, inverse kinematics, trajectory optimization, and motion retargeting.

## Installation

The package is not yet published to PyPI. Install from source using [uv](https://docs.astral.sh/uv/):

```bash
git clone <repo-url>
cd BetterRobot
uv sync
```

Or install into an existing uv project as a local dependency:

```bash
uv add --editable /path/to/BetterRobot
```

Or with plain pip:

```bash
pip install -e /path/to/BetterRobot
```

## Quick Start

```python
import better_robot as br
from robot_descriptions.loaders.yourdfpy import load_robot_description

urdf = load_robot_description("panda_description")
robot = br.Robot.from_urdf(urdf)

solution = br.solve_ik(
    robot=robot,
    target_link="panda_hand",
    target_pose=target_pose,  # shape (7,) wxyz+xyz
    solver="lm",              # "lm" | "gn" | "adam" | "lbfgs"
)
```
