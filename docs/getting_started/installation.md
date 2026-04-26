# Installation

BetterRobot is pure-Python plus PyTorch. CPU and CUDA both work; a
CUDA-enabled PyTorch build is detected automatically.

## Quick install

```bash
pip install better-robot
```

That gives you everything the library needs at runtime — `torch`,
`numpy`, the URDF and MJCF parsers (`yourdfpy`, `mujoco`), the viewer
stack (`viser`, `trimesh`), and the standard robot zoo
(`robot_descriptions`).

## Contributors

```bash
pip install 'better-robot[dev]'
```

The `dev` extra adds the contributor toolchain — `pytest`,
`hypothesis`, `pin` (Pinocchio reference oracle), `pyperf`, the Sphinx
docs stack, plus `ruff` / `pyright` / `mypy` / `pre-commit`.

## Verify the install

```python
import better_robot as br
import torch
from robot_descriptions import panda_description

model = br.load(panda_description.URDF_PATH, dtype=torch.float64)
print(model.nq, model.nv, model.njoints)
# 9 9 11
```

If that prints three integers you're ready for {doc}`forward_kinematics`.

## With `uv`

If you use `uv` (recommended for development):

```bash
uv add better-robot
# or, in this repo:
uv sync --extra dev
```
