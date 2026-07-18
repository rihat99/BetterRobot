# Installation

BetterRobot is pure-Python plus PyTorch. CPU and CUDA both work; a
CUDA-enabled PyTorch build is detected automatically.

## Quick install

BetterRobot is not currently published on PyPI. Install the current `dev`
source checkout instead:

```bash
git clone --branch dev https://github.com/rihat99/BetterRobot.git
cd BetterRobot
python -m pip install .
```

That installs the tensor algorithms and URDF loader: `torch`, `numpy`, and
`yourdfpy`. Install only the optional integrations you need:

```bash
python -m pip install '.[demos]'       # robot_descriptions examples
python -m pip install '.[viewer]'      # viser browser viewer
python -m pip install '.[io-mjcf]'     # MuJoCo-backed MJCF loading
python -m pip install '.[meshes]'      # direct trimesh APIs
python -m pip install '.[warp]'        # CUDA-validated opt-in Warp FK
```

Extras can be combined, for example
`python -m pip install '.[demos,viewer]'` from the repository root.

## Contributors

```bash
python -m pip install -e '.[dev,demos]'
```

The `dev` extra adds the contributor toolchain — `pytest`,
`hypothesis`, `pin` (Pinocchio reference oracle), `pyperf`, the Sphinx
docs stack, plus `ruff` / `pyright` / `mypy` / `pre-commit`. The separate
`demos` extra supplies `robot_descriptions` for the verification example and
robot fixtures below.

## Verify the install

```python
import better_robot as br
import torch
from robot_descriptions import panda_description

model = br.load(panda_description.URDF_PATH, dtype=torch.float64)
print(model.nq, model.nv, model.njoints)
# 8 8 14 with the currently locked Panda description
```

The Panda gripper mimic target is removed from the public reduced-coordinate
layout, so public `nq`/`nv` are one smaller than the full internal layout. If
the example prints those three integers, you are ready for
{doc}`forward_kinematics`.

## With `uv`

If you use `uv` (recommended for development):

```bash
uv sync --extra dev --extra demos
```

`uv sync` consumes the committed `uv.lock`; it is the reproducible contributor
setup for this repository.
