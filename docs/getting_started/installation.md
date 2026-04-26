# Installation

BetterRobot is published as `better-robot` on PyPI and is pure-Python
plus PyTorch. CPU and CUDA both work; a CUDA-enabled PyTorch build is
detected automatically.

## Quick install

```bash
pip install better-robot[urdf,examples]
```

The `examples` extra pulls in `robot_descriptions`, which provides
URDFs for Panda, G1, and friends without manual downloads. The `urdf`
extra adds `yourdfpy` (the URDF parser).

## Optional extras

| Extra | Adds | When to install |
|-------|------|-----------------|
| `urdf` | `yourdfpy` | Parsing URDFs. Most users. |
| `mjcf` | `mujoco` | Parsing MuJoCo XML scenes. |
| `viewer` | `viser`, `trimesh` | Live visualisation. |
| `geometry` | `trimesh` | Mesh-based collision and rendering. |
| `examples` | `robot_descriptions` | Running the examples. |
| `warp` | `warp-lang` | Future Warp backend (P11; not used today). |
| `bench` | `pytest-benchmark`, `pyperf` | Running the perf gates locally. |
| `docs` | `sphinx`, `furo`, `myst-parser`, `sphinx-autodoc2`, `sphinx-design`, `sphinx-copybutton`, `myst-nb` | Building this site. |
| `test` | `pytest`, `pytest-cov`, `pytest-xdist`, `hypothesis` | Running the test suite. |
| `dev` | the union of `test`, `bench`, plus `mypy` / `pre-commit` | Contributors. |
| `all` | every extra except `dev` | Trying everything in one shot. |

```{seealso}
{doc}`/conventions/20_PACKAGING` — the normative spec for what each
extra contains and the deprecation policy for changing them.
```

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
uv add 'better-robot[urdf,examples]'
# or, in this repo:
uv sync --extra urdf --extra examples
```
