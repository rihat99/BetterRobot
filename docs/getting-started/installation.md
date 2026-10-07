# Install BetterRobot

You install BetterRobot with `uv`, pick the optional extras you need, and check that the install works.
After this page you can run every tutorial in Getting started.

## Install the package with uv

BetterRobot is a plain Python package that needs Python 3.10+ and PyTorch. The project uses
[uv](https://docs.astral.sh/uv/) for environments. From a clone of the repository:

```bash
git clone <your BetterRobot clone URL> BetterRobot
cd BetterRobot
uv sync                      # core: torch, numpy and the URDF parser
```

The core has no viewer, no MuJoCo and no robot assets. You add those through extras.

## Pick the extras you need

| Extra | Installs | You need it for |
|---|---|---|
| `viewer` | `viser` | the browser viewer in [Visualize a robot](visualize.md) |
| `io-mjcf` | `mujoco` | loading MJCF (`.xml`) models |
| `meshes` | `trimesh` | loading visual and collision meshes |
| `demos` | `robot_descriptions` | the Panda and G1 models used in these tutorials |
| `dev` | pytest, Pinocchio, ruff, mkdocs, ... | running the tests and building these docs |

The tutorials use `demos`, `meshes` and `viewer`:

```bash
uv sync --extra demos --extra meshes --extra viewer
uv sync --all-extras         # everything, including dev
```

## Get a CUDA build of PyTorch

`pyproject.toml` pins `torch` to the PyTorch `cu129` wheel index
(`https://download.pytorch.org/whl/cu129`), so `uv sync` installs a CUDA 12.9 build on Linux. CUDA 12.9 wheels
also cover recent GPUs such as Blackwell (`sm_120`). Nothing in BetterRobot itself is compiled: the same
code runs on CPU and GPU, chosen by the device of your tensors (see [GPU and batching](gpu-and-batching.md)).

## Check the install

Run the tests that need no GPU:

```bash
uv run pytest tests/ -q -m "not bench and not cuda"
```

Or load a robot in Python. This needs the `demos` extra:

```python
import torch
import better_robot as br
from robot_descriptions import panda_description

model = br.load(panda_description.URDF_PATH)
print(model.name, model.nq, torch.cuda.is_available())
```

The first `robot_descriptions` import downloads the robot repository into `~/.cache`; later runs are offline.

Next: [load your first model](first-model.md).
