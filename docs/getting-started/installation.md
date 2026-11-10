# Install BetterRobot

You install BetterRobot with `uv` in one command and check that the install works. After this page you can run
every tutorial in Getting started.

## Install the package with uv

BetterRobot is a plain Python package that needs Python 3.12 and PyTorch. The project uses
[uv](https://docs.astral.sh/uv/) for environments. From a clone of the repository:

```bash
git clone <your BetterRobot clone URL> BetterRobot
cd BetterRobot
uv sync --all-extras --all-groups   # the package, every extra, and the dev and test-oracles groups
```

## What the extras add

| Extra | Installs | You need it for |
|---|---|---|
| `viewer` | `viser` | the browser viewer in [Visualize a robot](visualize.md) |
| `mjcf` | `mujoco` | loading MJCF (`.xml`) models through MuJoCo's compiler |
| `meshes` | `trimesh` | loading visual and collision meshes |
| `demos` | `robot_descriptions` | the Panda and G1 models used in these tutorials |
| `testing` | `pytest` | the pytest plugin `better_robot.testing.no_skips` in a downstream test suite |

The core alone (`torch`, `numpy` and the URDF parser) has no viewer, no MuJoCo and no robot assets. The `dev` group
holds the test, lint and docs tools; `test-oracles` holds Pinocchio and SciPy, the references of the parity tests.

## Get a CUDA build of PyTorch

`pyproject.toml` pins `torch` to the PyTorch `cu129` wheel index
(`https://download.pytorch.org/whl/cu129`), so `uv sync` installs a CUDA 12.9 build on Linux. CUDA 12.9 wheels
also cover recent GPUs such as Blackwell (`sm_120`). Nothing in BetterRobot itself is compiled: the same
code runs on CPU and GPU, chosen by the device of your tensors (see [GPU and batching](gpu-and-batching.md)).

## Check the install

Run the fast test tier (seconds; the other tiers are in [Run the tests](../guides/run-the-tests.md)):

```bash
uv run pytest tests/contract -q -m "not slow and not cuda"
```

Or load a robot in Python. This needs the `demos` extra:

```python
import torch
import better_robot as br
from robot_descriptions import panda_description

model = br.load(panda_description.URDF_PATH)
print(model.name, model.nq, torch.cuda.is_available())
```
```output
panda 8 False
```

The first `robot_descriptions` import downloads the robot repository into `~/.cache`; later runs are offline.

## Where next

- [Load your first model](first-model.md) — the Panda, and what a `Model` holds.
- [Visualize a robot](visualize.md) — needs the `viewer` and `meshes` extras installed above.
