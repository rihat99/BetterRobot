# Install and quickstart

BetterRobot needs Python 3.12 or newer and PyTorch. The project uses [uv](https://docs.astral.sh/uv/). From a
clone of the repository:

```bash
git clone https://github.com/rihat99/BetterRobot && cd BetterRobot
uv sync --all-extras --all-groups
```

This installs the package, every extra, and the tools for tests and docs. On Linux, `uv sync` installs PyTorch's
CUDA 12.9 build, because `pyproject.toml` points `torch` at that wheel index. Nothing in BetterRobot is compiled:
the device of your tensors decides whether a call runs on CPU or GPU.

The core needs only `torch`, `numpy` and a URDF parser. The extras add the rest:

| Extra | Installs | You need it for |
|---|---|---|
| `viewer` | `viser`, `pillow` | the browser viewer |
| `mjcf` | `mujoco` | loading MJCF (`.xml`) files |
| `meshes` | `trimesh` | reading mesh files |
| `demos` | `robot_descriptions` | the Panda and G1 robots used on these pages |
| `testing` | `pytest` | the `no_skips` pytest plugin in your own test suite |

## Quickstart

Load the Franka Panda and ask where its hand is. The first `robot_descriptions` import downloads the robot into
`~/.cache`; later runs work offline.

```python
import torch
from robot_descriptions import panda_description

import better_robot as br

model = br.load(panda_description.URDF_PATH, dtype=torch.float64)
state = br.forward_kinematics(model, model.q_neutral)
hand = state.frame_pose_of("panda_hand")
print(model.name, model.nq)
print(hand)
```
```output
panda 8
tensor([ 1.0731e-01, -3.2653e-17,  9.2494e-01,  9.2332e-01,  3.8245e-01,
        -3.2237e-02, -1.3353e-02], dtype=torch.float64)
```

A pose is `[tx, ty, tz, qx, qy, qz, qw]`: a position in metres and a unit quaternion with the scalar last.

Now move the hand 10 cm forward and solve inverse kinematics for the new pose. `solve_ik` works on batches, so the
target and the start get a batch axis of size one:

```python
target = hand.clone()
target[0] += 0.10
result = br.solve_ik(model, {"panda_hand": target[None]}, q0=model.q_neutral[None])  # a batch of one
reached = br.forward_kinematics(model, result.x["q"][0]).frame_pose_of("panda_hand")
print(f"hand error after IK: {float((reached[:3] - target[:3]).norm()):.1e} m")
```
```output
hand error after IK: 6.6e-12 m
```

To check the whole install, run the fast test tier (about 20 seconds):

```bash
uv run pytest tests/contract -q -m "not slow and not cuda"
```

Next, read [Model and io](model-and-io.md) to see what a `Model` holds.
