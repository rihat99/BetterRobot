# Installation

BetterRobot is a Python package built on PyTorch. The same API works on CPU
and CUDA; your PyTorch installation decides which devices are available.

## Install from the repository

BetterRobot is not currently published on PyPI. Install the `dev` branch from
a source checkout:

```bash
git clone --branch dev https://github.com/rihat99/BetterRobot.git
cd BetterRobot
python -m pip install .
```

The core install includes the URDF loader. Add only the integrations you need:

```bash
python -m pip install '.[demos]'    # robot_descriptions used by these tutorials
python -m pip install '.[viewer]'   # interactive browser viewer
python -m pip install '.[io-mjcf]'  # MJCF loading through MuJoCo
python -m pip install '.[warp]'     # optional fused FK and RNEA lanes
```

Contributors can install the test and documentation tools with a reproducible
lockfile:

```bash
uv sync --extra dev --extra demos
```

## Check the installation

The following imports BetterRobot, loads the tutorial robot, and checks that
its public coordinate vectors agree with the model dimensions.

```{testcode}
import better_robot as br
from robot_descriptions import panda_description

model = br.load(panda_description.URDF_PATH)

print(model.q_neutral.shape)
print((model.nq, model.nv, model.njoints))
```

```{testoutput}
torch.Size([8])
(8, 8, 14)
```

Continue with {doc}`01_robot_model` to see what the loaded object contains.
