# Load a robot

Use {py:func}`better_robot.load` for a robot description on disk. It selects a
parser from the filename, builds the joint tree, and returns a
{py:class}`better_robot.Model` on the requested device and dtype.

## URDF

The `demos` extra supplies the Panda file used here:

```{testcode}
import torch
import better_robot as br
from robot_descriptions import panda_description

model = br.load(panda_description.URDF_PATH, dtype=torch.float64)

assert model.q_neutral.dtype == torch.float64
assert model.q_neutral.shape == (model.nq,)
```

For your own file, replace `panda_description.URDF_PATH` with a path such as
`"robots/my_arm.urdf"`. Add `free_flyer=True` when the root should move in
space.

## MJCF

MJCF parsing uses MuJoCo and is an optional installation:

```bash
python -m pip install '.[io-mjcf]'
```

After that, `br.load("robot.xml")` selects the MJCF parser from the `.xml`
suffix. This call needs a real MJCF file, so the documentation test does not
pretend to execute it in the smaller docs environment.

## Build a model in Python

{py:class}`better_robot.ModelBuilder` is useful for generated robots and small
test models. Body and joint names are ordinary strings; placements use the
same seven-number pose layout as FK.

```{testcode}
import torch
from better_robot.io import ModelBuilder, build_model

builder = ModelBuilder("one_joint")
base = builder.add_body("base")
tip = builder.add_body("tip", mass=1.0, inertia=torch.eye(3))
builder.add_revolute_z(
    "shoulder",
    parent=base,
    child=tip,
    lower=-1.0,
    upper=1.0,
)
model = build_model(builder.finalize())

assert model.nq == 1
assert "shoulder" in model.joint_names
```

Use a named helper such as `add_revolute_z`, `add_prismatic`, or
`add_free_flyer_root` when one exists. The catch-all `add_joint` accepts a
`JointModel` object, not a string joint kind.

For custom mesh resolution or parser registration, use the qualified
`better_robot.io` APIs in the generated reference.
