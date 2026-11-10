# Every robot source becomes the same Model

After this page you know why every robot source, a URDF file, an MJCF file or a `ModelBuilder`, goes through one
intermediate representation and one builder function, and what the MJCF loader reads, ignores and refuses.

## Two steps: parse to IR, build the model

Loading has two stages with one owner each:

1. A parser reads a source and returns an `IRModel` (`better_robot.io.ir`): plain bodies, joints, frames,
   inertias and limits, with no topology decisions made yet.
2. `build_model(ir, root_joint=None, device=None, dtype=torch.float32)` orders the joints into a tree
   (parents first, the source's order wherever it is already topological), assigns `idx_q` and `idx_v`, packs
   every tensor field, names a frame after every body and returns a frozen [`Model`](model-and-state.md) whose
   `q_neutral` lies inside the limits.

```text
URDF file  --parse_urdf-->  \
MJCF file  --parse_mjcf-->   IRModel  --build_model-->  Model
ModelBuilder.finalize() -->  /
```

`better_robot.load(source, ...)` does both stages. It picks the parser from the file suffix or an explicit
`format="urdf" | "mjcf"`; an unknown suffix raises `ModelError`.

`better_robot.io.parse(path, format=...)` runs only the first stage and returns the `IRModel`.
Composition lives on the IR: parse several files, merge or edit their `bodies`, `joints` and `frames`
records, then call `build_model(ir)`. `load(path)` is `build_model(parse(path))` plus the root-joint
option below.

```python
import better_robot as br
from better_robot import ModelBuilder
from better_robot.io import build_model

builder = ModelBuilder("arm")
builder.add_body("base", mass=1.0)
builder.add_body("link", mass=0.5)
builder.add_revolute("shoulder", axis=(0.0, 0.0, 1.0), parent="base", child="link")

ir = builder.finalize()   # IRModel, no topology decisions yet
model = build_model(ir)
assert model.nq == 1
```

MJCF goes through MuJoCo's compiler: `parse_mjcf` reads the compiled `MjModel`, so defaults, angle units,
`limited="auto"`, `euler`/`axisangle` orientations and `fromto` geoms are MuJoCo's own; `tests/parity/mujoco/`
checks the result against `mj_kinematics`. A model's `q` is MuJoCo's `qpos`: a joint `ref` is folded into the
joint's placement, the limits stay in `qpos` units, and `q_neutral` is `qpos0` (a `ref`, a free body's
`pos`/`quat`) clamped into the limits.

The MJCF loader's scope is the passive rigid tree: bodies, joints, inertias, limits, damping, armature, geoms,
sites and gravity. What lies in that scope but cannot be expressed raises `ModelError`: a body with two joints, a
ball joint with a `range` (a cone limit), an `<equality>` other than an active linear coupling of two joints (that
one becomes a mimic joint), and an ellipsoid, plane, height-field or SDF geom. What lies outside it is ignored by
rule and never reaches the `Model`: actuators, tendons, sensors, contact settings (`<contact>` pairs and excludes,
geom `friction`, `condim`, `solref`, `solimp`, `margin`), joint springs (`stiffness`, `springref`) and
`frictionloss`, keyframes, cameras, lights, materials and textures (a geom keeps its `rgba`), the world body's
geoms (scenery; its sites become frames on `"universe"`) and every simulation option but gravity.

## Loading is a guide

Loading a file, building a robot in code, adding a floating base and choosing dtype and device are the steps of
[Load a robot](../guides/load-a-robot.md); the `io` API is in the [io reference](../reference/api/io.md).

## Where next

- [Load a robot](../guides/load-a-robot.md): every way to get a `Model`, step by step.
- [Model and state](model-and-state.md): what the built `Model` contains.
- [Joints, bodies and frames](joints-bodies-frames.md): the parts a parser fills in.
- [Architecture](architecture.md): where `io` sits in the package layers.
