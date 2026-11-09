# Every robot source becomes the same Model

After this page you can load a URDF or MJCF file, build a robot in code with `ModelBuilder`, turn a
fixed-base description into a floating-base robot, and choose the dtype and device of the result.
All paths go through one intermediate representation and one builder function.

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

## Load a robot description

`load` takes a path to a `.urdf` or MJCF `.xml` file. The `robot_descriptions` package provides paths for
many published robots; it is an optional dependency.

```python { .no-run }
import torch
import better_robot as br
from robot_descriptions import panda_description

model = br.load(panda_description.URDF_PATH, dtype=torch.float64)
```

Keyword arguments of `load`:

| Argument | Default | Effect |
|----------|---------|--------|
| `format` | `None` | `"urdf"` or `"mjcf"` forces a parser instead of reading the suffix |
| `root_joint` | `None` | root the tree on any joint model; `JointFreeFlyer()` adds 7 to `nq` and 6 to `nv` |
| `device`, `dtype` | `None`, `torch.float32` | where and in what precision the values live |

Mesh URIs stay the strings the file wrote; `viewer.robot_meshes(geometry, root=)` reads them under a root you name.

## Build a robot in code with `ModelBuilder`

`ModelBuilder` writes the same `IRModel` without a file. Add bodies, connect them with joints, attach
frames, then call `finalize()`.

```python
import torch
import better_robot as br
from better_robot import ModelBuilder
from better_robot.io import build_model

builder = ModelBuilder("arm", gravity=[0.0, 0.0, -9.81])
builder.add_body("base", mass=2.0)
builder.add_body("link", mass=1.0, com=torch.tensor([0.0, 0.0, 0.2]))
builder.add_revolute("shoulder", axis=(0.0, 0.0, 1.0), parent="base", child="link", lower=-1.5, upper=1.5)
builder.add_frame("tip", parent="link",
                  placement=torch.tensor([0.0, 0.0, 0.4, 0.0, 0.0, 0.0, 1.0]))
model = build_model(builder.finalize())
print(model.nq, model.nv, model.upper_position_limit)
```
```output
1 1 tensor([1.5000])
```

Joint methods: `add_revolute` (about `axis=`, a unit vector of the joint frame), `add_prismatic` (along
`axis=`), `add_helical`, `add_spherical`, `add_planar`, `add_fixed`, `add_free_flyer_root`, and `add_joint(kind=)`
for any `JointModel` instance. Each takes the
parent and child body names and an optional `origin` pose `[tx, ty, tz, qx, qy, qz, qw]` of the joint in
the parent body. `add_shape(body, shape, collision=True)` attaches a `model.Sphere`, `Box`, `Cylinder`,
`Capsule` or `Mesh` (each with its placement in the body frame and an optional colour) to a body.

`gravity=` is the world linear acceleration `(3,)`; the default is `(0, 0, -9.81)`. It becomes
`model.gravity (6,)`, which the [dynamics](dynamics.md) functions read.

## Make any robot floating-base

A legged robot or a humanoid is usually described with a fixed root. Pass `root_joint=JointFreeFlyer()` to
put a six-DoF free-flyer between the world and the root body; `build_model` takes the same argument for
an IR:

```python
floating = br.io.build_model(builder.finalize(), root_joint=br.JointFreeFlyer())
print(floating.nq, floating.nv)              # 8 7
```
```output
8 7
```

The base configuration is `q[:7] = [tx, ty, tz, qx, qy, qz, qw]` and the base velocity is `v[:6]`.
Because `nq != nv`, step configurations with `kinematics.integrate(model, q, v)`; see
[Joints, bodies and frames](joints-bodies-frames.md).

## Choose dtype and device at load time

`load(..., dtype=torch.float64, device="cuda")` creates every value tensor in that precision and place.
`model.to(...)` converts an existing model. Functions never cast your inputs: pass `q` in the model's dtype
and device ([validation contract](architecture.md#inputs-are-validated-once-at-the-boundary)).

API details: [io reference](../reference/api/io.md).

## Where next

- [Model and state](model-and-state.md): what the built `Model` contains.
- [Joints, bodies and frames](joints-bodies-frames.md): the parts a parser fills in.
- [Architecture](architecture.md): where `io` sits in the package layers.
