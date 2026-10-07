# Every robot source becomes the same Model

After this page you can load a URDF or MJCF file, build a robot in code with `ModelBuilder`, turn a
fixed-base description into a floating-base robot, and choose the dtype and device of the result.
All paths go through one intermediate representation and one builder function.

## Two steps: parse to IR, build the model

Loading has two stages with one owner each:

1. A parser reads a source and returns an `IRModel` (`better_robot.io.ir`): plain bodies, joints, frames,
   inertias and limits, with no topology decisions made yet.
2. `build_model(ir, root_joint=None, preserve_joint_order=False, device=None, dtype=torch.float32)`
   orders the joints into a tree, assigns `idx_q` and `idx_v`, packs every tensor into `ModelValues`
   and returns a frozen [`Model`](model-and-state.md).

```text
URDF file  --parse_urdf-->  \
MJCF file  --parse_mjcf-->   IRModel  --build_model-->  Model
ModelBuilder.finalize() -->  /
```

`better_robot.load(source, ...)` does both stages. It picks the parser from the file suffix, the argument
type, or an explicit `format="urdf" | "mjcf" | "builder"`.

## Load a robot description

`load` takes a path to a `.urdf` or MJCF `.xml` file. The `robot_descriptions` package provides paths for
many published robots; it is an optional dependency.

```python no-run
import torch
import better_robot as br
from robot_descriptions import panda_description

model = br.load(panda_description.URDF_PATH, dtype=torch.float64)
```

Keyword arguments of `load`:

| Argument | Default | Effect |
|----------|---------|--------|
| `format` | `"auto"` | force a parser instead of guessing |
| `free_flyer` | `False` | root the tree on a `JointFreeFlyer` (adds 7 to `nq`, 6 to `nv`) |
| `root_joint` | `None` | root the tree on any joint model; `free_flyer=True` is shorthand |
| `preserve_joint_order` | `False` | keep an already-topological source joint order instead of depth-first order |
| `device`, `dtype` | `None`, `torch.float32` | where and in what precision the values live |

Mesh and package paths in a URDF are resolved by the asset resolvers in `better_robot.io.assets`.

## Build a robot in code with `ModelBuilder`

`ModelBuilder` writes the same `IRModel` without a file. Add bodies, connect them with joints, attach
frames, then call `finalize()`.

```python
import torch
import better_robot as br
from better_robot.io import ModelBuilder, build_model

builder = ModelBuilder("arm", gravity=[0.0, 0.0, -9.81])
builder.add_body("base", mass=2.0)
builder.add_body("link", mass=1.0, com=torch.tensor([0.0, 0.0, 0.2]))
builder.add_revolute_z("shoulder", parent="base", child="link", lower=-1.5, upper=1.5)
builder.add_frame("tip", parent_body="link",
                  placement=torch.tensor([0.0, 0.0, 0.4, 0.0, 0.0, 0.0, 1.0]))
model = build_model(builder.finalize())
print(model.nq, model.nv, model.upper_pos_limit)
```

Joint methods: `add_revolute` (any axis) and `add_revolute_x/y/z`, `add_prismatic` and
`add_prismatic_x/y/z`, `add_spherical`, `add_planar`, `add_fixed`, `add_free_flyer_root`. Each takes the
parent and child body names and an optional `origin` pose `[tx, ty, tz, qx, qy, qz, qw]` of the joint in
the parent body. `add_collision_geom` attaches collision shapes.

`gravity=` is the world linear acceleration `(3,)`; the default is `(0, 0, -9.81)`. It becomes
`model.gravity (6,)`, which the [dynamics](dynamics.md) functions read.

## Make any robot floating-base

A legged robot or a humanoid is usually described with a fixed root. Pass `free_flyer=True` to put a
six-DoF free-flyer between the world and the root body. `load` also accepts the builder's `finalize`
method as a source:

```python
floating = br.load(builder.finalize, free_flyer=True)
print(floating.nq, floating.nv)              # 8 7
```

The base configuration is `q[:7] = [tx, ty, tz, qx, qy, qz, qw]` and the base velocity is `v[:6]`.
Because `nq != nv`, step configurations with `model.integrate(q, v)`; see
[Joints, bodies and frames](joints-bodies-frames.md).

## Choose dtype and device at load time

`load(..., dtype=torch.float64, device="cuda")` creates every value tensor in that precision and place.
`model.to(...)` converts an existing model. Functions never cast your inputs: pass `q` in the model's dtype
and device ([validation contract](architecture.md#inputs-are-validated-once-at-the-boundary)).

API details: [io reference](../reference/api/io.md).
