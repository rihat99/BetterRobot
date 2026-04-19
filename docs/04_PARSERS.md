# 04 · Parsers — URDF / MJCF → IR → `Model`

The parser layer is the **single entry point** for turning external robot
descriptions into BetterRobot `Model` instances. All parsers emit a common
**intermediate representation (IR)**; a single `build_model()` factory
consumes the IR and produces a frozen `Model`. Parsers never import
`kinematics/`, `dynamics/`, or `tasks/`; `Model` never imports `yourdfpy`,
`mujoco`, or any format library.

## 1. Public entry point

```python
# src/better_robot/io/__init__.py
import better_robot as br

model = br.load("robot.urdf")                   # URDF
model = br.load("robot.xml", format="mjcf")     # MJCF
model = br.load(urdf_obj)                       # already-parsed yourdfpy.URDF
model = br.load(build_fn)                       # programmatic builder (callable)
```

`br.load` is a thin suffix/type dispatcher:

```python
def load(
    source: str | Path | yourdfpy.URDF | Callable[[], "IRModel"],
    *,
    format: Literal["auto", "urdf", "mjcf", "builder"] = "auto",
    root_joint: JointModel | None = None,
    free_flyer: bool = False,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> Model: ...
```

### Key arguments

- **`root_joint`** — replace the default root (which is `JointFixed`) with
  an arbitrary joint. Passing `root_joint=JointFreeFlyer()` is what turns a
  "fixed base" URDF into a floating-base robot. This single argument
  eliminates the entire floating-base code path in `tasks/ik`.
- **`free_flyer=True`** — convenience shortcut for
  `root_joint=JointFreeFlyer()`.
- **`device`, `dtype`** — where the resulting `Model` tensors should live.
  Default: CPU / float32.

## 2. The intermediate representation

```python
# src/better_robot/io/ir.py
from dataclasses import dataclass, field
from typing import Optional, Sequence
import torch

@dataclass
class IRJoint:
    name: str
    parent_body: str                    # name of parent link
    child_body:  str                    # name of child link
    kind: str                           # "revolute" | "prismatic" | "fixed" | "ball" | "free" | ...
    axis: Optional[torch.Tensor] = None # (3,) unit axis (for axis-based joints)
    origin: torch.Tensor = field(default_factory=lambda: torch.zeros(7))  # SE3 in parent body frame
    lower: Optional[float] = None
    upper: Optional[float] = None
    velocity_limit: Optional[float] = None
    effort_limit:   Optional[float] = None
    mimic_source: Optional[str] = None
    mimic_multiplier: float = 1.0
    mimic_offset: float = 0.0

@dataclass
class IRBody:
    name: str
    mass: float = 0.0
    com: torch.Tensor = field(default_factory=lambda: torch.zeros(3))
    inertia: torch.Tensor = field(default_factory=lambda: torch.zeros(3, 3))   # 3x3 symmetric
    visual_geoms:   list["IRGeom"] = field(default_factory=list)
    collision_geoms: list["IRGeom"] = field(default_factory=list)

@dataclass
class IRGeom:
    kind: str                           # "sphere" | "box" | "capsule" | "cylinder" | "mesh"
    params: dict                        # radius / size / mesh path / ...
    origin: torch.Tensor                # (7,) SE3 in body frame
    rgba: Optional[tuple[float, float, float, float]] = None

@dataclass
class IRFrame:
    name: str
    parent_body: str
    placement: torch.Tensor             # (7,) SE3 in parent body frame
    frame_type: str = "op"              # "body" | "joint" | "fixed" | "op" | "sensor"

@dataclass
class IRModel:
    name: str
    bodies: list[IRBody]
    joints: list[IRJoint]
    frames: list[IRFrame] = field(default_factory=list)
    root_body: str = ""                  # name of world/base body
    gravity: torch.Tensor = field(default_factory=lambda: torch.tensor([0., 0., -9.81, 0., 0., 0.]))
```

The IR is **flat** and **ordered-unconstrained**. Nothing about BFS
traversal, topological sort, or `idx_q`/`idx_v` assignment lives in the IR
— those are derived by `build_model()`.

## 3. `build_model()` — the IR → `Model` factory

```python
# src/better_robot/io/build_model.py

def build_model(
    ir: IRModel,
    *,
    root_joint: JointModel | None = None,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> Model:
    """Consume an IRModel and return a frozen Model."""
```

Responsibilities:

1. **Replace the root body's parent joint** with `root_joint` if supplied
   (e.g. `JointFreeFlyer`). Default root is `JointFixed`.
2. **Resolve mimic edges** to `mimic_source`/`mimic_multiplier`/`mimic_offset`
   arrays.
3. **Topologically sort** joints so parents precede children.
4. **Assign `idx_q` / `idx_v`** by accumulating `nq_i` / `nv_i`.
5. **Select concrete `JointModel` instances** for each joint based on
   `IRJoint.kind` + `axis`. Example: `kind="revolute", axis=[1,0,0]` →
   `JointRX()`; `kind="revolute", axis=[0.5, 0.5, 0]` →
   `JointRevoluteUnaligned(axis=[0.5, 0.5, 0])`.
6. **Pack per-joint numeric buffers** (`joint_placements`,
   `lower_pos_limit`, `upper_pos_limit`, `velocity_limit`, `effort_limit`).
7. **Pack per-body inertias** into the 10-vector representation.
8. **Build frame list**: include `"body_<name>"` frames for every body by
   default; add anything user-supplied in `ir.frames`.
9. **Build name → id dicts**.
10. **Return `Model(frozen=True)`**.

## 4. `parsers/urdf.py`

```python
# src/better_robot/io/parsers/urdf.py
from ..ir import IRModel, IRBody, IRJoint, IRFrame, IRGeom

def parse_urdf(source: str | Path | "yourdfpy.URDF") -> IRModel:
    """Parse a URDF file or yourdfpy object into an IRModel.

    The URDF dependency is confined to this file; no other module imports
    yourdfpy. Mimic joints, continuous joints, and multi-child links are all
    handled at IR level. Free-flyer is *not* added here — it is added by
    `build_model(root_joint=JointFreeFlyer())` when the user requests it.
    """
```

Key points:

- Emits **exactly one `IRBody`** for every URDF `<link>`, and **exactly one
  `IRJoint`** for every `<joint>`. Bodies without mass get zero inertia.
- URDF `"continuous"` maps to a `JointRevoluteUnbounded` at the
  `build_model()` step; at IR level it's just `kind="continuous"`.
- `<mimic>` tags become `mimic_source`/`mult`/`off` fields; `build_model`
  resolves them to an index.
- Visual/collision meshes go into `IRBody.visual_geoms` / `collision_geoms`
  as `IRGeom("mesh", {"path": ..., "scale": ...})`. The parser does **not**
  load meshes — that is the collision layer's job.

## 5. `parsers/mjcf.py`

MJCF has strictly more expressive joint syntax than URDF (ball joints,
slider, hinge, free, composite joints, sites), so the MJCF parser fills more
of the IR than the URDF parser:

- `<joint type="ball">` → `IRJoint(kind="spherical")`
- `<joint type="free">` → `IRJoint(kind="free_flyer")`
- `<joint type="hinge" axis="1 0 0">` → `IRJoint(kind="revolute", axis=[1,0,0])`
- `<site>` → `IRFrame(frame_type="op")`
- `<body>` without `<joint>` children → `IRJoint(kind="fixed")`

**MJCF is a first-class input**, not a second-class afterthought. mjlab's
lesson: privileging URDF is a real cost for anyone who wants to target
MuJoCo environments.

Dependency: `mujoco.MjSpec` if available; the parser imports it lazily
inside the function body so `import better_robot` does not require mujoco.

## 6. `parsers/programmatic.py` — the builder DSL

mjlab's best idea: the robot is produced by a *Python function*, not an XML
file.

```python
# src/better_robot/io/parsers/programmatic.py
from ..ir import IRModel, IRBody, IRJoint, IRFrame, IRGeom

class ModelBuilder:
    """Fluent, imperative builder that emits an IRModel.

    Example:
        b = ModelBuilder("my_arm")
        base = b.add_body("base", mass=0.0)
        link1 = b.add_body("link1", mass=1.2, inertia=diag([...]))
        b.add_joint("joint1", kind="revolute_z", parent=base, child=link1,
                    origin=SE3(0,0,0.1), lower=-pi, upper=pi)
        ir = b.finalize()
    """

    def __init__(self, name: str) -> None: ...
    def add_body(self, name: str, *, mass: float = 0.0,
                 com: torch.Tensor = ..., inertia: torch.Tensor = ...) -> str: ...
    def add_joint(self, name: str, *, kind: str, parent: str, child: str,
                  origin: torch.Tensor, axis: torch.Tensor | None = None,
                  lower: float | None = None, upper: float | None = None,
                  mimic_source: str | None = None,
                  mimic_multiplier: float = 1.0,
                  mimic_offset: float = 0.0) -> str: ...
    def add_frame(self, name: str, *, parent_body: str, placement: torch.Tensor,
                  frame_type: str = "op") -> str: ...
    def add_collision_geom(self, body: str, kind: str, params: dict,
                           origin: torch.Tensor) -> None: ...
    def finalize(self) -> IRModel: ...
```

This is how `br.load(build_fn)` works: `build_fn` is a callable that
constructs a `ModelBuilder`, calls its methods, and returns the finalized
`IRModel`. `br.load` then runs `build_model()` on it.

### SMPL-like body as motivation

```python
# src/better_robot/io/builders/smpl_like.py
def make_smpl_like_body(
    height: float = 1.75,
    mass:   float = 70.0,
    *,
    shape_params: torch.Tensor | None = None,
) -> IRModel:
    """Build an SMPL-skeleton-topology IRModel with fixed shape parameters.

    This is NOT an SMPL loader — it constructs a kinematic tree with the
    same 24-joint topology as SMPL (pelvis as free-flyer root, plus 23 ball
    joints for the body), with body dimensions derived from the shape
    parameters.

    The purpose is to prove the data model is expressive enough to host an
    SMPL-like body without introducing any SMPL-specific code in the core.
    """
    b = ModelBuilder("smpl_body")
    pelvis = b.add_body("pelvis", mass=...)
    b.add_joint("root", kind="free_flyer", parent="world", child=pelvis, origin=SE3_identity())
    # ... left_hip, right_hip, spine, ... all JointSpherical
    return b.finalize()
```

This module is the v1 correctness test for "can users build an SMPL body?"
without being an SMPL loader.

## 7. Registration and dispatch

```python
# src/better_robot/io/__init__.py
from typing import Callable
from pathlib import Path
from .ir import IRModel
from .build_model import build_model
from .parsers.urdf import parse_urdf
from .parsers.mjcf import parse_mjcf

_PARSERS: dict[str, Callable[..., IRModel]] = {
    "urdf": parse_urdf,
    "mjcf": parse_mjcf,
}

def register_parser(suffix: str, fn: Callable[..., IRModel]) -> None:
    """Register a new format parser at runtime."""
    _PARSERS[suffix] = fn

def load(source, *, format="auto", **kwargs) -> "Model":
    if callable(source):
        ir = source()
    else:
        suffix = _detect_format(source, hint=format)
        ir = _PARSERS[suffix](source)
    return build_model(ir, **kwargs)
```

## 8. What disappears from the current codebase

- `src/better_robot/models/parsers/urdf.py` and `_urdf_impl.py` →
  rewritten into `src/better_robot/io/parsers/urdf.py`.
- `RobotURDFParser` → gone. The URDF parser becomes a module-level function
  that returns an `IRModel`, then hands it to `build_model()`.
- `from_urdf` as a `classmethod` on `RobotModel` → gone. There is one
  construction entry point, `build_model(ir)`, and one user-facing loader,
  `load()`.
- `yourdfpy` imports anywhere outside `io/parsers/urdf.py` → removed.

## 9. Extensibility notes

- Any new format (SDF, xacro, OSIM, custom JSON) is a new parser under
  `io/parsers/` that emits an `IRModel`. No changes to `Model`, `Data`, or
  `build_model` beyond registering the suffix dispatch.
- Any new joint kind (`helical_unaligned`, `custom_spline_joint`) requires
  (a) a new `JointModel` implementation under
  `data_model/joint_models/`, and (b) a new `kind` string the IR parser
  emits. `build_model` maps `kind` strings to `JointModel` constructors
  through a small dispatch table kept next to `build_model` itself.
- Geometry/mesh resolution is **deferred** until the collision layer asks
  for it. The IR just carries file paths. `io/` never loads a mesh.

## 10. Error handling

- `build_model()` is strict: missing inertia for a non-root body, cycles in
  the joint graph, mimic source that does not exist, or an unknown joint
  `kind` are all raised as `IRError` at build time, never as silent defaults.
- Parsers are *permissive* where the spec itself is ambiguous (e.g. URDF
  joints without a `<limit>` on a revolute → infinite limits; URDF continuous
  joints get a sentinel "continuous" kind; missing masses → zero with a
  warning the first time a user enables dynamics).
