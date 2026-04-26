# Parsers and the IR

A robot description format (URDF, MJCF, a Python builder) is not the
same thing as a `Model`. The format is whatever the asset authoring
tool happened to produce; the `Model` is the frozen, typed, batched,
device-polymorphic object the rest of BetterRobot builds on. The
glue between them is the `io/` layer.

The temptation, every time a new format request comes in, is to add
"parse this format" to whatever class loads URDFs. That path leads to
a `Robot` class that knows about every XML dialect, every mesh
loader, every package-path convention. The first time you want to
load a robot from a non-XML source — a programmatic builder, a
generated SMPL-like body, a JSON description from a research paper —
you discover the parser is the entry point and there is no clean way
to extend it.

We solved this by making the format-specific parsers all converge on
the same intermediate representation: an `IRModel` dataclass with
`schema_version`. Every parser produces an `IRModel`; one factory —
`build_model()` — turns any `IRModel` into a frozen `Model`.
Adding a new format is one parser file plus a registration; the
factory does not need to change. URDF and MJCF live as siblings; a
programmatic builder lives next to them; a generated SMPL skeleton
lives in `io/builders/`.

## The single entry point

```python
import better_robot as br

model = br.load("robot.urdf")                   # URDF
model = br.load("robot.xml", format="mjcf")     # MJCF
model = br.load(urdf_obj)                       # already-parsed yourdfpy.URDF
model = br.load(build_fn)                       # programmatic builder (callable)
```

`br.load` is a thin suffix / type dispatcher:

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

Source: `src/better_robot/io/__init__.py`.

The two arguments worth knowing:

- **`free_flyer=True`** is shorthand for
  `root_joint=JointFreeFlyer()`. It is what turns a fixed-base URDF
  into a floating-base robot. This single argument eliminates the
  entire "fixed vs floating base" code path elsewhere.
- **`root_joint`** lets you replace the default root with any
  `JointModel`. Most users want `JointFixed` (the default) or
  `JointFreeFlyer`; advanced users can plug in a custom joint kind
  for things like an underactuated rolling base.

## The intermediate representation

```python
@dataclass
class IRJoint:
    name: str
    parent_body: str
    child_body:  str
    kind: str                              # "revolute" | "prismatic" | "fixed" | "ball" | "free" | ...
    axis: torch.Tensor | None = None
    origin: torch.Tensor                   # (7,) SE3 in parent body frame
    lower: float | None = None
    upper: float | None = None
    velocity_limit: float | None = None
    effort_limit:   float | None = None
    mimic_source: str | None = None
    mimic_multiplier: float = 1.0
    mimic_offset: float = 0.0

@dataclass
class IRBody:
    name: str
    mass: float = 0.0
    com: torch.Tensor                      # (3,)
    inertia: torch.Tensor                  # (3, 3) symmetric
    visual_geoms:    list[IRGeom]
    collision_geoms: list[IRGeom]

@dataclass
class IRGeom:
    kind: str                              # "sphere" | "box" | "capsule" | "cylinder" | "mesh"
    params: dict
    origin: torch.Tensor                   # (7,) SE3 in body frame
    rgba: tuple[float, float, float, float] | None = None

@dataclass
class IRFrame:
    name: str
    parent_body: str
    placement: torch.Tensor                # (7,) SE3 in parent body frame
    frame_type: str = "op"

@dataclass
class IRModel:
    schema_version: int = 1
    name: str
    bodies: list[IRBody]
    joints: list[IRJoint]
    frames: list[IRFrame]
    root_body: str
    gravity: torch.Tensor                  # (6,) world spatial acceleration
    meta: dict                             # transit slot for parser hints (e.g. asset_resolver)
```

Source: `src/better_robot/io/ir.py`.

The IR is **flat** and **ordered-unconstrained**. Topological sort,
`idx_q` / `idx_v` assignment, mimic resolution, joint-kind dispatch
to concrete `JointModel` instances — none of that lives in the IR.
`build_model()` derives all of it.

### `schema_version`

`IRModel.schema_version` is the controlled change vector: a single
integer counter that bumps in the same PR that changes the IR shape,
with a one-line entry in `CHANGELOG.md`. Cached IR dumps (the `.npz`
fixtures under `tests/io/`) carry the version; old fixtures
regenerate when the version bumps.

```python
def build_model(ir: IRModel, *, free_flyer: bool = False) -> Model:
    if ir.schema_version != IRModel.schema_version:
        raise IRSchemaVersionError(...)
    ...
```

Documented in {doc}`/conventions/contracts` §2 as a typed exception.

## `build_model` — the IR → `Model` factory

```python
def build_model(
    ir: IRModel,
    *,
    root_joint: JointModel | None = None,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> Model:
    """Consume an IRModel and return a frozen Model."""
```

Source: `src/better_robot/io/build_model.py`.

Responsibilities, in order:

1. Replace the root body's parent joint with `root_joint` if supplied
   (default: `JointFixed`).
2. Resolve mimic edges to `mimic_source` / `mimic_multiplier` /
   `mimic_offset` arrays (the gather trick from
   {doc}`joints_bodies_frames`).
3. Topologically sort joints so parents precede children.
4. Assign `idx_q` / `idx_v` by accumulating per-joint dimensions.
5. Select concrete `JointModel` instances based on `IRJoint.kind`
   plus `axis`. `kind="revolute", axis=[1,0,0]` becomes `JointRX()`;
   `kind="revolute", axis=[0.5, 0.5, 0]` becomes
   `JointRevoluteUnaligned(axis=[0.5, 0.5, 0])`.
6. Pack per-joint numeric buffers (`joint_placements`,
   `lower_pos_limit`, `upper_pos_limit`, `velocity_limit`,
   `effort_limit`).
7. Pack per-body inertias into the 10-vector representation.
8. Build the frame list (default `body_<name>` frames per body, plus
   anything user-supplied in `ir.frames`).
9. Build name → id dicts.
10. Return `Model(frozen=True)`.

`build_model` is strict about violating invariants: missing inertia
on a non-root body, cycles in the joint graph, a mimic source that
does not exist, or an unknown joint `kind` all raise `IRError` at
build time. It is permissive where the format itself is ambiguous —
URDF joints without a `<limit>` on a revolute become unbounded; URDF
continuous joints get a sentinel `"continuous"` kind; missing masses
become zero with a warning.

## The URDF parser

```python
def parse_urdf(source: str | Path | "yourdfpy.URDF") -> IRModel:
    """Parse a URDF file or yourdfpy object into an IRModel."""
```

Source: `src/better_robot/io/parsers/urdf.py`.

Key points:

- Emits exactly one `IRBody` per URDF `<link>` and exactly one
  `IRJoint` per `<joint>`. Bodies without mass get zero inertia.
- URDF `"continuous"` becomes `JointRevoluteUnbounded` at the
  `build_model` step; at IR level it is just `kind="continuous"`.
- `<mimic>` tags become `mimic_source` / `mult` / `off` fields;
  `build_model` resolves them to indices.
- Visual / collision meshes go into `IRBody.visual_geoms` /
  `collision_geoms` as `IRGeom("mesh", {"path": ..., "scale": ...})`.
  The parser does not load meshes — that is the collision layer's
  job.
- `yourdfpy` is a lazy import — pulled in only when `parse_urdf` is
  actually called. A user who never parses a URDF never pays the
  cost.

## The MJCF parser

```python
def parse_mjcf(source: str | Path) -> IRModel: ...
```

Source: `src/better_robot/io/parsers/mjcf.py`.

MJCF has strictly more expressive joint syntax than URDF (ball
joints, slider, hinge, free, composite joints, sites), so the MJCF
parser fills more of the IR than the URDF parser:

- `<joint type="ball">` → `IRJoint(kind="spherical")`
- `<joint type="free">` → `IRJoint(kind="free_flyer")`
- `<joint type="hinge" axis="1 0 0">` → `IRJoint(kind="revolute",
  axis=[1,0,0])`
- `<site>` → `IRFrame(frame_type="op")`
- `<body>` without `<joint>` children → `IRJoint(kind="fixed")`

MJCF is a first-class input, not a second-class afterthought —
mjlab's lesson, which we took to heart. Dependency: `mujoco.MjSpec`,
imported lazily — pulled in only when `parse_mjcf` is actually called.

## The programmatic builder

The third path: the robot is produced by a Python function, not an
XML file.

```python
class ModelBuilder:
    """Fluent, imperative builder that emits an IRModel.

    Prefer the named ``add_*`` helpers below. The catch-all
    ``add_joint(kind=JointModel-instance)`` is kept for advanced uses
    (custom joint kinds registered via the JointModel extension seam).

    Example:
        b = ModelBuilder("my_arm")
        base  = b.add_body("base", mass=0.0)
        link1 = b.add_body("link1", mass=1.2, inertia=diag([...]))
        b.add_revolute_z(name="joint1", parent=base, child=link1,
                         origin=SE3.from_translation([0, 0, 0.1]),
                         lower=-pi, upper=pi)
        ir = b.finalize()
    """

    def __init__(self, name: str) -> None: ...
    def add_body(self, name: str, *, mass: float = 0.0,
                 com: torch.Tensor = ..., inertia: torch.Tensor = ...) -> str: ...

    # Named joint helpers (the documented API)
    def add_fixed(self, name, parent, child, *, origin=None) -> str: ...
    def add_revolute(self, name, parent, child, axis, **kw) -> str: ...
    def add_revolute_x(self, name, parent, child, **kw) -> str: ...
    def add_revolute_y(self, name, parent, child, **kw) -> str: ...
    def add_revolute_z(self, name, parent, child, **kw) -> str: ...
    def add_prismatic_x(self, name, parent, child, **kw) -> str: ...
    def add_prismatic_y(self, name, parent, child, **kw) -> str: ...
    def add_prismatic_z(self, name, parent, child, **kw) -> str: ...
    def add_spherical(self, name, parent, child, **kw) -> str: ...
    def add_planar   (self, name, parent, child, **kw) -> str: ...
    def add_free_flyer_root(self, name, child, **kw) -> str: ...
    def add_helical(self, name, parent, child, axis, pitch, **kw) -> str: ...

    # Catch-all for custom joint kinds (JointModel instances only)
    def add_joint(self, *, name, parent, child,
                  kind: "JointModel",
                  origin=None, lower=None, upper=None, ...) -> str: ...

    def add_frame(self, name, *, parent_body, placement, frame_type="op") -> str: ...
    def add_collision_geom(self, body, kind, params, origin) -> None: ...

    def finalize(self) -> IRModel: ...
```

Source: `src/better_robot/io/parsers/programmatic.py`.

`ModelBuilder` is the entry point for any "robot defined in Python"
case — a researcher generating a robot with parameterised geometry,
a sample SMPL-like body, a humanoid built from joint primitives.
`add_joint(kind=...)` accepts a `JointModel` *instance*, not a
string; this rules out the historical mismatch where
`add_joint(kind="revolute_z", ...)` and
`IRJoint(kind="revolute", axis=(0,0,1))` produced different things.
The named helpers (`add_revolute_z`, `add_free_flyer_root`, …) are the
documented API; passing a string to `add_joint` raises a typed error
pointing at the named form.

`br.load(build_fn)` works by calling `build_fn()`, which constructs a
`ModelBuilder`, fills it in, and returns the finalised `IRModel`.
`br.load` then runs `build_model()` on it.

### Example: SMPL-like body

```python
def make_smpl_like_body(height: float = 1.75, mass: float = 70.0,
                       *, shape_params=None) -> IRModel:
    """Build an SMPL-skeleton-topology IRModel with fixed shape parameters.

    NOT an SMPL loader — constructs a kinematic tree with the same
    24-joint topology as SMPL (pelvis as free-flyer root, plus 23 ball
    joints), with body dimensions derived from the shape parameters.
    """
    b = ModelBuilder("smpl_body")
    pelvis = b.add_body("pelvis", mass=...)
    b.add_free_flyer_root("root", child=pelvis, origin=SE3_identity())
    # ... left_hip, right_hip, spine, ... all JointSpherical
    return b.finalize()
```

Source: `src/better_robot/io/builders/smpl_like.py`. This demonstrates
that the data model is expressive enough to host an SMPL-skeleton
body without introducing any SMPL-specific code in core. The
SMPL-and-muscle extension lives in the sibling
`better_robot_human` package under the `[human]` extra.

## Asset resolution — the `AssetResolver` Protocol

URDF and MJCF mesh paths come in three flavours: absolute filesystem
paths, relative paths against the URDF directory, and ROS-style
`package://<pkg>/<path>` URLs that need a package map. Hand-rolling
that logic in every parser, viewer, and collision-mesh loader is
the historical pain point. The Protocol:

```python
class AssetResolver(Protocol):
    """Resolve a mesh / asset URI to a local file path.

    Parsers and visualisers should use a resolver instead of hand-rolling
    path logic. The resolver may consult a package map, a base directory,
    an embedded asset bundle, or a downloader cache.
    """
    def resolve(self, uri: str, *, base_path: Path | None = None) -> Path: ...
    def exists (self, uri: str, *, base_path: Path | None = None) -> bool: ...
```

Source: `src/better_robot/io/assets.py`.

Concrete resolvers shipped in core (no heavy dependencies):

| Resolver | Use |
|----------|-----|
| `FilesystemResolver` | absolute paths + paths relative to a base dir; the default for `parse_urdf` / `parse_mjcf` |
| `PackageResolver`    | `package://<pkg>/<path>` translation; takes a `{pkg: root}` map |
| `CompositeResolver`  | tries children in order; returns the first hit |
| `CachedDownloadResolver` | for `robot_descriptions`-style packages; downloads to `~/.cache/better_robot/assets/` |

Parsers accept `resolver=...`; if omitted, they build a
`FilesystemResolver` rooted at the source's directory. The resolver
that was used at parse time is stored on `model.meta["asset_resolver"]`
so downstream consumers (the viewer, collision mesh loaders) inherit
it. See {doc}`viewer` for the integration.

## Registration and dispatch

```python
# src/better_robot/io/__init__.py
_PARSERS: dict[str, Callable[..., IRModel]] = {
    "urdf": parse_urdf,
    "mjcf": parse_mjcf,
}

def register_parser(suffix: str, fn: Callable[..., IRModel]) -> None:
    """Register a new format parser at runtime."""
    _PARSERS[suffix] = fn
```

Adding SDF support, Drake YAML support, or a custom JSON description
format is a single-file extension that registers a parser via the
seam in {doc}`/conventions/extension` §9. The IR shape does not
change; `build_model` does not change; existing tests do not break.

## Sharp edges

- **Parsers stay at the boundary.** `yourdfpy` and `mujoco` are only
  imported inside `io/parsers/urdf.py` and `io/parsers/mjcf.py` — never
  from the kinematics, dynamics, or optim layers. The contract test
  `test_optional_imports.py` enforces this on every PR. The discipline
  keeps `import better_robot` light and the layered DAG honest.
- **Meshes are not loaded by parsers.** The IR carries the URI;
  loading happens in `viewer/` and `collision/`. Both go through the
  `AssetResolver`.
- **`schema_version` is a single int.** Bumping it requires a
  CHANGELOG entry and may force regeneration of cached IR `.npz`
  fixtures.

## Where to look next

- {doc}`model_and_data` — what `build_model` produces and how the
  consumer side reads from it.
- {doc}`/conventions/extension` §9 — recipe for adding a new parser
  format.
- {doc}`/conventions/extension` §13 — recipe for a custom asset
  resolver.
