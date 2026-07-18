# Parsers and the IR

URDF, MJCF, and programmatic construction all meet at an in-process
intermediate representation, `IRModel`.  Format-specific code produces the
IR; {py:func}`better_robot.io.build_model` sorts and validates its topology,
selects joint implementations, packs tensors, and returns a frozen
`better_robot.Model`.

The IR is a build boundary, not a persistence format.  Its serialized or
pickled shape has no compatibility guarantee; re-parse the source description
after an upgrade.

## Loading a model

The normal entry point is `better_robot.load`:

```python
import better_robot as br

urdf_model = br.load("robot.urdf")
mjcf_model = br.load("robot.xml")       # .xml maps to the MJCF parser
floating = br.load("robot.urdf", free_flyer=True)
```

Its current signature is:

```python
def load(
    source: str | Path | Any | Callable[[], IRModel],
    *,
    format: Literal["auto", "urdf", "mjcf", "builder"] = "auto",
    root_joint: JointModel | None = None,
    free_flyer: bool = False,
    preserve_joint_order: bool = False,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> Model: ...
```

For paths, `format="auto"` lowercases the suffix without its leading dot and
looks it up in the parser registry.  The built-in keys are `"urdf"`, `"mjcf"`,
and `"xml"`.  A `yourdfpy.URDF` object is also accepted.  A non-path callable
is called with no arguments and must return an `IRModel`:

```python
def make_ir() -> IRModel:
    builder = ModelBuilder("arm")
    # fill builder ...
    return builder.finalize()

model = br.load(make_ir)
```

Callable detection happens before the `format` hint, so callers should use a
callable directly rather than rely on `format="builder"` for a path.

`free_flyer=True` supplies `JointFreeFlyer()` only when `root_joint` is not
already provided.  `preserve_joint_order=True` selects the stable Kahn
topological sort; the default remains the historical DFS body traversal.
Already-topological input keeps its source order under the stable path, while
non-topological input is repaired deterministically.  `Model.q_permutation`
can describe the resulting source-to-model coordinate gather.

`load` does **not** have a `resolver=` parameter.  Use the direct parser plus
`build_model` when a custom asset resolver is required; see
{ref}`asset-resolver-behavior`.

## The intermediate representation

The dataclasses in `src/better_robot/io/ir.py` have these fields:

```python
@dataclass
class IRJoint:
    name: str
    parent_body: str
    child_body: str
    kind: str
    axis: torch.Tensor | None = None
    origin: torch.Tensor = field(default_factory=lambda: torch.zeros(7))
    lower: float | None = None
    upper: float | None = None
    velocity_limit: float | None = None
    effort_limit: float | None = None
    mimic_source: str | None = None
    mimic_multiplier: float = 1.0
    mimic_offset: float = 0.0
    pitch: float = 0.0
    joint_model: JointModel | None = None

@dataclass
class IRGeom:
    kind: str
    params: dict
    origin: torch.Tensor
    rgba: tuple[float, float, float, float] | None = None

@dataclass
class IRBody:
    name: str
    mass: float = 0.0
    com: torch.Tensor = field(default_factory=lambda: torch.zeros(3))
    inertia: torch.Tensor = field(default_factory=lambda: torch.zeros(3, 3))
    visual_geoms: list[IRGeom] = field(default_factory=list)
    collision_geoms: list[IRGeom] = field(default_factory=list)

@dataclass
class IRFrame:
    name: str
    parent_body: str
    placement: torch.Tensor
    frame_type: str = "op"

@dataclass
class IRModel:
    name: str
    bodies: list[IRBody]
    joints: list[IRJoint]
    frames: list[IRFrame] = field(default_factory=list)
    root_body: str = ""
    gravity: torch.Tensor = field(
        default_factory=lambda: torch.tensor(
            [0.0, 0.0, -9.81, 0.0, 0.0, 0.0]
        )
    )
    meta: dict = field(default_factory=dict)
```

The default gravity tensor is `[0, 0, -9.81, 0, 0, 0]`.  `joint_model` is a
programmatic-builder payload; file parsers leave it as `None`.  The IR list
order need not be topological because ordering and coordinate offsets are
assigned by `build_model`.

## `build_model`

```python
def build_model(
    ir: IRModel,
    *,
    root_joint: JointModel | None = None,
    preserve_joint_order: bool = False,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> Model: ...
```

The factory:

1. identifies an explicit `parent_body="world"` root edge or inserts a
   synthetic fixed root joint;
2. sorts regular joints by DFS or the opt-in stable Kahn path;
3. maps each `IRJoint.kind` (or its programmatic `joint_model`) to a concrete
   `JointModel`;
4. builds full and public `q`/`v` layouts, including affine reduction for
   supported scalar mimic joints;
5. packs placements, limits, inertias, gravity, and the mimic maps;
6. creates `body_<name>` frames and appends explicit `IRFrame` entries;
7. checks the packed topology/layout invariants; and
8. returns a frozen `Model`, with the source IR in `model.meta["ir"]` and
   entries from `ir.meta` copied alongside it.

Structural errors such as disconnected topology, multiple world-root edges,
unknown joint kinds, invalid mimic references, or mimic cycles raise
`IRError` or a more specific documented validation error.  Missing URDF
inertial data is **not** one of those errors: the URDF parser initializes
missing mass, COM, and inertia values to zero without a warning, and
`build_model` accepts those zero values.  It also uses a zero inertia row when
an expected IR body record is absent.

## URDF parsing

The direct parser signature is:

```python
def parse_urdf(
    source: str | Path | Any,
    *,
    resolver: AssetResolver | None = None,
) -> IRModel: ...
```

`source` may be a path or an already-loaded `yourdfpy.URDF` object.  The
parser emits one `IRBody` for each entry in `urdf.link_map` and one `IRJoint`
for each entry in `urdf.joint_map`.  It records joint axes, origins, limits,
and mimic metadata.  URDF `continuous` remains `kind="continuous"` in the IR
and becomes a `JointRevoluteUnbounded` during `build_model`.

Sphere, box, cylinder, capsule, and mesh visual/collision elements are copied
to `IRGeom` metadata.  Mesh files are not loaded by this parser.  When
available, the `yourdfpy` object's own filename handler may rewrite its mesh
path before it is stored.

`yourdfpy` is imported when `parse_urdf` runs (and by `load` while checking
whether a non-callable source is a `yourdfpy.URDF`).  A missing dependency at
the direct parser boundary is reported as `BackendNotAvailableError`.

## MJCF parsing

The direct parser signature is:

```python
def parse_mjcf(
    source: str | Path,
    *,
    resolver: AssetResolver | None = None,
) -> IRModel: ...
```

It uses `mujoco.MjSpec.from_file` and imports `mujoco` only when the parser is
called.  The current lowering covers bodies; hinge, slide, ball, and free
joints; sites as operational frames; and sphere, capsule, cylinder, and box
collision metadata.  A non-root body with no explicit joint receives a fixed
joint.

MJCF mesh, plane, height-field, and ellipsoid geometry, visual geometry,
tendons, and actuators are not lowered.  Multiple MJCF joints on one body are
not combined into a composite joint; the emitted parallel edges are rejected
later by the tree builder.  This is a current parser limitation, not supported
composite-joint behavior.

## Programmatic construction

`better_robot.io.ModelBuilder` is the fluent IR builder.  This is a
minimal complete example using the actual tensor-valued placement API:

```python
import torch
from better_robot.io import ModelBuilder, build_model

identity = torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])

builder = ModelBuilder("one_link")
base = builder.add_body("base")
tip = builder.add_body("tip", mass=1.2, inertia=torch.eye(3))
builder.add_revolute_z(
    "joint",
    parent=base,
    child=tip,
    origin=identity,
    lower=-3.14,
    upper=3.14,
)
model = build_model(builder.finalize())
```

The public methods and their current signatures are:

```python
ModelBuilder(name: str)

add_body(name, *, mass=0.0, com=None, inertia=None) -> str
add_frame(name, *, parent_body, placement, frame_type="op") -> str
add_collision_geom(body, kind, params, origin) -> None

add_revolute(name, *, parent, child, axis, origin=None,
             lower=None, upper=None, velocity_limit=None,
             effort_limit=None, unbounded=False, mimic_source=None,
             mimic_multiplier=1.0, mimic_offset=0.0) -> str
add_revolute_x(name, **kwargs) -> str
add_revolute_y(name, **kwargs) -> str
add_revolute_z(name, **kwargs) -> str

add_prismatic(name, *, parent, child, axis, origin=None,
              lower=None, upper=None, velocity_limit=None,
              effort_limit=None) -> str
add_prismatic_x(name, **kwargs) -> str
add_prismatic_y(name, **kwargs) -> str
add_prismatic_z(name, **kwargs) -> str

add_spherical(name, *, parent, child, origin=None) -> str
add_planar(name, *, parent, child, origin=None) -> str
add_helical(name, *, parent, child, axis, pitch, origin=None,
            lower=None, upper=None) -> str
add_free_flyer_root(name="free_flyer", *, child, origin=None) -> str
add_fixed(name, *, parent, child, origin=None) -> str

add_joint(name, *, kind=None, parent, child, origin=None, axis=None,
          lower=None, upper=None, velocity_limit=None, effort_limit=None,
          mimic_source=None, mimic_multiplier=1.0,
          mimic_offset=0.0) -> str
finalize() -> IRModel
```

The axis-specific helpers forward their keyword arguments to the generic
revolute or prismatic method.  The catch-all `add_joint` requires
`kind=<JointModel instance>`; strings and `None` raise `TypeError`.
`add_collision_geom` records IR metadata only and does not make collision
queries operational.  `finalize` requires exactly one root-body candidate;
the deeper topology and joint validation occurs in `build_model`.

`better_robot.io.builders` also exports array-driven kinematic-tree builders
and `make_smpl_like_body` / `make_smpl_like_model`.  The SMPL-like helpers
construct a 24-joint topology but are not SMPL mesh, pose, or shape loaders.
`joint_offsets` and explicit per-body inertial arguments affect construction;
the currently accepted `shape_params` argument is not read by the
implementation.

(asset-resolver-behavior)=
## Asset resolvers and their current consumers

The runtime-checkable Protocol in `src/better_robot/io/assets.py` has one
method:

```python
@runtime_checkable
class AssetResolver(Protocol):
    def resolve(self, uri: str) -> Path: ...
```

There is no `exists` method and no per-call `base_path` keyword.  A failed
resolution is represented by `FileNotFoundError`.

| Resolver | Constructor and behavior |
|----------|--------------------------|
| `FilesystemResolver` | `FilesystemResolver(base_path)` resolves a path against that fixed directory and rejects URI schemes. |
| `PackageResolver` | `PackageResolver(packages)` resolves `package://<name>/<path>` through the explicit mapping. |
| `CompositeResolver` | `CompositeResolver(resolvers)` tries children in order until one does not raise `FileNotFoundError`. |
| `CachedDownloadResolver` | `CachedDownloadResolver(cache_dir)` creates the caller-selected cache directory and downloads `http(s)` URLs by their final path component. No default cache directory is supplied. |

The direct URDF and MJCF parsers accept `resolver=`.  For a path source they
construct `FilesystemResolver(Path(source).parent)` when none is supplied.
The parser stores that object in `ir.meta["asset_resolver"]`; `build_model`
copies it to `model.meta["asset_resolver"]`.

The parsers do not call `resolver.resolve` while lowering geometry.  The
implemented downstream consumer is `URDFMeshMode` in the viewer: it consults
the stored resolver for mesh paths and falls back to the raw path if resolution
fails.  The collision package does not currently consume parser geometry or
asset resolvers.  To install a custom resolver, bypass `load`:

```python
from better_robot.io import build_model
from better_robot.io.parsers import parse_urdf

ir = parse_urdf("robot.urdf", resolver=my_resolver)
model = build_model(ir)
```

## Runtime parser registration

Registration is a function call, not a decorator:

```python
def register_parser(
    suffix: str,
    fn: Callable[..., IRModel],
) -> None: ...
```

Keys must match the lowercase suffix **without** a leading dot:

```python
from better_robot.io import register_parser

def parse_sdf(source) -> IRModel:
    ...

register_parser("sdf", parse_sdf)
model = br.load("robot.sdf")
```

This extends automatic suffix dispatch at runtime.  The static `format` type
annotation still lists only `auto`, `urdf`, `mjcf`, and `builder`, so custom
formats are most naturally selected by their registered path suffix.

## Sharp edges

- `IRJoint.origin` defaults to seven zeros, not an identity quaternion.  The
  shipped parsers and `ModelBuilder` helpers normally provide an explicit
  identity pose when no source origin exists.
- Missing URDF inertial fields become zeros silently.
- `load` cannot forward parser-specific options such as `resolver=`.
- Parser-emitted collision geometry is metadata only while the collision
  computation surface remains reserved; see {doc}`collision_and_geometry`.
- Re-parse source descriptions instead of persisting the IR across versions.

## Where to look next

- {doc}`model_and_data` — the frozen model and mutable data/cache split.
- {doc}`/conventions/extension` §9 — runtime parser registration.
- {doc}`/conventions/extension` §13 — custom resolver construction.
- {doc}`viewer` — the implemented parser-geometry consumer.
