# Collision and Geometry

The collision package currently defines a **reserved API**, not a working
collision pipeline.  Its tensor-valued primitive dataclasses can be used as
containers, and `register_pair` records a function in the private dispatch
table.  Distance evaluation, closest-point kernels, robot decomposition,
collision penalties, residual evaluation, and task integration are all
deferred and raise `NotImplementedError`.

This distinction matters at the task boundary: `better_robot.solve_ik`
does not accept collision geometry or construct collision residuals.  See
{doc}`/reference/roadmap` for the explicit-raise inventory.

## Shipped source layout

```
src/better_robot/collision/
├── __init__.py
├── geometry.py         # primitive containers; penalty stub
├── pairs.py            # registration helper; distance stub
├── closest_pts.py      # closest-point stubs
└── robot_collision.py  # reserved RobotCollision surface
```

There is no `collision/broadphase.py` today.  AABB pruning, spatial hashing,
BVHs, mesh fitting, and collision-specific mesh loading are not implemented.

## Usable primitive containers

`geometry.py` exports five frozen dataclasses:

```python
@dataclass(frozen=True)
class Sphere:
    center: torch.Tensor
    radius: torch.Tensor

@dataclass(frozen=True)
class Capsule:
    a: torch.Tensor
    b: torch.Tensor
    radius: torch.Tensor

@dataclass(frozen=True)
class Box:
    center: torch.Tensor
    half_extents: torch.Tensor
    rotation: torch.Tensor

@dataclass(frozen=True)
class HalfSpace:
    normal: torch.Tensor
    offset: torch.Tensor

@dataclass(frozen=True)
class Plane(HalfSpace):
    pass
```

The comments in source describe `center`, `a`, `b`, `half_extents`, and
`normal` with a trailing dimension of three, and `rotation` with a trailing
quaternion dimension of four.  The constructors themselves perform no shape,
dtype, device, normalization, or broadcasting validation.  `Plane` is a thin
subclass of `HalfSpace`, not a separate distance implementation.

## Reserved pair and closest-point computation

The registration helper is implemented:

```python
def register_pair(type_a: type, type_b: type):
    ...
```

It returns a decorator that stores the decorated callable under
`(type_a, type_b)` in the private `_PAIR` dictionary.  The public evaluator is
still a stub:

```python
def distance(a, b) -> torch.Tensor:
    raise NotImplementedError(...)
```

No sphere/sphere, sphere/capsule, capsule/capsule, half-space, or box kernel is
registered by the package.  `closest_pts.py` declares only these two reserved
helpers, both of which raise `NotImplementedError`:

```python
def point_to_segment(p: torch.Tensor, a: torch.Tensor,
                     b: torch.Tensor) -> torch.Tensor: ...

def segment_to_segment(a1: torch.Tensor, b1: torch.Tensor,
                       a2: torch.Tensor, b2: torch.Tensor) -> torch.Tensor: ...
```

Consequently the package makes no current numerical, vectorization, or
autograd guarantee for collision distances.

## Reserved `RobotCollision` surface

`RobotCollision` is an importable mutable dataclass with these fields:

```python
@dataclass
class RobotCollision:
    frame_ids: tuple[int, ...]
    local_a: torch.Tensor
    local_b: torch.Tensor
    radii: torch.Tensor
    self_pairs: torch.Tensor
    allowed_pairs_mask: torch.Tensor
```

The intended method signatures are pinned, but every method raises
`NotImplementedError`:

```python
@classmethod
def from_model(
    cls,
    model: Model,
    *,
    mode: Literal["capsule", "sphere"] = "capsule",
    allow_adjacent: bool = False,
) -> "RobotCollision": ...

def world_capsules(self, data: Data) -> Capsule: ...
def self_distances(self, data: Data) -> torch.Tensor: ...
def world_distances(
    self,
    data: Data,
    world: Sequence[Sphere | Capsule | Box],
) -> torch.Tensor: ...
```

In particular, `from_model` does not accept `resolver=`, does not inspect
`Model.meta`, and does not fit or cache capsules.  There is no implemented
`update` method.

## Reserved penalty; no residual exports

The following function is declared but not evaluated:

```python
def colldist_from_sdf(d: torch.Tensor, margin: float) -> torch.Tensor:
    raise NotImplementedError(...)
```

Its docstring records an intended piecewise penalty, but that formula is
design intent rather than shipped behavior.

BetterRobot does not export `SelfCollisionResidual` or
`WorldCollisionResidual`. Their former constructors advertised no executable
behavior and were removed. A real residual surface waits on the owner-gated
collision-package decision, including stable output-shape and Jacobian
contracts.

## Meshes and asset resolvers

The URDF parser can preserve collision geometry as `IRGeom` metadata, but the
collision package does not consume that metadata today.  It does not import
`trimesh`, invoke a `better_robot.io.AssetResolver`, or load mesh
files.  The viewer is currently the only in-tree consumer that resolves and
loads parser-emitted visual meshes; see {doc}`viewer` and
{doc}`parsers_and_ir`.

## Where to look next

- {doc}`residuals_and_costs` — implemented residual and cost-stack behavior.
- {doc}`tasks` — the currently supported `solve_ik` task surface.
- {doc}`/reference/roadmap` — collision symbols that deliberately raise.
