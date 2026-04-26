# Collision & Geometry

Collision lives in its own layer, **parallel** to `kinematics/` and
`dynamics/`, and is **optional**: a user who only cares about IK pays
nothing for the collision machinery. The design follows PyRoki's pattern
(capsule-first, pair-dispatch SDF, `RobotCollision` as a separate object).

## 1. Goals

1. One geometry module that knows nothing about robots.
2. One pairwise SDF dispatch table that computes signed distance between
   *any* ordered pair of primitives.
3. One `RobotCollision` layer that sits on top of `Model` and provides
   self-collision and world-collision residuals.
4. Batched over pairs and over the leading `B...` axes.
5. Differentiable — every distance function has an autograd-friendly
   implementation.

## 2. Directory layout

```
src/better_robot/collision/
├── __init__.py
├── geometry.py         # primitive types + geometry utilities
├── pairs.py            # pairwise SDF dispatch table
├── closest_pts.py      # segment-segment, point-capsule, … kernels
├── robot_collision.py  # RobotCollision — the Model-level layer
└── broadphase.py       # (future) AABB pruning, spatial hashing
```

## 3. Primitive geometry

```python
# src/better_robot/collision/geometry.py

@dataclass(frozen=True)
class Sphere:
    """Sphere in body-local frame."""
    center: torch.Tensor        # (..., 3)
    radius: torch.Tensor        # (...,)

@dataclass(frozen=True)
class Capsule:
    """Capsule defined by two endpoints + radius, body-local frame."""
    a: torch.Tensor             # (..., 3) endpoint A
    b: torch.Tensor             # (..., 3) endpoint B
    radius: torch.Tensor        # (...,)

@dataclass(frozen=True)
class Box:
    center: torch.Tensor        # (..., 3)
    half_extents: torch.Tensor  # (..., 3)
    rotation: torch.Tensor      # (..., 4) SO3 quaternion, body-local

@dataclass(frozen=True)
class HalfSpace:
    """ n · x + d >= 0 """
    normal: torch.Tensor        # (..., 3)
    offset: torch.Tensor        # (...,)

@dataclass(frozen=True)
class Plane(HalfSpace):
    """Alias for HalfSpace used for world-ground checks."""
```

All shapes are **tensor-valued** from day one: a `Sphere` with `center`
shape `(B, K, 3)` represents `B*K` spheres, not one. This is how the same
data structure expresses "one sphere" and "one robot's worth of self-collision
spheres" without a second type.

## 4. Pairwise SDF dispatch table

```python
# src/better_robot/collision/pairs.py
from typing import Callable

_PAIR: dict[tuple[type, type], Callable] = {}

def register_pair(type_a, type_b):
    def _inner(fn):
        _PAIR[(type_a, type_b)] = fn
        return fn
    return _inner

def distance(a, b) -> torch.Tensor:
    """Signed distance between primitives `a` and `b`, broadcasting over
    their leading batch shapes. Positive = separated, 0 = touching,
    negative = penetration.
    """
    key = (type(a), type(b))
    fn = _PAIR.get(key) or _PAIR.get((type(b), type(a)))
    if fn is None:
        raise NotImplementedError(f"no SDF for {key}")
    return fn(a, b) if key in _PAIR else fn(b, a)
```

First round of registered pairs:

| Pair | Kernel |
|------|--------|
| (Sphere, Sphere) | `‖c_a - c_b‖ - (r_a + r_b)` |
| (Sphere, Capsule) | `point_to_segment(c_a, c_b.a, c_b.b) - (r_a + r_b)` |
| (Capsule, Capsule) | `segment_to_segment(c_a.a, c_a.b, c_b.a, c_b.b) - (r_a + r_b)` |
| (Sphere, HalfSpace) | `n · c - d - r` |
| (Capsule, HalfSpace) | `min(n · a, n · b) - d - r` |
| (Sphere, Box) | clamped closest-point on box |
| (Capsule, Box) | coarse: capsule → segment, box → segment set; fine path later |

All kernels are **vectorised over the leading batch shape** and
differentiable. Closest-point helpers live in `collision/closest_pts.py`.

## 5. `RobotCollision`

```python
# src/better_robot/collision/robot_collision.py

@dataclass
class RobotCollision:
    """Capsule decomposition of a robot.

    Each collision capsule is attached to a *frame* (not a joint) via a
    local-frame `(a, b, radius)` triple. At query time,
    `RobotCollision.update(model, data)` uses `data.frame_pose_world` to
    transform each capsule into the world frame.
    """
    frame_ids:       tuple[int, ...]            # per capsule
    local_a:         torch.Tensor               # (n_caps, 3)
    local_b:         torch.Tensor               # (n_caps, 3)
    radii:           torch.Tensor               # (n_caps,)
    self_pairs:      torch.Tensor               # (n_pairs, 2) — indices into capsules
    allowed_pairs_mask: torch.Tensor            # (n_caps, n_caps) bool

    @classmethod
    def from_model(cls, model: Model, *, mode: Literal["capsule", "sphere"] = "capsule",
                   allow_adjacent: bool = False) -> "RobotCollision": ...

    def world_capsules(self, data: Data) -> Capsule:
        """Return a (B..., n_caps) Capsule with world-frame endpoints."""

    def self_distances(self, data: Data) -> torch.Tensor:
        """(B..., n_pairs) signed distances of all self-pairs."""

    def world_distances(self, data: Data, world: Sequence[Sphere | Capsule | Box]) -> torch.Tensor:
        """(B..., n_caps, len(world)) signed distances against external geometry."""
```

### Construction: capsule mode

`RobotCollision.from_model(model, mode="capsule")`:

1. For every body with visual / collision geometry, choose a covering
   capsule (mesh → capsule approximation is the expensive case; URDF
   `<cylinder>` / `<box>` capsule fallbacks are trivial).
2. Record the capsule in the **frame of the body** (so the local
   coordinates do not change when the robot moves).
3. Build the set of *allowed pairs* (exclude adjacent links by default;
   expose an override for "I really want to detect elbow-shoulder
   self-collision").
4. Return a single `RobotCollision` whose tensors live on the model's
   device/dtype.

Capsule fitting is **offline** and cached — users construct one
`RobotCollision` per robot once per session.

## 6. The residual side

```python
# src/better_robot/residuals/collision.py

@register_residual("self_collision")
class SelfCollisionResidual(Residual):
    """(B..., n_candidate_pairs) — **stable dim across iterations**.

    `dim = number_of_candidate_pairs`. Pairs outside the safety margin
    contribute zero — but the slot exists, so the Jacobian has a
    corresponding row of zeros. Active-pair compaction is a
    *kernel-internal* optimisation; the public residual dim does not
    fluctuate. This keeps LM's preallocated buffers and damping stable
    across iterations
    (see `07_RESIDUALS_COSTS_SOLVERS.md §10 <07_RESIDUALS_COSTS_SOLVERS.md>`_).

    The residual is:
        r_p = -colldist_from_sdf(d_p, margin) * weight   if d_p < margin
        r_p = 0                                           otherwise

    Analytic Jacobian:
        Available as a **sparse** block with nonzero entries only in the
        kinematic chains of the two involved capsules. ``ResidualSpec``
        (see 07 §7) declares the sparsity:
            structure="block",
            affected_joints=tuple of joints in the kinematic chains,
            dynamic_dim=True   # active subset varies; output_dim stable.
    """

    def __init__(self, model: Model, robot_collision: RobotCollision, *,
                 margin: float = 0.02, weight: float = 1.0): ...

    def __call__(self, state: ResidualState) -> torch.Tensor: ...
    def jacobian(self, state: ResidualState) -> torch.Tensor | None: ...   # sparse analytic
    def spec(self, state: ResidualState) -> "ResidualSpec":
        """structure='block'; dynamic_dim=True; affected_joints=…"""
    def apply_jac_transpose(self, state, vec):
        """Sparse J^T @ vec; skips inactive rows internally."""
```

Important: the residual is the only object the task layer sees. The
analytic Jacobian lives inside the residual, not in the IK solver — this is
how the current `_analytic_collision_jacobian` stops being a special case
inside `tasks/ik/solver.py`.

## 7. `colldist_from_sdf`

The smoothing function used to turn signed distances into soft penalty
residuals lives in `collision/geometry.py`:

```python
def colldist_from_sdf(d: torch.Tensor, margin: float) -> torch.Tensor:
    """Smooth one-sided penalty.

    d >= margin        → 0
    0 <= d < margin    → -0.5/margin * (d - margin)^2
    d < 0              → d - 0.5 * margin
    """
```

Named carefully so users searching for the PyRoki function find it.

## 8. World collision

```python
@register_residual("world_collision")
class WorldCollisionResidual(Residual):
    """Penalise penetration with an external geometry set.

    Accepts any sequence of geometry primitives (Sphere / Capsule / Box /
    HalfSpace). Internally calls `collision.pairs.distance(cap, world_geom)`
    over all active (capsule, world) pairs.
    """
```

Used for ground planes, table geometry, and obstacle avoidance.

## 9. Broadphase (future)

For v1 we do the O(n_pairs) brute force over the active allowed-pair set,
which is fine for Panda-scale robots. `collision/broadphase.py` is a
placeholder for AABB pruning, spatial hashing, or BVH.

## 9.1 Asset resolution for collision meshes

Collision-mesh URIs (URDF `<collision><geometry mesh filename="..."/>`,
MJCF `<mesh file="..."/>`) resolve through the same `AssetResolver`
Protocol used by the parsers and the viewer (
[04_PARSERS.md §11](04_PARSERS.md)). `RobotCollision.from_model(model,
resolver=...)` accepts a resolver kwarg; when omitted, it reads
`model.meta["asset_resolver"]` (the resolver used at parse time). Mesh
loading uses `trimesh` (lazy import, gated by the `[geometry]` extra in
[20_PACKAGING.md](../conventions/20_PACKAGING.md)).

## 10. Migration from the pre-skeleton tree (reference)

The pre-skeleton geometry layer landed in `collision/` as follows;
listed here so historical readers can trace what moved where:

- `algorithms/geometry/primitives.py` → `collision/geometry.py` (types).
- `algorithms/geometry/distance.py` → split between
  `collision/pairs.py` (dispatch) and `collision/geometry.py`
  (`colldist_from_sdf`).
- `algorithms/geometry/distance_pairs.py` → `collision/pairs.py`,
  re-registered on the dispatch table.
- `algorithms/geometry/_utils.py` (segment-segment, etc.) →
  `collision/closest_pts.py`.
- `algorithms/geometry/robot_collision.py` → `collision/robot_collision.py`.
  The key change is the **frame-indexed** capsule storage; the original
  keyed capsules by link index, which would not survive the move to a
  frame-first data model. Sphere mode kept behind a flag.
- `costs/collision.py` → `residuals/collision.py`. The sparse analytic
  Jacobian helper that lived inside the old `tasks/ik/solver.py` as
  `_analytic_collision_jacobian` is now `SelfCollisionResidual.jacobian`.
