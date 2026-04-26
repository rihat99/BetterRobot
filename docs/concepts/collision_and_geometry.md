# Collision and Geometry

Collision is the layer where robotics libraries traditionally bloat.
Once you let collision queries leak into the kinematics path, every
function picks up a new optional argument; once you let world geometry
become a first-class construct, every loader has to know how to load
it. The right discipline is to keep collision optional, parallel to
kinematics rather than above it, and to expose it through residuals
that compose with the rest of the cost stack.

The library follows PyRoki's pattern. Geometry primitives know
nothing about robots. A pairwise SDF dispatch table computes signed
distance between any ordered pair of primitives. `RobotCollision`
sits on top of `Model` and turns it into a set of capsules at
specific frames; collision residuals query the dispatch table and
add their slot to the cost stack. A user who only cares about IK
without collision pays nothing for the machinery — `trimesh` is
imported lazily inside the collision layer, never from core
kinematics.

A note on status: the geometry primitives, the dispatch table, and
`RobotCollision` are live. The collision *residuals*
(`SelfCollisionResidual`, `WorldCollisionResidual`) are stubs,
listed in {doc}`/reference/roadmap`. The signatures are pinned —
`solve_ik`'s collision branch is wired — only the residual bodies
remain.

## Directory layout

```
src/better_robot/collision/
├── geometry.py         # primitive types + geometry utilities
├── pairs.py            # pairwise SDF dispatch table
├── closest_pts.py      # segment-segment, point-capsule, … kernels
├── robot_collision.py  # RobotCollision — the Model-level layer
└── broadphase.py       # (future) AABB pruning, spatial hashing
```

## Primitive geometry

```python
@dataclass(frozen=True)
class Sphere:
    center: torch.Tensor        # (..., 3)
    radius: torch.Tensor        # (...,)

@dataclass(frozen=True)
class Capsule:
    a: torch.Tensor             # (..., 3) endpoint A
    b: torch.Tensor             # (..., 3) endpoint B
    radius: torch.Tensor

@dataclass(frozen=True)
class Box:
    center: torch.Tensor        # (..., 3)
    half_extents: torch.Tensor  # (..., 3)
    rotation: torch.Tensor      # (..., 4) SO3 quaternion

@dataclass(frozen=True)
class HalfSpace:
    """ n · x + d >= 0 """
    normal: torch.Tensor        # (..., 3)
    offset: torch.Tensor        # (...,)

@dataclass(frozen=True)
class Plane(HalfSpace):
    """Alias for HalfSpace used for world-ground checks."""
```

Source: `src/better_robot/collision/geometry.py`.

All shapes are tensor-valued from day one: a `Sphere` with `center`
shape `(B, K, 3)` represents `B*K` spheres, not one. The same data
structure expresses "one sphere" and "one robot's worth of
self-collision spheres" without needing a second type.

## The pairwise SDF dispatch table

```python
_PAIR: dict[tuple[type, type], Callable] = {}

def register_pair(type_a, type_b):
    def _inner(fn):
        _PAIR[(type_a, type_b)] = fn
        return fn
    return _inner

def distance(a, b) -> torch.Tensor:
    """Signed distance between primitives a and b, broadcasting over
    their leading batch shapes. Positive = separated, 0 = touching,
    negative = penetration.
    """
    key = (type(a), type(b))
    fn = _PAIR.get(key) or _PAIR.get((type(b), type(a)))
    if fn is None:
        raise NotImplementedError(f"no SDF for {key}")
    return fn(a, b) if key in _PAIR else fn(b, a)
```

Source: `src/better_robot/collision/pairs.py`.

Registered pairs:

| Pair | Kernel |
|------|--------|
| (Sphere, Sphere) | `‖c_a - c_b‖ - (r_a + r_b)` |
| (Sphere, Capsule) | `point_to_segment(c_a, c_b.a, c_b.b) - (r_a + r_b)` |
| (Capsule, Capsule) | `segment_to_segment(c_a.a, c_a.b, c_b.a, c_b.b) - (r_a + r_b)` |
| (Sphere, HalfSpace) | `n · c - d - r` |
| (Capsule, HalfSpace) | `min(n · a, n · b) - d - r` |
| (Sphere, Box) | clamped closest-point on box |
| (Capsule, Box) | coarse: capsule → segment, box → segment set |

All kernels are vectorised over the leading batch shape and
differentiable. Closest-point helpers (segment-segment, point-capsule,
point-box) live in `closest_pts.py`.

## `RobotCollision`

```python
@dataclass
class RobotCollision:
    """Capsule decomposition of a robot.

    Each collision capsule is attached to a frame (not a joint) via a
    local-frame (a, b, radius) triple. At query time,
    RobotCollision.update(model, data) uses data.frame_pose_world to
    transform each capsule into the world frame.
    """
    frame_ids:       tuple[int, ...]            # per capsule
    local_a:         torch.Tensor               # (n_caps, 3)
    local_b:         torch.Tensor               # (n_caps, 3)
    radii:           torch.Tensor               # (n_caps,)
    self_pairs:      torch.Tensor               # (n_pairs, 2) capsule indices
    allowed_pairs_mask: torch.Tensor            # (n_caps, n_caps) bool

    @classmethod
    def from_model(cls, model: Model, *, mode: Literal["capsule", "sphere"] = "capsule",
                   allow_adjacent: bool = False) -> "RobotCollision": ...

    def world_capsules(self, data: Data) -> Capsule: ...
    def self_distances (self, data: Data) -> torch.Tensor: ...    # (B..., n_pairs)
    def world_distances(self, data: Data, world: Sequence[Sphere | Capsule | Box]) -> torch.Tensor: ...
```

Source: `src/better_robot/collision/robot_collision.py`.

Construction (capsule mode):

1. For every body with visual / collision geometry, choose a covering
   capsule (mesh → capsule fitting is the expensive case; URDF
   `<cylinder>` / `<box>` capsule fallbacks are trivial).
2. Record the capsule in the **frame of the body** so the local
   coordinates do not change when the robot moves.
3. Build the set of *allowed pairs* (exclude adjacent links by
   default; expose an override for "I really want elbow-shoulder
   collision detection").
4. Return a `RobotCollision` whose tensors live on the model's
   device / dtype.

Capsule fitting is offline and cached: users construct one
`RobotCollision` per robot, once per session, and reuse it across IK
calls.

## The residual side

```python
@register_residual("self_collision")
class SelfCollisionResidual(Residual):
    """(B..., n_candidate_pairs) — stable dim across iterations.

    dim = number_of_candidate_pairs. Pairs outside the safety margin
    contribute zero — but the slot exists, so the Jacobian has a
    corresponding row of zeros. Active-pair compaction is a
    kernel-internal optimisation; the public residual dim does not
    fluctuate. This keeps LM's preallocated buffers and damping
    stable across iterations.

    The residual is:
        r_p = -colldist_from_sdf(d_p, margin) * weight   if d_p < margin
        r_p = 0                                           otherwise

    Analytic Jacobian:
        Available as a sparse block with nonzero entries only in the
        kinematic chains of the two involved capsules. ResidualSpec
        declares the sparsity:
            structure="block",
            affected_joints=tuple of joints in the kinematic chains,
            dynamic_dim=True   # active subset varies; output_dim stable.
    """
```

Source: `src/better_robot/residuals/collision.py`. Currently raises
`NotImplementedError` — the geometry side is live, the residual side
is stubbed pending the analytic sparse Jacobian path. See
{doc}`/reference/roadmap`.

The crucial invariant is that the residual's `dim` is the **number
of candidate pairs**, fixed at construction, not the number of
*active* pairs at the current configuration. Pairs outside the
margin contribute zero to the residual and zero rows to the
Jacobian, but the slot exists so LM's preallocated buffers do not
have to grow or shrink between iterations. Active-pair compaction
happens internally; the public surface stays stable. This is what
makes `solve_ik` with collision residuals work without re-allocating
per iteration.

## `colldist_from_sdf` — the smooth penalty

```python
def colldist_from_sdf(d: torch.Tensor, margin: float) -> torch.Tensor:
    """Smooth one-sided penalty.

    d >= margin     →  0
    0 <= d < margin → -0.5/margin * (d - margin)^2
    d < 0           → d - 0.5 * margin
    """
```

Named carefully so users searching for the PyRoki function find it.
The smoothness across `d = margin` matters for gradient flow into the
LM step.

## World collision

```python
@register_residual("world_collision")
class WorldCollisionResidual(Residual):
    """Penalise penetration with an external geometry set.

    Accepts any sequence of geometry primitives (Sphere / Capsule /
    Box / HalfSpace). Internally calls
    collision.pairs.distance(cap, world_geom) over all active
    (capsule, world) pairs.
    """
```

Used for ground planes, table geometry, and obstacle avoidance. Same
stubbed status as `SelfCollisionResidual`.

## Asset resolution for collision meshes

Collision-mesh URIs (URDF `<collision><geometry mesh filename="..."/>`,
MJCF `<mesh file="..."/>`) resolve through the same `AssetResolver`
Protocol used by the parsers and the viewer. `RobotCollision.from_model`
accepts a `resolver=...` kwarg; when omitted, it reads
`model.meta["asset_resolver"]` (the resolver used at parse time).
Mesh loading uses `trimesh` (lazy import — pulled in only when
`RobotCollision.from_model` actually loads a mesh). See
{doc}`parsers_and_ir` for the resolver model.

## Broadphase (future)

For v1 we do the O(n_pairs) brute force over the active allowed-pair
set, which is fine for Panda-scale robots. `collision/broadphase.py`
is reserved for AABB pruning, spatial hashing, or BVH; G1-class
humanoids will need it once the residual side lands.

## Sharp edges

- **Capsules are stored in body-local frames.** They do not change
  with `q`; world placements are computed from `data.frame_pose_world`
  on each `RobotCollision.update`.
- **Allowed pairs exclude adjacent links by default.** Override with
  `RobotCollision.from_model(..., allow_adjacent=True)` if you really
  do need to detect elbow-shoulder collision.
- **`SelfCollisionResidual.dim` is stable; the *active* pairs are
  not.** LM allocates once for the candidate set and writes zeros
  into inactive rows.
- **Trimesh is the only mesh library.** Mesh loading lives in one
  module so the layer-dependency contract test can enforce it.

## Where to look next

- {doc}`residuals_and_costs` — how the (still-stubbed) collision
  residuals plug into a `CostStack`.
- {doc}`tasks` — `solve_ik(..., robot_collision=...)` is wired and
  ready for the residual to land.
- {doc}`/reference/roadmap` — what is currently stubbed and where.
