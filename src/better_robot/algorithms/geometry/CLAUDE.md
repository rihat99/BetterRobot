# algorithms/geometry/ — Collision Geometry

Geometry primitives, distance functions, and robot collision model.

## Public API

```python
from better_robot.algorithms.geometry import (
    CollGeom, Sphere, Capsule, Box, HalfSpace, Heightmap,
    RobotCollision,
    compute_distance, colldist_from_sdf,
    DISTANCE_FUNCTIONS,
)
```

## Files

### `primitives.py` — Collision geometry types

| Class | Fields | Description |
|-------|--------|-------------|
| `CollGeom` | (base) | Abstract base for all geometry types |
| `Sphere` | `center (3,)`, `radius float` | Sphere |
| `Capsule` | `point_a (3,)`, `point_b (3,)`, `radius float` | Line-segment capsule with hemispherical caps |
| `Box` | `center (3,)`, `half_extents (3,)`, `rotation (3,3)` | Oriented box |
| `HalfSpace` | `normal (3,)`, `point (3,)` | Infinite half-space defined by a point and normal |
| `Heightmap` | `heights (H,W)`, `cell_size float`, `origin (3,)` | Terrain heightmap |

`HalfSpace.from_point_and_normal(point, normal)` — convenience constructor.

`Capsule.from_trimesh(mesh)` — fits the minimum bounding cylinder of a `trimesh.Trimesh` mesh and returns the equivalent capsule (endpoints derived from cylinder axis and half-height). Returns a zero-radius capsule for empty meshes.

### `_utils.py` — Geometric utility functions

Internal helpers for distance computation. All differentiable w.r.t. input positions.

| Function | Description |
|----------|-------------|
| `normalize_with_norm(x)` | Returns `(normalized, norm)`. Safe against zero vectors (uses eps). |
| `closest_segment_point(a, b, pt)` | Closest point on segment `[a,b]` to `pt`. |
| `closest_segment_to_segment_points(a1, b1, a2, b2)` | Returns `(c1, c2)` — closest points between two segments. |

### `distance_pairs.py` — Pairwise distance functions

All return signed distance: positive = separated, negative = penetrating.

| Function | Signature | Description |
|----------|-----------|-------------|
| `sphere_sphere(s1, s2)` | `(Sphere, Sphere) → scalar` | Center distance minus sum of radii |
| `sphere_capsule(sphere, capsule)` | `(Sphere, Capsule) → scalar` | Distance from sphere center to capsule axis, minus radii |
| `capsule_capsule(c1, c2)` | `(Capsule, Capsule) → scalar` | Closest-point distance between axes, minus radii |
| `halfspace_sphere(hs, sphere)` | `(HalfSpace, Sphere) → scalar` | Signed distance from sphere to halfspace |
| `halfspace_capsule(hs, capsule)` | `(HalfSpace, Capsule) → scalar` | Min of signed distances from both capsule endpoints to halfspace |

### `distance.py` — Distance dispatcher and SDF smoothing

**`DISTANCE_FUNCTIONS`**: dict mapping `(type_a, type_b)` → distance function. Covers all pairs in `distance_pairs.py`.

**`compute_distance(geom_a, geom_b) → Tensor`**: Dispatches to the appropriate function. Tries both `(type_a, type_b)` and `(type_b, type_a)` orderings. Raises `NotImplementedError` if no function found.

**`colldist_from_sdf(dist, activation_dist) → Tensor`**: Converts signed distance to an optimization-friendly cost.
- Returns values ≤ 0: zero when `dist ≥ activation_dist`, increasingly negative for penetration.
- Use `-result` as a positive cost.
- Based on PyRoki pattern (arxiv 2310.17274).

### `robot_collision.py` — `RobotCollision`

Fully implemented. Robot collision model supporting **capsule mode** (recommended) and **sphere mode** (legacy).

```python
@dataclass
class RobotCollision:
    _mode: str                          # 'capsule' or 'sphere'

    # Sphere mode fields (None when mode == 'capsule')
    _local_centers: Tensor | None       # (num_spheres, 3)
    _radii: Tensor | None               # (num_spheres,)
    _link_indices: Tensor | None        # (num_spheres,)

    # Capsule mode fields (None when mode == 'sphere')
    _local_points_a: Tensor | None      # (num_capsules, 3)
    _local_points_b: Tensor | None      # (num_capsules, 3)
    _capsule_radii: Tensor | None       # (num_capsules,)
    _capsule_link_indices: Tensor | None  # (num_capsules,)

    # Shared
    _active_pairs_i: tuple
    _active_pairs_j: tuple
```

**`from_urdf(urdf, model, ignore_pairs=None) → RobotCollision`** *(capsule mode)*

Auto-generates one capsule per link from URDF collision geometry.  For each link the collision meshes are merged via trimesh and wrapped with the minimum bounding cylinder (`trimesh.bounds.minimum_cylinder`).  Links with no collision geometry get a degenerate zero-radius capsule and are excluded from active pairs.

**`from_capsule_decomposition(capsule_decomposition, model) → RobotCollision`** *(capsule mode)*

Manual capsule specification. Dict format: `link_name → {'point_a': [...], 'point_b': [...], 'radius': r}` (single) or `{'points_a': [...], 'points_b': [...], 'radii': [...]}` (multiple per link).

**`from_sphere_decomposition(sphere_decomposition, model) → RobotCollision`** *(sphere mode)*

Legacy. Dict format: `link_name → {'center': [x,y,z], 'radius': r}` or `{'centers': [...], 'radii': [...]}`.

**`compute_self_collision_distance(model, cfg) → Tensor`**

Returns `(num_active_pairs,)` signed distances. Dispatches to `_get_world_capsules` or `_get_world_spheres` based on mode. Uses `compute_distance` which dispatches to `capsule_capsule()` or `sphere_sphere()`.

**`compute_world_collision_distance(model, cfg, world_geom) → Tensor`**

Returns `(num_robot_geoms * len(world_geom),)` signed distances from each robot geometry to each world primitive. Works for both modes; `compute_distance` dispatcher handles mixed pairs automatically.

## Usage Example

```python
from better_robot.algorithms.geometry import RobotCollision, HalfSpace

# Recommended: auto-generate capsules from URDF
robot_coll = RobotCollision.from_urdf(urdf, model)

# Legacy: manual sphere decomposition
robot_coll = RobotCollision.from_sphere_decomposition(
    {"panda_link3": {"center": [0, 0, 0], "radius": 0.08}}, model
)

# Self-collision distances
dists = robot_coll.compute_self_collision_distance(model, cfg)

# World collision (e.g., floor halfspace)
floor = HalfSpace.from_point_and_normal(
    point=torch.tensor([0., 0., 0.]),
    normal=torch.tensor([0., 0., 1.]),
)
world_dists = robot_coll.compute_world_collision_distance(model, cfg, [floor])
```
