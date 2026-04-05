# algorithms/geometry/ — Collision Geometry

Geometry primitives and robot collision model. Distance functions are stubs — not yet implemented.

## Public API

```python
from better_robot.algorithms.geometry import (
    CollGeom, Sphere, Capsule, Box, HalfSpace, Heightmap,
    RobotCollision,
)
```

## Files

### `primitives.py` — Collision geometry types

| Class | Fields | Description |
|-------|--------|-------------|
| `CollGeom` | (base) | Abstract base for all geometry types |
| `Sphere` | `center (3,)`, `radius float` | Sphere |
| `Capsule` | `p0 (3,)`, `p1 (3,)`, `radius float` | Line-segment capsule |
| `Box` | `center (3,)`, `half_extents (3,)`, `rotation (3,3)` | Oriented box |
| `HalfSpace` | `normal (3,)`, `offset float` | Infinite half-space `n·x + d ≥ 0` |
| `Heightmap` | `heights (H,W)`, `cell_size float`, `origin (3,)` | Terrain heightmap |

`HalfSpace.from_point_and_normal(point, normal)` — convenience constructor.

### `robot_collision.py` — `RobotCollision`

Stub for per-link collision geometry. Currently raises `NotImplementedError`.

Planned: load per-link geometry from URDF, produce batched distance queries.

## Planned Extensions

Once `distance.py` and `distance_pairs.py` are implemented:
```python
# Dispatcher: (GeomType, GeomType) → distance function
dist = compute_distance(sphere_a, sphere_b)

# Pairs implemented:
# sphere-sphere, sphere-capsule, capsule-capsule
# halfspace-sphere, halfspace-capsule
```
