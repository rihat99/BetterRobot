# IO, Collision, Viewer, And Human Models

## Current State

The parser design is good:

- URDF, MJCF, and programmatic builders emit `IRModel`.
- `build_model` is the single IR-to-`Model` factory.
- `load` is the public dispatcher.

Collision and viewer are partially scaffolded. Human body support appears in builders and docs as future work, but it needs an explicit architecture before OpenSim, SMPL-like bodies, and anatomical joints are added.

## IO Recommendations

### Keep IR As The Parser Contract

All importers should emit `IRModel`. Do not let parser-specific objects leak into `Model`.

Add versioning:

```python
@dataclass
class IRModel:
    schema_version: int = 1
    ...
```

This makes breaking IR changes easier during early development.

### Fix Builder Kind Naming

The `ModelBuilder` example uses `kind="revolute_z"`, while `build_model` expects `kind="revolute"` with `axis=(0, 0, 1)`. Pick one.

Recommended:

- Public builder helpers use readable methods:
  - `add_revolute_z(...)`,
  - `add_revolute(...)`,
  - `add_prismatic(...)`,
  - `add_free_flyer_root(...)`.
- Raw `add_joint(kind=...)` remains available for custom kinds.

This avoids users memorizing string variants.

### Optional Dependencies

Move optional integrations out of core dependencies:

- `yourdfpy` under `better-robot[urdf]`.
- `trimesh` under `better-robot[geometry]` or `better-robot[viewer]`.
- `viser` under `better-robot[viewer]`.
- `robot_descriptions` under `better-robot[examples]`.
- `pin` under `better-robot[dev]` or `better-robot[test]`.

Core import should be cheap and not require viewer/parser packages.

### Asset Handling

Add a typed asset resolver:

```python
class AssetResolver(Protocol):
    def resolve(self, uri: str, *, base_path: Path | None = None) -> Path: ...
```

URDF/MJCF mesh paths, package URLs, and generated collision assets should all use this path.

## Collision Recommendations

Collision should have two levels:

1. Geometry primitives and signed-distance functions.
2. Robot collision model attached to frames/bodies.

Keep primitives tensor-valued:

- `Sphere`,
- `Capsule`,
- `Box`,
- `HalfSpace`,
- later convex mesh or SDF handles.

Implement stable residual dimensions first:

- one residual per candidate pair,
- zero when outside margin,
- optional active-pair compaction only inside specialized kernels.

Borrow from cuRobo:

- automatic collision sphere fitting for robot links,
- self-collision matrix/mask,
- map-reduce collision kernels,
- separate world collision from self collision,
- collision pair generation cached by model topology.

## Robot Collision Schema

Suggested:

```python
@dataclass(frozen=True)
class LinkCollision:
    body_id: int
    frame_id: int
    spheres: Sphere | None = None
    capsules: Capsule | None = None
    boxes: Box | None = None

@dataclass(frozen=True)
class RobotCollision:
    links: tuple[LinkCollision, ...]
    self_pairs: torch.Tensor
    disabled_pairs: torch.Tensor
```

The residual can transform local geometry using frame poses from `Data`.

## Viewer Recommendations

Viewer should remain topmost:

- no core module imports viewer,
- viewer imports tasks and IO only when needed,
- backend imports are lazy,
- mock backend remains the testing default.

Make viewer value-type friendly:

- accept `SE3` and raw tensor poses,
- render `Trajectory`,
- render collision geometry,
- render target frames.

Do not let viewer requirements drive core model storage.

## Human Model Roadmap

Human and biomechanical models need a separate plan because they introduce concepts not present in normal rigid robots.

Support in stages:

1. SMPL-like kinematic body builder.
2. Marker frames and marker residuals.
3. Anatomical joint constraints and coupled coordinates.
4. OpenSim `.osim` parser into IR.
5. Muscle path and actuator metadata.
6. Muscle dynamics residuals or action models.

OpenSim concepts to preserve:

- anatomical coordinates,
- custom/coupled joints,
- subject scaling,
- marker IK,
- muscle parameters,
- staged muscle quantities such as length, velocity, activation, and force.

Concepts to avoid:

- OpenSim XML as BetterRobot's canonical format,
- global system initialization lifecycle,
- non-batched mutable simulation state,
- parser-specific object graphs in core algorithms.

## Tests To Add

- `load` imports optional parser dependencies lazily.
- Builder examples run exactly as documented.
- URDF and MJCF produce equivalent simple models for shared fixtures.
- Asset resolver handles relative paths and package-like paths.
- Collision primitive SDFs match analytic cases.
- Collision residual dimension is stable.
- Viewer modules do not import optional GUI packages outside backend files.
- SMPL-like builder produces a valid `Model` and marker frames.
