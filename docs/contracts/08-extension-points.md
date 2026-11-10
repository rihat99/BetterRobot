# 08 Extension points

The library grows by adding *to* fixed seams, not by editing their consumers. There are five seams: terms
(plain functions), joint kinds and loaders (subclasses of an abstract base; parser functions), the public surface
(names without `_`), the optimizer's linear sink (below), and collision `distance_fn` obstacles (below). A new seam needs an ADR. The how-to lives in [extending](../contributing/extending.md).

## C-EXT-1 Terms are plain functions (MUST)

A *term* is a plain function (`src/better_robot/optim/problem.py::TermFn`) whose argument names are the
`Problem` entries it reads and which returns residuals `(..., r)`. Robot-specific terms are *factories* in
`terms`: a function that takes the model and options and returns a `TermFn` closing over them. Terms carry
no state and are never subclasses; `src/better_robot/optim/problem.py::Term` is the record `Problem` keeps per
registered term, not a base class for users. See ADR [0003](decisions/0003-terms-plain-functions.md).

```python { .no-run }
# good  (src/better_robot/terms/robot.py::frame_pose)
def frame_pose(model, *, frame, target, state="state", q="q") -> TermFn:   # factory
    def term(state, target): ...                                           # reads "state", "target"
    return term
# bad
class PoseTerm:
    def residual(self, variables): ...
```

Enforcement: `tests/contract/test_extension_points.py` (every extension point is a base class or a plain function,
no registry); `[review]`.

## C-EXT-2 Joint kinds subclass the abstract `JointModel`; loaders are functions (MUST)

A joint kind is a final frozen dataclass under `src/better_robot/model/joints/base.py::JointModel` (record
[0013](decisions/0013-joint-kinds.md)): it declares `nq`, `nv`, `manifold` and `axis` and implements `placement`,
`motion_subspace`, `neutral` and `random`; the abstract methods make a missing member fail where the kind is
defined, not at a distant dispatch. Every class in `model.joints` and `model.manifolds` is abstract or `@final`:
kinds are data, not a hierarchy. There is no joint registry: the kind is passed to `ModelBuilder.add_joint(kind=)`
and the built model keeps that object (record [0025](decisions/0025-extension-model.md)). Consumers that need a
kind's class (`src/better_robot/kinematics/_fk.py`, the viewer's markers) test the final classes; any other kind takes the
generic path through `placement`. A new file format is a function returning `IRModel`, passed to `io.build_model`; there is no
parser registry (`tests/contract/test_extension_points.py`).

```python { .no-run }
# good: a kind subclasses the base and is passed where it is used
@final
@dataclass(frozen=True)
class JointRevolute(JointModel):
    axis: tuple[float, float, float]
    nq: ClassVar[int] = 1
    nv: ClassVar[int] = 1
    manifold: ClassVar[Manifold] = EUCLIDEAN
builder.add_joint("elbow", kind=JointRevolute(axis=(0.0, 1.0, 0.0)), parent="upper_arm", child="forearm")
# bad: a kind string looked up in a registry
register_joint("revolute", JointRevolute)
```

Enforcement: `tests/contract/test_abstract_or_final.py` (every class in `model.joints` and `model.manifolds` is
abstract or final, every kind a frozen `JointModel`); `tests/model/test_joints.py` (every kind passes
`check_joint_model`); `[review]`.

## C-EXT-3 Public means no leading underscore and a place in `__all__` (MUST)

A name is public when it is in its package `__all__` and has no `_` prefix; everything else may change
without notice. Private helpers across packages are imported from their package, not re-exported.

Enforcement: `tests/contract/test_public_surface.py`, `tests/contract/test_submodule_public_imports.py`;
`tests/contract/test_ruff_ratchets.py` (ruff's `SLF001`: no `obj._private` access across modules).

## C-EXT-4 This major keeps no deprecated alias (MUST)

A public name that changes is renamed or removed outright, and the changelog's entry names the replacement; there is
no alias that warns. The first deprecation of a later release brings a table of deprecated names with their removal
version and a test that fails when that version is reached (principle 8.9); it changes this rule. Internal names are
renamed outright.

Enforcement: `tests/contract/test_versioning.py::test_src_emits_no_deprecation_warning`.

## C-EXT-5 New solvers, viewer layers and packages follow one recipe each (MUST)

- Solver: a subclass of `src/better_robot/optim/solvers.py::Solver`: `init(problem, x) -> State` and
  `step(problem, state) -> State`, both pure (a new `State` per step, no writes to the problem), listed in
  `__all__`.
- Viewer layer: subclass `src/better_robot/viewer/layer.py::Layer`; implement `build(scene, root)` and `set_step(step)`;
  override `build_gui`, `apply_theme`, `geometry` and `dispose` only when needed; set `listed = False` to keep it out
  of the sidebar.
- Package: add its row to `ALLOWED` in `tests/contract/test_layer_dependencies.py` and the package to the diagram
  in `07`; this needs an ADR (`00-index`, "When to write an ADR").

Enforcement: `tests/contract/test_layer_dependencies.py`; `[review]`.

## C-EXT-6 Pre-1.0 versioning: minor releases may break (MUST)

Until 1.0 the version is `0.MINOR.PATCH`. A minor release may change or remove public API, and every such change is
in the changelog with its migration; a patch release changes no public signature. A public signature changes only
together with the tracked snapshot `tests/contract/public_api.json`, so no change reaches a release unnoticed. The
public surface and the horizon of each kind of artifact are on [Public surface](public-surface.md).

Enforcement: `tests/contract/test_versioning.py::test_version_is_pre_1_0`; `tests/contract/test_api_snapshot.py`.

## Optimization: normal equations are assembled through a sink (SHOULD)

`Problem.linearize` never builds a matrix itself: it hands Jacobian products to a sink and asks it for the
normal equations. The three sinks in `src/better_robot/optim/linear.py` (`DenseSink`, `BandSink`, `ArrowSink`) share two
methods, `add(columns_a, columns_b, values)` to accumulate a block of `JᵀJ` and `finish(g)` to return a
`Normal`, `BandedNormal` or `ArrowNormal` with the common `solve` / `restrict` / `predicted_decrease`
interface. A future block-sparse structure (for example a contact graph) is a new sink plus a new normal class
behind this seam; solvers and terms do not change.

## Collision: obstacles plug in through `distance_fn` (MUST)

`terms.sphere_clearance(spheres, distance_fn, *, margin)` is the seam for any environment shape: a new obstacle
type is a function `distance_fn(centers: (B..., S, 3), radii: (S,)) -> (B..., S)` or `(B..., S, K)` (signed
distance, positive outside, differentiable in `centers`), usually a closure over a `collision.sphere_*`
primitive. The term and the solvers never change for a new shape. Robot geometry beyond spheres and closest
points follow record [0016](decisions/0016-geometry-out-of-model.md).

```python { .no-run }
# good: a new shape is a new function
def to_table(centers, radii):
    return collision.sphere_box(centers, radii, box_center=center, box_rotation=rotation, half_extents=size)
terms.sphere_clearance(spheres, to_table, margin=0.02)
# bad: a new term class or a solver branch per obstacle type
```

Enforcement: `[review]`; `tests/collision/test_collision.py` runs a closure through `check_jacobian`.
