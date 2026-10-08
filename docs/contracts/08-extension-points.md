# 08 Extension points

The library grows by adding *to* fixed seams, not by editing their consumers. There are five seams: terms
(plain functions), joint types and loaders (registries behind protocols), the public surface (names
without `_`), the optimizer's linear sink (below), and collision `distance_fn` obstacles (below). A new seam needs an ADR. The how-to lives in [extending](../contributing/extending.md).

## C-EXT-1 Terms are plain functions (MUST)

A *term* is a plain function (`optim.TermFn`, `optim/problem.py:28`) whose argument names are the
`Problem` entries it reads and which returns residuals `(..., r)`. Robot-specific terms are *factories* in
`terms`: a function that takes the model and options and returns a `TermFn` closing over them. Terms carry
no state and are never subclasses; `optim.Term` (`optim/problem.py:46`) is the record `Problem` keeps per
registered term, not a base class for users. See ADR [0003](decisions/0003-terms-plain-functions.md).

```python { .no-run }
# good  (terms/robot.py:62)
def frame_pose(model, *, frame, target, state="state", q="q") -> TermFn:   # factory
    def term(state, target): ...                                           # reads "state", "target"
    return term
# bad
class PoseTerm:
    def residual(self, variables): ...
```

Enforcement: `tests/contract/test_protocols.py` (every public term factory returns a callable); `[review]`.

## C-EXT-2 Joint types and loaders plug in through a protocol (MUST)

A new joint type implements the `JointModel` protocol in `data_model/joint_models/base.py` structurally
(no base class, no registry: the model object is passed where it is needed); a new file format implements
the `io` parser signature and registers its suffix with `io.register_parser(suffix, fn)`
(`io/__init__.py:42`). Consumers (`kinematics`, `dynamics`, `viewer`) SHOULD dispatch through the protocol
rather than `isinstance` on concrete joint classes `[planned: L-1]`: today `kinematics/state.py:84`,
`io/build_model.py:387`, `io/parsers/programmatic.py:394` and `joint_models/composite.py:32` still do. A
joint-type registry does not exist and needs an ADR before one is added.

```python { .no-run }
# good  (illustrative)
register_parser(".sdf", parse_sdf)              # io: suffix -> parser
joint = JointPlanar(...)                        # any object satisfying JointModel
# bad
if isinstance(joint, JointRevolute): ... elif isinstance(joint, JointPrismatic): ...
```

Enforcement: `tests/contract/test_protocols.py` (parser registry, `JointModel` protocol); `[review]`.

## C-EXT-3 Public means no leading underscore and a place in `__all__` (MUST)

A name is public when it is in its package `__all__` and has no `_` prefix; everything else may change
without notice. Private helpers across packages are imported from their package, not re-exported.

Enforcement: `tests/contract/test_public_api.py`, `tests/contract/test_submodule_public_imports.py`;
`[ruff:SLF001]` [planned: L-7] (no `obj._private` access across modules).

## C-EXT-4 Deprecate for one minor release before removing (SHOULD)

A public symbol that changes keeps the old name for one minor release, emitting
`DeprecationWarning` with the replacement, and is listed in `docs/reference/changelog.md`. Internal names are
renamed outright.

Enforcement: `[review]`; changelog entry checked in review.

## C-EXT-5 New solvers, viewer layers and packages follow one recipe each (MUST)

- Solver: a class satisfying `optim.Solver` (`optim/solvers.py:30`): `init(problem, x) -> State` and
  `step(problem, state) -> State`, both pure (a new `State` per step, no writes to the problem), registered in
  `__all__`.
- Viewer layer: subclass `viewer.Layer` (`viewer/layer.py:25`); implement `build(scene, root)` and `set_step(step)`;
  override `build_gui`, `apply_theme`, `geometry` and `dispose` only when needed; set `listed = False` to keep it out
  of the sidebar.
- Package: add it to `LAYER_RANK` in `tests/contract/test_layer_dependencies.py:23` at its rank in the `07` DAG and
  to the diagram in `07`; this needs an ADR (`00-index`, "When to write an ADR").

Enforcement: `tests/contract/test_protocols.py`; `tests/contract/test_layer_dependencies.py`; `[review]`.

## C-EXT-6 Pre-1.0 versioning: minor releases may break (MUST)

Until 1.0 the version is `0.MINOR.PATCH`. A patch release changes no public signature or result. A minor release
may change or remove public API after the deprecation period of `C-EXT-4`, and every such change is listed under
a "Breaking" heading in `docs/reference/changelog.md` with the migration.

Enforcement: `[review]`; changelog entry checked in review.

## Optimization: normal equations are assembled through a sink (SHOULD)

`Problem.linearize` never builds a matrix itself: it hands Jacobian products to a sink and asks it for the
normal equations. The three sinks in `optim/linear.py` (`DenseSink`, `BandSink`, `ArrowSink`) share two
methods, `add(columns_a, columns_b, values)` to accumulate a block of `JᵀJ` and `finish(g)` to return a
`Normal`, `BandedNormal` or `ArrowNormal` with the common `solve` / `restrict` / `predicted_decrease`
interface. A future block-sparse structure (for example a contact graph) is a new sink plus a new normal class
behind this seam; solvers and terms do not change.

## Collision: obstacles plug in through `distance_fn` (MUST)

`terms.sphere_clearance(spheres, distance_fn, *, margin)` is the seam for any environment shape: a new obstacle
type is a function `distance_fn(centers: (B..., S, 3), radii: (S,)) -> (B..., S)` or `(B..., S, K)` (signed
distance, positive outside, differentiable in `centers`), usually a closure over a `collision.sphere_*`
primitive. The term and the solvers never change for a new shape. Robot geometry beyond spheres and closest
points follow [ADR 0008](decisions/0008-geometry-brick.md).

```python { .no-run }
# good: a new shape is a new function
def to_table(centers, radii):
    return collision.sphere_box(centers, radii, box_center=center, box_rotation=rotation, half_extents=size)
terms.sphere_clearance(spheres, to_table, margin=0.02)
# bad: a new term class or a solver branch per obstacle type
```

Enforcement: `[review]`; `tests/collision/test_collision.py` runs a closure through `check_jacobian`.
