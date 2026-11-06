# 0014 Lie and spatial algebra are tensor functions only

**Status:** Accepted (owner). **Date:** 2026-11-01

## Context
`lie/types.py::SE3`, `SO3`, the alias `Pose = SE3`, and `spatial.Motion`, `Force`, `Inertia`, `Symmetric3` sit beside
tensor functions; nothing outside their tests constructs `SE3` or `SO3`. `dynamics/_world.py` re-implements
`cross_motion`, `cross_force`, `adjoint_inverse` and the 6x6 inertia because the classes do not fit a batched kernel.
`spatial` imports `lie.types` inside methods to avoid a cycle it created, and `Force.cross_motion` only raises and
carries an F821. Users: BetterHuman `Inertia.from_mesh(...).data`, BetterRetarget `Force` and `Motion` (4 sites),
BetterVideoReconstruction `Force` (1 site).

## Decision
We will provide Lie and spatial algebra as functions on tensors only (`spatial.cross_motion`, `cross_force`,
`transform_motion`, `transform_force`, `transform_inertia`, `inertia_matrix`, `rotational_inertia`, `inertia_of_sphere`, `inertia_of_box`,
`inertia_of_capsule`, `inertia_of_ellipsoid`, `inertia_of_mesh`) and remove the value classes and `Pose`.

## Example
```python { .no-run }
inertia = spatial.inertia_of_mesh(vertices, faces, density=1000.0)   # (..., 10), differentiable; was Inertia.from_mesh(...).data
wrench_b = spatial.transform_force(transform_ba, wrench_a)           # (..., 6), linear first
```

## Consequences
One implementation of each operation, shared by `dynamics` and the callers; the `spatial` to `lie.types` edge
disappears. The `(..., 10)` inertia layout and differentiability that BetterHuman's mass pipeline needs stay. The cost
is operator syntax: no `T1 * T2`, only `se3.compose(T1, T2)`.

## Alternatives
Keep `Inertia` alone for the mesh integrator: the class survives for one function. Classes as the only form, in the
style of jaxlie: every batched kernel wraps and unwraps, the largest rewrite. jaxlie and Pinocchio use value types
with operators, both built for one query at a time; PyPose's `LieTensor` is a tensor subclass and pulls the library
into its type system. Every hot path here is already tensor code.

## Compatibility
Breaks `SE3`, `SO3`, `Pose`, `Motion`, `Force`, `Inertia`, `Symmetric3` and `spatial/ops.py`. Migration: BetterHuman 1
source line and 6 test lines (`Inertia.from_mesh(...).data` becomes `inertia_of_mesh(...)`), BetterRetarget 4 sites,
BetterVideoReconstruction 1. Adding a class back later is compatible.

## Enforcement
`tests/contract/test_se3_vocabulary.py` (C-NAME-5); `tests/contract/test_layer_dependencies.py` (`spatial` imports
only `lie`); the dynamics Pinocchio parity in `tests/parity/pinocchio/` exercises the shared functions; the public
surface is pinned by `tests/contract/test_public_surface.py`.

## Open questions
None.
