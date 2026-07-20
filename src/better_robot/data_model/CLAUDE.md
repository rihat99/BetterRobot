# `data_model/` — Robot Identity and Query State

`Model` is the shallowly frozen robot description; `Data` is a mutable
workspace for one query. Do not share `Data` across concurrent evaluations or
mutate model tensors in place.

`Model` stores only `structure`, `values`, `reference_configurations`, and
`meta`. Its flat attributes are explicit properties; never duplicate part
fields on `Model` or replace them with `__getattr__` forwarding.

`ModelStructure` owns immutable topology, names and lookup/traversal methods,
coordinate permutations, and both Python and device index tables.
`ModelValues` owns differentiable placements, inertias and inertia access,
limits, gravity, and mimic tensors. Raw Torch passes consume this pair;
optional whole-pass kernels use the same seam. Build both parts directly in
`io/build_model.py`. `Model.with_values(...)` rebinds checked values while
preserving structure, and `Model.to(...)` moves each part and wraps them.

## Coordinate rules

- Joint 0 is the zero-DOF universe; a floating base is joint 1 as
  `JointFreeFlyer`.
- `nq != nv` for manifold joints. Use `Model.integrate` and
  `Model.difference`; use `idx_qs`/`idx_vs` for slices.
- `Model.q_permutation` supplies scalar gather tables for external joint
  orders and supports arbitrary leading batches.
- Mimic targets have zero-width public slices. The packed expansion and offset
  maps are the source of truth for kinematics, dynamics, and limits.

## Extending joints

Implement the `JointModel` protocol in `joint_models/`, add the public kind and
dimensions, then wire the centralized dispatch and `io/build_model.py`.
Algorithms must not duplicate parser-string dispatch. Preserve scalar-last
quaternions and linear-first tangents throughout.
