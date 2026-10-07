# Extending BetterRobot

BetterRobot has four extension points: joint models, optimization terms, viewer layers and file
parsers. None of them uses a registry of classes; you pass your object where it is needed. This page
gives the contract for each; the normative rules are in [contracts/08-extension-points](../contracts/08-extension-points.md).

## Add a joint model

A joint model satisfies the `JointModel` protocol in
`better_robot/data_model/joint_models/base.py` structurally; inheritance is not required. The
built-in joints (`fixed`, `free_flyer`, `planar`, `prismatic`, `helical`, `mimic`, `composite`, ...)
in the same folder are the reference implementations.

| Member | Contract |
|--------|----------|
| `kind`, `nq`, `nv`, `axis` | static joint description |
| `joint_transform(q_slice)` | `(B..., nq) -> (B..., 7)` pose of the child relative to the joint origin |
| `joint_motion_subspace(q_slice)` | `(B..., nq) -> (B..., 6, nv)` |
| `joint_velocity(q_slice, v_slice)` | `(B..., 6)` spatial velocity in the joint frame |
| `integrate(q, v)` | manifold retraction; keeps quaternions on the unit sphere |
| `difference(q0, q1)` | local tangent from `q0` to `q1` |
| `neutral()` | identity or default configuration |
| `random_configuration(generator, lower, upper)` | `(nq,)` valid random configuration within the limits |

Optional dynamics hooks provide the joint bias acceleration or the derivative of the motion subspace
(`joint_motion_subspace_derivative`). When a hook is absent the dynamics use zero, which is correct
only when the motion subspace is constant in `q`.

There is no joint registry. Pass the joint object to `ModelBuilder`; the built model keeps that exact
object. Add a Pinocchio parity test when Pinocchio has the same joint (see [Testing](testing.md)).

## Add a term

A term is a plain function: it reads named variables and cached values and returns a residual with
the problem's leading batch axes and a flat event axis. [Write a term](../guides/write-a-term.md) is
the full guide. Robot-aware terms go in `better_robot.terms`; `better_robot.optim` imports only torch
and must not learn about robots.

## Add a viewer layer

Subclass `better_robot.viewer.layer.Layer`. A layer holds its own `(T, ...)` arrays, creates scene
nodes in `build(scene, root)` (register handles with `_track` so visibility and disposal work), and
writes one step in `set_step(step)`. See [Viewer](../concepts/viewer.md) for the built-in layers.

## Add a parser

`better_robot.io.register_parser(suffix, fn)` is the only registration API. The suffix has no leading
dot and is matched case-insensitively. `fn` receives the source and returns an `IRModel`; `load` then
calls `build_model`. A later registration for the same suffix replaces the earlier one, and it takes
effect once your module is imported. See [Loading and parsing](../concepts/io.md) for the
intermediate model.

## Rules for every extension

- Imports point down the package graph; `tests/contract/test_layer_dependencies.py` enforces it.
  Nothing in the computational core imports `viewer`.
- Optional dependencies are imported lazily inside the function that needs them and raise
  `BackendNotAvailableError` when missing (`tests/contract/test_optional_imports.py`).
- Add an abstraction only when at least two in-tree callers need it.
- Follow the [design contracts](../contracts/00-index.md) for names, shapes and validation; the per-extension-point rules are in [08-extension-points](../contracts/08-extension-points.md).
