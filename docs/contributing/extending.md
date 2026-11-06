# Extending BetterRobot

BetterRobot has four extension points: joint kinds, optimization terms, viewer layers and file
parsers. None of them uses a registry of classes; you pass your object where it is needed. This page
gives the contract for each; the normative rules are in [contracts/08-extension-points](../contracts/08-extension-points.md).

## Add a joint kind

A joint kind subclasses the abstract `JointModel` in `src/better_robot/model/joints/base.py` as a final frozen
dataclass (records 0013 and 0025); the built-in kinds in the same folder (`revolute.py`, `prismatic.py`,
`planar.py`, ...) are the reference implementations, one file per kind.

| Member | Contract |
|--------|----------|
| `nq`, `nv` | class variables: the widths of the joint's `q` and `v` slices |
| `manifold` | class variable: the `Manifold` of `better_robot.model.manifolds` whose `integrate`/`difference` step the coordinates |
| `axis` | the joint axis in the joint frame as a tuple of floats (a dataclass field), or the class variable `None` |
| `placement(q)` | `(B..., nq) -> (B..., 7)` pose of the child in the joint frame |
| `motion_subspace(q)` | `(B..., nq) -> (B..., 6, nv)`; the child's twist in its own frame is `S(q) @ v` |
| `neutral()` | `(nq,)` configuration with the identity placement |
| `random(generator, *, lower, upper)` | `(nq,)` configuration inside the limits |

A new manifold is a `Manifold(name=, nq=, nv=, integrate=, difference=)` whose maps are the group exponential and
logarithm, so that `difference(q0, integrate(q0, v)) == v` and the derivative of `placement(integrate(q, t v))` at
`t = 0` is `motion_subspace(q) @ v`. `better_robot.testing.check_joint_model(joint, dtype=, generator=)` checks
both and the shapes; run it on every new kind.

There is no joint registry. Pass the kind to `ModelBuilder.add_joint(kind=...)`; the built model keeps that exact
object, forward kinematics evaluates it through `placement`, and the viewer draws it as a sphere. Add a Pinocchio
parity test when Pinocchio has the same joint (see [Testing](testing.md)).

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

A parser is a function from a path to an `IRModel`; pass its result to `better_robot.io.build_model`, as `load`
does for URDF and MJCF. There is no registry (record 0025): `load` knows those two formats by suffix, and your
format lives in your function. See [Loading and parsing](../concepts/io.md) for the intermediate model.

## Rules for every extension

- Imports point down the package graph; `tests/contract/test_layer_dependencies.py` enforces it.
  Nothing in the computational core imports `viewer`.
- Optional dependencies are imported lazily inside the function that needs them and raise
  `OptionalDependencyError` when missing (`tests/contract/test_optional_imports.py`).
- Add an abstraction only when at least two in-tree callers need it.
- Follow the [design contracts](../contracts/00-index.md) for names, shapes and validation; the per-extension-point rules are in [08-extension-points](../contracts/08-extension-points.md).

## Where next

- [Testing](testing.md): the parity, contract and docs tests every extension must pass.
- [Write a term](../guides/write-a-term.md): a term with an analytic Jacobian, step by step.
