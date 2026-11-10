# 0016 Geometry is a value, not a field of `Model`

**Status:** Accepted (owner); amendment proposed in 0045. **Date:** 2026-11-01

## Context
`io/build_model.py::build_model` stores the whole parsed IR and the asset resolver in `Model.meta`, so the IR stays
alive in every model; `viewer/robot.py::robot_meshes` reads both. `IRGeom.params` is an untyped dict and the keys that
change a model's behaviour also sit in the untyped `meta`. ADR 0008 already makes geometry a separate brick; it has no
type yet.

## Decision
We will add a `Geometry` value (`shapes: tuple[Shape, ...]` of final typed records `Sphere`, `Box`, `Cylinder`,
`Capsule` and `Mesh`, each with frame, placement, `collision: bool` and, for a mesh, its URI), built by
`io.build_geometry(ir, model)` or `io.load_geometry(path, model)` and read by `viewer.robot_meshes(geometry, *,
root=None)`, and remove `Model.meta` with its two readers `viewer.RobotView` and `viewer.robot_layers`.

## Example
```python { .no-run }
model = br.load("panda.urdf")
geometry = br.io.load_geometry("panda.urdf", model)               # the one added line in an app
meshes = br.viewer.robot_meshes(geometry, root=Path("assets"))    # was robot_meshes(model)
```

## Consequences
`Model` is a pure kinematic and dynamic description and no longer keeps the IR alive. `viewer` stops importing `io`.
The collision brick can later read the same `Geometry`. An app that wants meshes loads geometry explicitly.

## Alternatives
A typed `Model.source: IRModel | None` until the brick lands: the viewer stays coupled to the IR and every model keeps
the IR alive. Pinocchio keeps collision and visual shapes in a `GeometryModel` beside the `Model`, and Drake keeps
them in `SceneGraph` beside the plant; both separate geometry from the kinematic description.

## Compatibility
Breaks `Model.meta`, `robot_meshes(model)`, `RobotView` and `robot_layers`. Migration: BetterRetarget's 2
`robot_meshes(model)` calls and the BetterRobot viewer examples; each app adds one `load_geometry` line, and a caller
of `robot_layers` builds its layers from `robot_meshes`, `robot_skeleton` and the layer classes.

## Enforcement
`tests/contract/test_layer_dependencies.py` (no `viewer` to `io` edge, function-level imports included);
`tests/contract/test_model_fields.py` of 0009 (no `meta`); new `tests/io/test_geometry.py` (a URDF with every shape
kind round-trips into `Geometry`).

## Open questions
None.
