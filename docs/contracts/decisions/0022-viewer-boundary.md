# 0022 The viewer stays in-tree, with the public API the studio needs

**Status:** Accepted (owner); amended by 0048. **Date:** 2026-11-01

## Context
viser stays lazy inside `viewer` (ADR 0004). `viewer/robot.py` imports `kinematics.state` inside functions, which the
rank test does not see. `viewer/_math.py::compose_poses`, `rotate` and `quat_z_to` re-implement `lie` in numpy, and
BetterHuman's studio copies them again (`viz/studio/layers.py::_quat_matrix`). Two fake viser servers diverge.
The studio reaches `Layer._visible` and `Layer._handles`, re-runs `cls.__init__` on a live layer
(`StudioLayers._renew`) and reads viser's `_tab_handles`; `Viewer.set_visible` writes `layer._visible`.

## Decision
We will keep `viewer` in-tree at the top of the package DAG with `viewer/robot.py` the only module that knows `Model`,
`KinematicState` and `Geometry`, ship the stand-in server in `better_robot.testing` (0035), keep `Layer.visible` and
`Layer.set_visible` as the documented seam, add `Layer.set_opacity` and `Viewer.replace(old, new)`, let `Viewer.add`
take an iterable, keep layers on numpy poses `(T, ..., 7)` in `xyzw`, delete the numpy quaternion helpers, and make `Timeline`
and `Viewer.play` keyword-only.

## Example
```python { .no-run }
viewer = br.Viewer()
poses = br.forward_kinematics(model, q).joint_poses.numpy()            # (T, njoints, 7)
meshes = br.viewer.RigidMeshLayer("panda", br.viewer.robot_meshes(geometry), poses)
skeleton = br.viewer.robot_skeleton(model, poses, name="panda skeleton")
viewer.add([meshes, skeleton], group="panda")          # an iterable of layers
meshes.set_opacity(0.4)
viewer.replace(skeleton, br.viewer.SkeletonLayer(...))   # no private attribute, no re-run of __init__
timeline = br.viewer.Timeline(nsteps, fps=30, loop=True)
```

## Consequences
The studio uses public names only. One quaternion conversion with `lie` lives in `robot.py`. Two new public names
(`Layer.set_opacity`, `Viewer.replace`) plus the iterable form of `add`; `Layer.visible` and `Layer.set_visible`, which
exist today, become the documented way to show and hide a layer.

## Alternatives
Layers take `[R | p]` matrices: every layer and every app call changes. The viewer as its own distribution: only worth
it once a second front end appears. Pinocchio delegates display to separate viewer packages (gepetto-viewer,
meshcat); Drake ships a Meshcat visualizer in-tree beside the plant.

## Compatibility
Adds names; makes `Timeline(nsteps, fps, loop)` and `Viewer.play(block)` keyword-only (a break for positional callers);
removes `skeleton_spec` and `spread_offsets` from `__all__`. Apps keep passing numpy poses.

## Enforcement
`tests/contract/test_layer_dependencies.py` with function-level and `TYPE_CHECKING` imports counted (only
`viewer/robot.py` imports `model`, `kinematics`); `tests/contract/test_optional_imports.py` (C-DEP-2, viser lazy);
`tests/contract/test_no_probing.py` (no `._visible` or `._handles` outside `viewer/layer.py`); ruff `SLF001` for
private-attribute reaches; `tests/viewer/` runs on the stand-in server.

## Open questions
None.
