# 0048 One pointer callback at a time; a layer object draws in one place; `add` takes an iterable only

**Status:** Accepted (integrator, 2026-11-10). **Date:** 2026-11-10

## Context
Record [0004](0004-viser-viewer.md) has the viewer own "viser's single scene-pointer slot"; viser now accepts several
pointer callbacks, so the slot is the viewer's choice, not viser's. Record [0022](0022-viewer-boundary.md) lets
`Viewer.add` take an iterable beside a single layer, adds `Layer.set_opacity` to the base class although only mesh
layers have surfaces, and says nothing about adding one layer object twice: a second `add` of the same object left
two scene entries driving one set of handles. Its example passes the poses of `RigidMeshLayer` positionally.

## Decision
We will keep one pointer callback at a time (`Viewer.on_pointer` raises `ArgumentError` naming `clear_pointer` when
one is held), refuse a layer object that is already in the viewer in `add` and `replace`, have `Viewer.add` take an
iterable of layers only, and keep `Layer.set_opacity` on the base class, where it raises `ArgumentError` for a layer
that draws no surface and the base checks the `opacity` value once for the constructors and `set_opacity`.

## Example
```python { .no-run }
meshes = br.viewer.RigidMeshLayer("panda", br.viewer.robot_meshes(geometry), body_poses=poses)
viewer.add([meshes])                       # one layer is a list of one
viewer.add([meshes])                       # ArgumentError: the layer is already in this viewer
viewer.on_pointer(pick)
viewer.on_pointer(other)                   # ArgumentError: call clear_pointer first
viewer.replace(meshes, br.viewer.RigidMeshLayer("panda", other_meshes, body_poses=poses))
```

## Consequences
A tool that takes over the pointer releases it explicitly, so two tools never both react to one click. A layer has
one place in the scene, so `replace` and `remove` act on exactly one entry; drawing the same data twice needs two
layer objects. `set_opacity` on a grid or a panel fails loudly instead of doing nothing. Cost: a caller with one layer
writes `add([layer])`.

## Alternatives
Several pointer callbacks, as viser allows: every tool must then filter events meant for another. `add(*layers)` or
both forms: two ways to call one method. `set_opacity` only on the mesh layers: callers that hold a `Layer` would
probe for the method.

## Compatibility
Breaks `add(layer)`, a second `on_pointer` without `clear_pointer`, and adding one layer object twice. Reversible:
each refusal can be relaxed later without breaking callers.

## Enforcement
`tests/viewer/test_time_pointer.py` (`test_on_pointer_holds_one_scene_pointer_callback`,
`test_the_same_layer_object_is_never_added_twice`); `tests/viewer/test_core.py`
(`test_set_opacity_fades_mesh_surfaces_live_and_refuses_other_layers`,
`test_mesh_layers_check_opacity_at_construction_as_set_opacity_does`); `tests/viewer/test_errors.py` (`add` refuses a single layer).

## Open questions
None.
