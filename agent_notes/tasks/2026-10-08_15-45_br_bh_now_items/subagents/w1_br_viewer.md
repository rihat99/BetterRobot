# w1 slice E `br_viewer` — report

All five items are done. Not committed.

## What changed (file: change)
- `src/better_robot/viewer/layer.py`: class attribute `Layer.times: np.ndarray | None = None` (sample times
  `(nsteps,)` in seconds, set by `Viewer.add`).
- `src/better_robot/viewer/viewer.py`:
  - `Viewer.add(layer, group, *, offset=None, times=None)`. `times` is validated: shape `(layer.nsteps,)` and
    non-decreasing, otherwise `ValueError`. It is stored on `layer.times`.
  - private `_index(layer, step) -> int`: `min(step, nsteps-1)` when `times is None`; otherwise the latest sample
    with `times[i] <= step / fps` (searchsorted with a 1e-9 tolerance so a sample stamped exactly on a step is not
    lost to rounding), clipped to `[0, nsteps-1]`.
  - private `_span(layer)`: the number of timeline steps the layer needs (`floor(times[-1]*fps)+1` when timed), fed
    to `timeline.extend`.
  - `_push` and `_points` (record camera fit) now go through `_index`. Layers still receive a plain int sample index.
  - `on_pointer(kind, callback)` (`kind` in `"click"`, `"rect-select"`, otherwise `ValueError`) clears any held
    registration, then registers through `server.scene.on_pointer_event`. `clear_pointer()` calls
    `remove_pointer_callback()` only when the viewer holds the slot (`self._pointer`).
  - Class docstring: `set_step(min(k, nsteps-1))` became `set_step(_index(layer, k))`.
- `src/better_robot/viewer/tools/picking.py`: public `nearest_on_ray(points (N,3), origin (3,), direction (3,),
  radius) -> int | None`. It is a verbatim port of BH `studio/app.py:_nearest_on_ray`, plus a docstring.
  Exported from `viewer.tools`.
- `src/better_robot/viewer/layers/camera.py` (new): `CameraLayer` moved there unchanged.
  `layers/skinned.py` re-exports it (`__all__`) and drops the `Callable` import it no longer uses.
  `layers/__init__.py` and `viewer/__init__.py` now import it from `.camera`.
- `tools/`: no BR tool used the scene pointer (the gizmo uses transform controls), so there was no call site to
  move. The only users are in BH (listed below).
- `timeline.py`: docstring "Frame counter" became "Step counter". `_math.as_frames` docstring: "one frame" became
  "one step". There were no GUI labels saying "frame".
- `src/better_robot/viewer/CLAUDE.md`: one bullet on the time model and on pointer-slot ownership.

## Tests
- `tests/viewer/test_time_pointer.py` (new, 5 tests):
  - a 10 Hz layer and a 30 Hz layer on one 30 fps timeline (exact index map, timeline length 28, `show(4)` pushes
    1 and 4);
  - `times` default and validation;
  - `on_pointer` replace, `clear_pointer`, and the bad-`kind` error, using the fake server;
  - `nearest_on_ray` (nearest in front, radius, behind origin, miss);
  - the `CameraLayer` module plus its re-export.
- `tests/viewer/fake_server.py`: `FakeScene.on_pointer_event` and `remove_pointer_callback`, modelled on viser's
  single slot. It asserts if a second registration arrives while one is held.
- Live smoke test on a real viser server at 127.0.0.1:8093 (stopped afterwards): two timed layers give
  `timeline.nsteps == 28`; `on_pointer` click, then rect-select, then `clear_pointer` twice raises no warnings and
  leaves `_scene_pointer_cb is None`.

## Docs touched
- `docs/concepts/viewer.md`:
  - the timeline section uses "step" wording and documents `times` (with a `.no-run` snippet);
  - the tools section covers `nearest_on_ray` and the `on_pointer` / `clear_pointer` rule;
  - new "Lazy inputs" section: `step -> array` is the convention, but only `CameraLayer(image=...)` accepts it.
    The other layers need eager arrays for `nsteps` and `world_points`, so supporting it is not a 3-line change.
  - new "Limits" section: one integer timeline, viser only, `viewer.server.gui` is the sanctioned studio API, and
    `record` needs Chromium.
  - "positions that frame `record`'s camera" became "positions that `record`'s camera fits".
- `docs/contracts/01-naming.md`: only the C-NAME-4 time-axis sentence changed (time axis is the integer `step`;
  `times` is optional seconds).
- `docs/contracts/decisions/0004-viser-viewer.md`: dated "Note 2026-10-08": viser stays the only backend,
  `server.gui` is sanctioned, the viewer owns the pointer slot, and an npz export is planned.

## Gate output tails
- Viewer tests: `35 passed, 1 skipped`.
- Full suite: `2 failed, 1620 passed, 11 skipped, 7 deselected`. Both failures are in other slices that are still
  in progress, not the viewer:
  - `tests/io/test_joint_dynamics.py::test_mjcf_dynamics_reach_model_values` (slice A);
  - `tests/lie/test_euler_matrix_interop.py::test_euler_quaternion_roundtrip_batched_away_from_gimbal_lock`
    (slice C).
- `ruff check src tests`: 86 findings in total (baseline 82; the extra ones come from other slices'
  concurrent edits). The viewer and tests/viewer have 11 findings, all pre-existing FBT and BLE findings in
  untouched signatures (line numbers shifted). The new files have none.
- `mkdocs build --strict`: built, INFO only. The INFO says ADRs 0007 and 0008 (slices C and D) are not in nav.

## BH call sites for wave 2 (not edited)
- `BetterHuman/src/better_human/viz/studio/app.py:69,299,306,308,309`: direct
  `viewer.server.scene.on_pointer_event` and `remove_pointer_callback` calls. Switch them to
  `viewer.on_pointer(kind, cb)` and `viewer.clear_pointer()`. They still work today, but mixing them with
  `viewer.on_pointer` would double-register.
- `app.py:255,265,384`: replace the local `_nearest_on_ray` with `better_robot.viewer.tools.nearest_on_ray`
  (same signature and behaviour).
- No BH import of `CameraLayer` or `layers.skinned` was found. Nothing in BH breaks.

## Left / open
- Lazy callables are only on `CameraLayer.image`, as documented. Extending them to other layers needs an explicit
  `nsteps` argument and is out of scope.
- `times` is set via `Viewer.add`, not via layer constructors, because the subclasses do not forward kwargs to
  `Layer.__init__`.
