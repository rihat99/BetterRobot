# P3a state — viewer polish batch 1 (stopped early: context budget)

## Done (items 1-3), verified
1. Gizmo lag — `src/better_robot/viewer/tools/gizmo.py`: `PoseGizmo` owns one daemon worker thread + a
   `threading.Condition` holding a single pending pose. `handle.on_update` and `handle.on_drag_end` only overwrite
   the pending pose (`_submit`); the worker runs the callbacks with the newest pose, one at a time, off viser's
   pool. A raising callback prints its traceback and the worker keeps going. New public `wait(timeout) -> bool`
   (blocks until idle); `remove()` also stops the worker. `_notify` removed.
   - `tests/viewer/fake_server.py`: `FakeHandle.on_drag_end = on_update`.
   - `tests/viewer/test_tools.py`: old gizmo test now calls `gizmo.wait()` (callbacks are async now); new
     `test_gizmo_drops_stale_poses_and_calls_back_off_the_caller_thread` (50 rapid fires, 50 ms callback ->
     1..4 calls, last pose x=49 wins, never on the main thread).
   - `examples/viewer/03_ik_gizmo.py`: iters 30 -> 15, comment that `state` has one writer (the worker).
     `--check` prints residual 6.6e-12 m.
   - Timing (scratch `gizmo_bench.py`, real viser on 127.0.0.1:8094, 32 drag events through a
     32-thread pool like viser): old 5.80 s / 32 solves; new iters=30 0.61 s / 4 solves; new iters=15
     0.30 s / 3 solves (0.16 s of that is the emulated 5 ms drag spacing). Final error equal (2.3e-3 m).
2. Lights — `Theme.apply` calls `scene.configure_default_lights(enabled=True, cast_shadow=False)`; custom
   directional/hemisphere lights and `Theme.key_light_direction` removed (no other users in BR/BH).
   fake_server needed no change (generic `configure_*`).
3. Theme — `Theme.brand` -> `Theme.brand_color` (navy #1f3a68); new `Theme.control_width="medium"`;
   `configure_theme(dark_mode, brand_color, control_layout="collapsible", control_width, show_logo=False,
   show_share_button=False)`. Colours re-tuned for light bg: joint navy, link/robot/point muted greys,
   force/torque/target desaturated, lighter grid. DARK gets a lighter navy joint. `LIGHT` stays default.
   No BH code reads `brand` or `key_light_direction` (BH uses `theme.joint/link/force`, `DARK`, `Theme`).
- Also: removed unused `import torch` in `examples/viewer/01_robot_pose.py` (ruff F401, gate).

## Gates
- `tests/viewer`: 24 passed (was 23 + 1 new). ruff (viewer, tests/viewer, examples): all checks passed.
- Full suite: 1143 passed, 8 failed — all `tests/docs/test_snippets.py[docs/contracts/0*.md]`, pages being
  written by the docs phase concurrently; not viewer-related.

## Screenshots
`vis/p3a/panda_theme_lights_step000.png`, `..._step030.png` (CPU SwiftShader, headless.py copy on 8094).
Camera is far away (robot tiny); shows the default lights, no shadow streak, lighter grid. Retake with an
explicit close camera (`client.camera.position/look_at`) in batch 2. Robot mesh reads very pale under
viser default lights; consider robot colour ~0.70 grey.

## Remaining (exact, in order)
4. Rename frame->step: `Layer.set_frame`->`set_step`, `num_frames`->`num_steps`, Timeline (`frame`, `_frame`,
   `on_frame`, GUI label "Step"; check Timeline has no existing `step` member first), `Viewer.show(step)`,
   `record`/`render_frames` params (render/__init__.py). ~140 hits in src/tests/examples (grep
   `set_frame|num_frames|on_frame|_frame\b|\.frame\b|render_frames`). BH call sites: viz/human_layers.py:132,151;
   viz/studio/state.py:219,225; viz/studio/frames_editor.py:73,98,106,112,138; viz/studio/layers.py:83,158,215,
   231,236,251,268,272; viz/studio/app.py:75 (re-grep after rename).
5. Timeline only when num_steps > 1 (lazy folder; single-step `play()` just serves).
6. Sidebar: one "Layers" folder, checkbox per layer (`group/name`), synced with `set_visible`;
   `Viewer.rebuild(layer)`; `Layer.visible` property; drop `GroundLayer` isinstance special case.
7. Joint shapes by kind in `SkeletonLayer` + `skeleton_spec(model)`; keep `tau=` colouring.
8. `RobotView.update` isinstance chain -> per-layer update or typed attributes.

## P3a-2 (items 4-6 done; tests/viewer 26 passed)
API old -> new:
- `Layer.set_frame(frame)` -> `set_step(step)`; `Layer.num_frames` / ctor kwarg `num_frames=` -> `nsteps`;
  `geometry(frame)` -> `geometry(step)` (all built-in layers, `_Shifted`).
- `Timeline(num_frames=)` -> `Timeline(nsteps=)`; `.frame`/`._frame` -> `.step`/`._step`; `on_frame` -> `on_step`;
  `extend(nsteps)`; GUI slider label "Frame" -> "Step". `Viewer.show(step)`.
- `render.render_frame(layers, step, ...)`; `render_frames(layers, steps=None, ...)`; `record(..., steps=...)`
  (was `frames=`); `frame_camera(layers, steps=...)`. Function names unchanged (they make image frames).
- Timeline folder is built lazily on the first layer with `nsteps > 1` (`Viewer._timeline_folder` starts None);
  single-step `play()` is a no-op that just serves.
- Sidebar: one "Layers" folder (lazy, on first listed layer), one checkbox per layer, label `name` in the
  default group "scene", else `group/name`; synced by `Viewer.set_visible`. Per-group GUI folders are gone
  (`_Group.folder` removed; `_Entry.visible` removed -> `Layer.visible` read-only property).
- New: `Viewer.rebuild(layer)`, `Layer.visible`, `Layer.listed = True` (GroundLayer False),
  `Layer.apply_theme(theme)` hook (no-op; GroundLayer resolves `plane` from `theme.up`) replacing the
  isinstance(GroundLayer) case in `Viewer._add`.
BH call sites to update (not changed): viz/human_layers.py:151 (num_frames); viz/studio/layers.py:83,158,231;
viz/studio/state.py:219,225; viz/studio/frames_editor.py:73,98,106,112,138 (set_frame). Check also the `frames=`
hits at human_layers.py:149, persistence.py:162,167, layers.py:192, frames_editor.py:147 (time or robot frames?).
Docs still on old API (out of scope, make tests/docs/test_snippets fail): docs/concepts/viewer.md:27,33,132;
docs/getting-started/visualize.md:81 (`record(..., frames=range)` -> `steps=`); docs/contracts/01-naming.md:67;
docs/contributing/extending.md:43.
Remaining: items 7-8 (not P3a-2).
