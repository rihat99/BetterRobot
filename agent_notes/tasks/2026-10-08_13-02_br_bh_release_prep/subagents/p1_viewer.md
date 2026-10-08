# P1 viewer — adversarial architecture review

Scope: BR `src/better_robot/viewer` (2275 lines), BH `src/better_human/viz` (1991 lines), BVR `tools/viewer`
(2871 lines, read-only). viser 1.0.26 (checked with `uv run python -c "import viser; print(viser.__version__)"`).
Budget note: core files read in part (signatures, docstrings, key bodies); web sources cited from docs, not
re-fetched in this session; claims about rerun/meshcat should be re-checked before they drive a decision.

## Current state (facts, with paths)

- Retained-mode, clip-preloaded design. `Layer` (`viewer/layer.py`) owns numpy `(T, ...)` arrays by reference,
  creates viser handles in `build(scene, root)`, writes slice k in `set_step(k)`. Optional: `build_gui`,
  `apply_theme`, `world_points(step)`. Flat: layers live under `/<group>/<name>`; `Viewer.add(layer, group,
  offset=)` (`viewer/viewer.py:79`). No parenting of a layer to another layer's frame.
- One `Timeline` (`viewer/timeline.py:11`): integer `step`, one `fps`, `loop`, `speed`; playback thread paces by
  wall clock and drops steps (`_run`). Layers shorter than the timeline are clamped (`min(k, nsteps-1)`,
  `viewer.py:38`). `extend()` grows it. No per-layer time base, no fractional time, no interpolation.
- `Viewer.show` is the single push path inside `server.atomic()`; `RLock` guards add/remove vs playback
  (`viewer.py:65`).
- Built-in layers: Ground, Arrows, Frames, Points, Skeleton, Mesh, RigidMesh, Spheres, SkinnedMesh, Camera
  (`layers/*.py`), PlotPanel (uPlot in the sidebar, `layers/plot.py:47`). `CameraLayer` (frustum + image,
  image may be a lazy `step -> image` callable) lives in `layers/skinned.py:92`, not its own file.
  Only `PlotPanel` implements `build_gui`; opacity etc. are constructor-only.
- Robot adapter: `robot.py` is the only `Model`-aware file (`RobotView.update(q)`, `robot_layers`,
  `_torque_colors`). BH mirrors it with `viz/human_layers.py` (`HumanView`, `forces_layer`, ...).
- Tools: `tools/gizmo.py` (`PoseGizmo` over `add_transform_controls`, `on_change`, blocking `wait`);
  `tools/picking.py` pure numpy (`CameraFrustum`, `ray_hit`, `vertices_in_sphere`, `vertices_in_rect`,
  `project`). No pointer routing: tools do not own pointer events.
- Capture: `Viewer.save` writes a `.viser` file via `get_scene_serializer` (`viewer.py:145`);
  `Viewer.record` drives headless Chromium (`capture.py`, `$BETTER_ROBOT_CHROME`) + ffmpeg for webm,
  `fit_camera` from `world_points`.
- BH HumanStudio (`viz/studio/app.py`) builds its whole UI on the escape hatch: `viewer.server.gui.*`
  (tab group, dropdowns, sliders, buttons, markdown; `app.py:95-131`), `viewer.server.scene.on_pointer_event`
  (`app.py:69`) and manual `remove_pointer_callback`; its own `_locked` threading wrapper; `_nearest_on_ray`
  (`app.py:349`) is a picking primitive that BR `tools/picking.py` lacks.
- viser allows ONE scene pointer callback at a time: a single slot `self._scene_pointer_cb`
  (`viser/_scene_api.py:166`, dispatched at `:2617`).
- Docs: `docs/concepts/viewer.md` has no limits/roadmap section; `docs/reference/roadmap.md` has no viewer item
  (grep for viewer/viser/rerun/meshcat: 0 hits).

BVR `tools/viewer` features (facts): example/clip dropdown that disposes and rebuilds a `ClipScene`
(`app.py:204`, `scene.py:978`); multiple body sources with an active source and capability-gated controls
(`frame_sources`, `force_sources`, `translucency_sources`, `app.py:247-267`); video pane in the GUI with a view
dropdown, synced per frame (`app.py:136-150, 384-396`); onboard camera that drives every client's camera along the
reconstructed trajectory at 60 Hz from a fractional, wall-clock playhead, decoupled from slow scene posing
(`app.py:344-366`); time-coloured camera frustums with labels and a Catmull-Rom camera path
(`scene.py:146-166`); per-frame human/scene point clouds, scene surface points vs mesh, scene normals
(`scene.py:210, 449, 556`); skinned body via `add_mesh_skinned` with top-4 weights (`scene.py:225`,
`helpers.py:110`); force section: per-joint vs per-frame aggregation, root force/torque, measured sensor forces vs
estimated forces in force-plate (force-at-COP) format (`app.py:29-43`, `scene.py:815-967`); opacity and
point-size/frustum-size sliders (`app.py:181-188, 269-289`); unlit GLB and white background (`helpers.py:126-138`).

## How references evolved (with sources)

- Newton (`references/sim/newton/newton/_src/viewer/`): one backend-neutral `ViewerBase` (`viewer.py`) with
  immediate-mode `begin_frame/end_frame` and `log_mesh/log_instances/log_lines/log_points/log_array/log_scalar/
  log_contacts/log_gizmo` (`viewer.py:442-1205`), and backends `viewer_gl.py`, `viewer_viser.py`,
  `viewer_rerun.py`, `viewer_usd.py`, `viewer_file.py`, `viewer_null.py`. Lesson: the stable seam is a small
  set of geometry primitives, not the toolkit; a null and a file backend make tests and offline recording cheap.
  It also ships `picking.py` and contact logging as first-class.
- mjlab (`references/design/mjlab.md:403`): lists "viewer coupling (viser/mjviser hard deps)" as a known
  weakness. Same risk as our `server.gui` escape hatch spreading into BH.
- Drake (`references/design/drake.md:59`, https://drake.mit.edu/doxygen_cxx/classdrake_1_1geometry_1_1_meshcat.html):
  Meshcat is a path-addressed scene tree (`SetObject`/`SetTransform`), plus separate visualizer systems
  (`MeshcatVisualizer`, `ContactVisualizer`) that publish into it, and `StartRecording/PublishRecording` for
  animations. Lesson: domain adapters (contacts, illustration vs proximity geometry) sit on top of a dumb scene
  API, and transforms are parented so a mounted camera or tool follows its link.
- rerun (https://rerun.io/docs/concepts/timelines, /spaces-and-transforms, /blueprint): many timelines per
  recording (frame, sim time, log time), data logged at native rate and queried "latest at" a time; entity-path
  hierarchy with `Transform3D` and `Pinhole` so 3D data projects into 2D image views; layout ("blueprint")
  separate from data; `.rrd` files are a data log, not a video. Lesson: multi-rate data and 2D/3D linked views
  needed a time model richer than one integer step.
- viser (https://viser.studio): scene nodes are parented by path name, `add_3d_gui_container`, `.viser`
  serializer for static playback, one pointer callback slot (verified above). Lesson: our layer flatness is our
  choice, not viser's limit.
- BVR viewer (above): written outside BR; it re-implemented a playback clock, onboard camera, video pane,
  multi-source comparison and force overlays because BR had no bricks for them.

## Findings (ranked)

- F1 (high) One integer time base. Evidence: `Timeline` int step + single fps; layers clamped by index. Consequence:
  mocap 100 Hz + video 30 Hz + force plates 1000 Hz cannot be shown together without the user resampling
  everything to one grid; no smooth camera follow or slow-motion interpolation (BVR wrote its own fractional
  playhead, `app.py:344`). Retrofitting time later touches every layer's `set_step` signature.
- F2 (high) No pointer/tool routing. Evidence: viser single `_scene_pointer_cb`; BH studio hand-registers and
  removes callbacks (`studio/app.py:69`). Consequence: picking, marquee, measurement and gizmo-click tools cannot
  coexist; every studio reinvents a mode switch and its locking.
- F3 (high) GUI has no BR-level brick; the escape hatch is the plugin API. Evidence: only `PlotPanel` has
  `build_gui`; HumanStudio uses `server.gui` for everything. Consequence: BH (and BVR, if it adopts BR) couple to
  viser's API directly; swapping or wrapping the backend, or testing studios against `fake_server.py`, gets
  harder each release (mjlab names this exact debt).
- F4 (medium) Preload-everything memory model. Evidence: layers hold full `(T, ...)` arrays; only `CameraLayer`
  accepts a lazy callable. Consequence: long sequences (10 min SMPL-X at 100 Hz as raw vertices ~7.5 GB float32)
  and dataset browsers need a lazy `step -> array` source on every layer; today only skinning avoids it.
- F5 (medium) Flat layers, no parenting. Evidence: `/<group>/<name>` + `offset`. Consequence: head camera, tool
  frame, sensor on a link, labels riding frustums (BVR `scene.py:151`) all need precomputed world poses; multi-model
  scenes are side-by-side offsets only, not placed in a shared world.
- F6 (medium) Missing overlay bricks BVR needed: video/image pane synced to the timeline, 2D reprojection of
  keypoints onto video, camera path + time-coloured frustums, lines/trail layer, label/text layer, contact/COP/
  wrench layer (force-at-COP, friction cone), measured-vs-estimated comparison, per-layer opacity/size controls,
  client-camera follow. Consequence: each project re-implements them (BVR already did, ~2.9k lines).
- F7 (medium) No "source/comparison" concept. Evidence: BVR active body source + capability gating. Consequence:
  ground truth vs estimate vs retarget comparisons (BH, BVR core use case) are app code, not a brick.
- F8 (low) Recording is video/.viser only; no data log export (npz/rrd) or reload into Python for analysis.
- F9 (low) Hygiene: `CameraLayer` in `layers/skinned.py`; `_nearest_on_ray` belongs in BR `tools/picking.py`;
  no viewer entry in `docs/reference/roadmap.md`; concepts page has no "limits" section.

## Roadmap

| brick | why (user) | prerequisite | when |
|---|---|---|---|
| Time model: layers expose `times (T,)` (seconds) or `rate`; timeline maps playhead time -> per-layer index (latest-at, optional lerp) | multi-rate mocap/video/plates (BVR, biomech) | F1 seam: `set_step` gets an index computed by the viewer from time; default `times = arange(T)/fps` keeps today's behaviour | next (seam now) |
| `ToolManager`: one pointer callback, routes to the active tool; tools = `activate/deactivate/on_pointer` | studios with pick + marquee + measure | F2; tools own no viser callback directly | next |
| Measurement tool (distance, angle, point-to-plane) on `ray_hit` | users checking fits, retargets | ToolManager | later |
| Panel brick: `Viewer.add_panel(name, build)` returning a folder/tab; layers' `build_gui` reused | BH studio, BVR | F3; keeps `server` escape hatch | next |
| Lazy layer sources: `data` may be array or `step -> array` for all layers (pattern from `CameraLayer`) | long clips, dataset browsing | none (CameraLayer already does it) | next |
| Parenting: `Viewer.add(layer, parent=(layer, frame_index))` -> viser path nesting | mounted cameras, tool frames, labels | layers accept a root under another layer's frame node | later |
| `VideoPanel` + `ReprojectionLayer` (2D keypoints over frames, via `CameraFrustum.project`) | BVR video reconstruction | time model; `project` exists | next |
| `LinesLayer`/`TrailLayer`, `LabelsLayer` (viser `add_line_segments`, `add_spline_catmull_rom`, `add_label`) | trajectories, COM path, camera path | none | next |
| `WrenchLayer`: force-at-COP, contact normals, friction cones, per-joint vs per-frame aggregation | dynamics/contact users, BVR plates | ArrowsLayer; aggregation lives in adapters (`robot.py`, BH `human_layers.py`) | next |
| `FollowCamera`: drive all clients' cameras from a pose track at render rate | onboard view, demos | time model (fractional) | later |
| Source comparison: `Viewer.add_source(name, layers)` + active-source / overlay / side-by-side modes | GT vs estimate (BH, BVR) | groups already toggle; add mutual-exclusion and capability tags | later |
| Clip browser: dropdown that disposes and rebuilds groups (`remove_group` exists) | dataset review (BVR `ClipScene`) | lazy sources | later |
| Data export: dump the timeline's layer arrays to npz; optional rerun `.rrd` writer as an extra | offline analysis, sharing | time model; lazy rerun import at a boundary | maybe |
| Backend seam (newton-style): layers call a small primitive API (mesh, instances, lines, points, image) instead of raw viser | headless/null backend for tests, rerun/USD later | layers stop using `scene.*` directly; `fake_server.py` already approximates a null backend | maybe |
| Own three.js client | only if viser blocks (2D panels, linked views) | backend seam | maybe |

## Prep changes recommended now (each ≤ 1 day, non-breaking)

1. Add an optional `times: np.ndarray | None` (seconds, `(T,)`) attribute to `Layer` (default `None` = uniform at
   timeline fps) and route the step-to-index mapping through one private `Viewer._index(layer, step)` that today
   returns `min(step, layer.nsteps - 1)`. No behaviour change; the multi-rate brick then edits one function.
2. Wrap pointer events in BR now: `Viewer.on_pointer(callback)` / `Viewer.clear_pointer()` that own the single
   viser slot, and switch HumanStudio to it. Smallest step toward a ToolManager; removes the
   `remove_pointer_callback` dance from BH.
3. Move `_nearest_on_ray` (BH `studio/app.py:349`) into BR `tools/picking.py` as `nearest_on_ray`; move
   `CameraLayer` to `layers/camera.py` (re-export unchanged).
4. Generalize the lazy-callable input of `CameraLayer` into a documented `Layer` convention (array or
   `step -> array`) in `docs/concepts/viewer.md`; no code change needed yet.
5. Add a "Limits and next bricks" section to `docs/concepts/viewer.md` and viewer rows to
   `docs/reference/roadmap.md` (F1-F7), so users of BVR/BH see what is planned before writing app code.
6. Add a `LinesLayer` (polyline/trail with optional per-vertex colour) and a `LabelsLayer`: both are thin viser
   wrappers, each under ~80 lines, and unblock camera paths, COM trails and annotated frames.

## Open questions for the owner

1. Is BVR expected to move onto the BR viewer (then F6/F7 bricks are `next`), or stay separate (then `later`)?
2. Time model: seconds-based `times` per layer (rerun-like) or keep integer steps and require resampling? The
   seam in prep change 1 keeps both open; the choice decides the public signature.
3. Should viser stay the only backend (accept coupling, expose `server` freely) or do we want a narrow
   primitive seam (newton-style) so a null/file/rerun backend is possible? This decides whether new layers may
   call `scene.*` directly.
4. Should GUI panels be a BR brick, or is `viewer.server.gui` the sanctioned plugin API for studios?
5. Is a rerun `.rrd` export wanted (new optional dependency), or is npz enough?
