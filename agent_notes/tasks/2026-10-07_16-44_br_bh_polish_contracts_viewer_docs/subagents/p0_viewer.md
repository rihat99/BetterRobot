# P0 — Viewer audit and viser 1.0.26 capabilities (mars_aux, 2026-10-07 17:15)

V/ = `src/better_robot/viewer/`; viser = `.venv/lib/python3.11/site-packages/viser/`.
**Verdict:** every wish is doable with viser 1.0.26. Gizmo lag has a measured cause. Headless PNG capture works with
bare headless Chromium (no Playwright python needed) but is only fast on the GPU.

## 1. Architecture as is
- `Viewer` (V/viewer.py:53–218) builds the Timeline GUI unconditionally (`add_folder("Timeline")` ~:86);
  `Timeline.build_gui` (V/timeline.py:113) adds Play, Frame slider (min=max=0 without a trajectory), Speed, Loop.
- `_add` (viewer.py:108): one folder per group + one checkbox per layer (clutter); isinstance special-case for
  `GroundLayer`. Group offset via `node.position`; `record` wraps layers in `_Shifted` (viewer.py:34).
- `Theme.apply` (V/theme.py:39) hard-codes `control_layout="collapsible"`, `show_logo=False`; disables viser default
  lights, adds own key light (directional + shadows) and hemisphere fill. Brand colour (0.16,0.45,0.85) = medium blue.
- `RobotView` (V/robot.py:28) builds RigidMeshLayer, SkeletonLayer, FramesLayer, SpheresLayer; `update()` (:76) is
  an isinstance chain. `robot_layers()` (:104) thin wrapper.
- Rasterizer: V/render/__init__.py (138) + rasterizer.py (190). Exists only for it: `Geometry`, `Layer.geometry`
  (V/layer.py:13,58), every `geometry()` in layers/mesh.py, frames.py, skinned.py, `_Shifted`, `Viewer.record`
  (viewer.py:177), tests/viewer/test_render.py (87), examples/viewer/04_record.py. MP4 needs ffmpeg on PATH.

## 2. viser 1.0.26
- Theme: `configure_theme(titlebar_content, control_layout floating|collapsible|fixed, control_width
  small|medium|large, dark_mode, show_logo, show_share_button, brand_color)` (_gui_api.py:449); `add_folder` (525),
  `add_modal` (571), `add_tab_group` (602).
- Primitives (_scene_api.py): `add_cylinder` (2186), `add_box` (2041), `add_icosphere` (2121), `add_frame` (1117),
  `add_batched_axes` (1174), `add_grid` (1248), `add_mesh_simple` (1576), `add_mesh_trimesh` (1654),
  `add_batched_meshes_simple` (1702), `add_batched_meshes_trimesh` (1822), `add_batched_glb` (1894),
  `add_line_segments` (717), splines. No capsule/arrow primitive.
- Lights: directional, ambient, hemisphere, point, rectarea, spot (301–490); `configure_environment_map` (545);
  `configure_default_lights(enabled, cast_shadow)` (632). `set_up_direction` (210).
- Gizmo: `add_transform_controls` (2372) with `on_update`, `on_drag_start`, `on_drag_end`.
- `get_render(height, width, transport_format)` (_viser.py:395) is per client → needs a connected browser. No
  offscreen renderer in viser.
- Unused today: tab groups, modals, titlebar, control_width, environment map, get_render, add_cylinder/add_box
  (viewer draws everything via `add_batched_meshes_simple` with own unit meshes).

## 3. Headless capture (measured)
- No Python Playwright anywhere under ~ (brief was wrong). Not needed: launch
  `~/.cache/ms-playwright/chromium_headless_shell-1243/chrome-headless-shell-linux64/chrome-headless-shell
  --no-sandbox --window-size=960,720 <url>` via subprocess, wait for `server.get_clients()`, set
  `client.camera.position/look_at`, call `client.get_render(480, 640)`. Set the camera explicitly (default far away).
- Binary needs `libasound.so.2` (host lacks it; scratch copy via LD_LIBRARY_PATH worked — see memory
  headless-chrome-libasound). Browser connected after 1.75 s; first render 8.9 s after launch.

| Backend (Panda 640×480) | ms/frame | 60 frames |
|---|---|---|
| SwiftShader CPU, current theme + shadows | 1577 | 95 s |
| SwiftShader CPU, viser default lights, no shadows | 478 | ~29 s |
| GPU `--use-angle=gl-egl --enable-gpu --ignore-gpu-blocklist` | 86–88 | ~5 s |
| GPU Vulkan | 87 | ~5 s |

Images correct; current key light throws a long shadow streak on the ground. Scripts: scratchpad `headless.py`,
`headless2.py`.

## 4. Gizmo IK lag (root cause measured)
- viser runs each sync callback on `ThreadPoolExecutor(max_workers=32)` (_viser.py:761); every drag message submits
  a new `follow()`. One `follow()` (float64, iters=30): solve 37–58 ms, `robot.update` 1.4 ms, `show` 2 ms.
- 32 concurrent solves: 20.3 s total vs ~1.3 s sequential. `torch.get_num_threads()` = 192 → ~15× oversubscription.
  Callbacks also race on `state["q"]` (examples/viewer/03_ik_gizmo.py `follow`).
- Fix: latest-wins worker inside `PoseGizmo` (V/tools/gizmo.py:59): one daemon thread + Condition holding a single
  pending pose; drag overwrites, worker solves the newest. Optionally cap torch threads; `on_drag_end` final refine.

## 5. Joint shapes
- Today: batched icospheres (fixed joints 0.4× radius) + cylinder bones (V/layers/skeleton.py:50–81,
  robot.py:109–127); only positions reach the layer — orientation, kind and axis dropped.
- Model has `model.joint_models[j].kind` and `.axis` (data_model/joint_models/base.py:60–63). Kinds: revolute,
  prismatic, spherical, free_flyer, fixed, planar, helical, translation, composite.
- Mapping (one `add_batched_meshes_simple` per kind, rotation = joint rotation × `quat_z_to(axis)`): revolute/helical
  → cylinder along axis; prismatic → thin box along axis; spherical → sphere; free_flyer → small triad
  (`add_batched_axes`); fixed → nothing; planar/translation → small cube. Changes SkeletonLayer API (full poses);
  BH studio uses SkeletonLayer.

## 6. BH studio reuse
Reuses Viewer, Layer, PointsLayer, SkeletonLayer, Theme, DARK, PoseGizmo, CameraFrustum, ray_hit, vertices_in_rect,
vertices_in_sphere; re-implements PartsLayer, SegmentationLayer; uses DARK in app.py and layers.py.

## Ranked changes
1. S gizmo latest-wins worker (tools/gizmo.py, examples/viewer/03_ik_gizmo.py, tests/viewer/test_tools.py).
2. S viser default lights; remove custom lights + `key_light_direction` (theme.py, tests fake_server/test_core).
3. S light theme everywhere + navy brand (theme.py, BH app.py, layers.py).
4. S Timeline only when num_frames > 1 (viewer.py, timeline.py, tests).
5. M joint shapes by kind (layers/skeleton.py, robot.py, _math.py, tests/viewer/test_robot.py, BH users).
6. M sidebar redesign (titlebar, control_width, Scene/Tools tab group, fewer per-layer checkboxes).
7. M/L drop rasterizer; record via headless Chromium + get_render (removes render/, Geometry, Layer.geometry,
   geometry() methods, _Shifted, test_render.py; reworks 04_record.py + docs).
8. S/M decouple GroundLayer isinstance in `Viewer._add` and the chain in `RobotView.update`.

## Open questions
- Exact navy hex. Chrome binary + libasound as system dep for recording OK? GPU for recording (86 ms/frame) vs CPU
  (478–1577)? MP4 may keep depending on ffmpeg?
