# Scan: the three existing viewers (BR / BH / BVR)

Paths: BR = `dev/BetterRobot/src/better_robot/viewer/`, BH = `dev/BetterHuman/src/better_human/viz/`,
BVR = `BetterVideoReconstruction/tools/viewer/` (all under `/home/rikhat.akizhanov/better/`).
All three are viser. Line numbers below are from the current files.

## 1. Inventory

### BR viewer — 1,872 lines in 18 files (+76 lines docs/concepts/viewer.md)
- `Visualizer` (visualizer.py:33, 213 l): facade. Holds model, lazily builds `ViserBackend` + `Scene.default`
  (visualizer.py:~70-80), `update(q|Data)`, `add_trajectory`, `show_frame`, `add_ik_targets`, `add_ik_result`.
- `Scene` (scene.py:26, 169 l): dict of named "modes"; `default()` runs FK at q_neutral and attaches best
  primary mode + GridOverlay + FrameAxesOverlay (scene.py:56-88); `update(data)` loops visible modes
  (scene.py:150); `update_from_q` runs FK with frames then update (scene.py:157).
- `RenderMode` Protocol (render_modes/base.py:34): `is_available(model,data)`, `attach(ctx,model,data)`,
  `update(data)`, `set_visible`, `detach`. `RenderContext` (base.py:20) carries backend, theme, batch_index.
  Implementations: `URDFMeshMode` (urdf_mesh.py:99, visual geoms from `io.IRModel`, 255 l), `SkeletonMode`
  (skeleton.py:48, spheres+cylinders, 218 l). Overlays use the same protocol: `GridOverlay`, `FrameAxesOverlay`,
  `ForceVectorsOverlay` (175 l), `TargetsOverlay` (185 l, transform controls for IK targets).
- `RendererBackend` Protocol (renderers/base.py:18): add_mesh/sphere/cylinder/capsule/frame, remove,
  set_transform/visible/color/scale, add_gui_button/slider/checkbox. One real impl `ViserBackend`
  (viser_backend.py:18, 375 l) + `MockBackend` for tests (renderers/testing.py:28, 158 l).
- `TrajectoryPlayer` (trajectory_player.py:22, 87 l), `Theme` (themes.py:12, 7 fields), `PrimitiveHandle`
  (primitive.py:15), `build_joint_panel` (panels.py:16), quaternion helpers (helpers.py).
- Start: `v = Visualizer(model, port=8080); v.update(q); v.show()` (visualizer.py:40-45 docstring).
- viser features used: add_mesh_simple x3, add_mesh_trimesh, add_icosphere, add_frame, add_grid,
  add_transform_controls, gui button/checkbox/slider/folder. No skinned mesh, no batched meshes, no
  timeline GUI, no frustums, no images, no plots, no arrows primitive (arrow = cylinder+cone meshes, viser_backend.py:121).

### BH viz — 3,263 lines in 18 files
- `SkinnedMeshMode` (skinned_mesh.py:16, 87 l): a BR `RenderMode` plug-in (imports `better_robot.viewer.RenderContext`,
  skinned_mesh.py:8). CPU LBS each update via `body.vertices_from_data(data)` then full `backend.add_mesh`
  re-upload (skinned_mesh.py:44-48, 68-72). Used by BH examples 01/03/04.
- Studio (`viz/studio/`, 3,171 l) = a single-pose body editor, not a playback viewer:
  - `HumanStudio` (app.py:43, 575 l): owns `viser.ViserServer` (app.py:60), layers dict (app.py:79-82),
    `GizmoController` (gizmo.py, 246 l), tabs; `python -m better_human.viz.studio` → `studio.run()` (__main__.py:35).
  - `StudioSession` (state.py:86, 250 l): model state + observer: `set_q/set_shape/set_segmentation/
    set_frame_set` each recompute then `_notify(kind)` with kind in {body, pose, shape, segmentation, frames}
    (state.py:119,143,180,210,222).
  - `Layer` Protocol (layers/base.py:46): `build(server, session)`, `update(session, kind)`, `set_visible`.
    Implementations: `MeshLayer` (mesh.py:77, 294 l), `SkeletonLayer` (258 l), `FramesLayer` (171 l),
    `PartsLayer` (132 l, exploded segmentation). Colors are module constants (base.py:16-35).
  - Editing tools: `SegmentEditor` (298 l, paint vertex parts, uses `better_robot.spatial.Inertia` for mass),
    `FramesEditor` (283 l, place contact frames on vertices), `picking.py` (144 l, CPU ray/rect/sphere picking),
    `posing.py` (151 l, joint handles + local rotation, pure torch), `persistence.py` (165 l, save/load pose,
    segmentation, frame sets to disk).
- viser features: add_mesh_skinned (mesh.py:221, client-side LBS, top-4 weights mesh.py:13), add_mesh_simple x4,
  add_point_cloud x2, add_icosphere, add_cylinder, add_transform_controls (gizmo), gui tab_group/markdown/checkbox.

### BVR viewer — 2,871 lines (app 551, scene 990, loading 1114, helpers 197, __init__ 19) + view_gt.py 141
- `Viewer` (app.py:58): GUI + clip switching (`load_clip` app.py:401) + playback thread (`run` app.py:468)
  + 60 Hz camera thread (app.py:357, 472). `threading.RLock` guards clip swap vs playback (app.py:112).
- `ClipScene` (scene.py:97): builds every node for one clip under one root frame `/clip` so `dispose`
  removes all by one call (scene.py:122-127); `apply_frame(f)` (scene.py:602) pushes frame f inside
  `server.atomic()` (scene.py:606). Layer visibility is a plain shared `dict[str,bool]` read live
  (scene.py:99-101, `set_static_layer` scene.py:516).
- `ClipData` (loading.py:772): one dataclass with ~25 domain fields (scene/human point clouds, extrinsics,
  fov, gravity, skinned bodies, forces, sensors, plates, skeletons, video frames, fps, views). Loaders for
  BVR npz formats, SMPL-X/MHR skinning prep, BEDLAM (loading.py:49-1114).
- Start: `view_3d(root, port=8082)` (app.py:524) → `Viewer(server, clips,...).run()`.
- viser features (most of the three): add_mesh_skinned (scene.py:225, bones updated per frame scene.py:~642),
  add_point_cloud, add_mesh_trimesh (bones as unit cylinders), add_icosphere, add_arrows x6 (batched arrows),
  add_camera_frustum x2, add_spline_catmull_rom (camera path), add_glb (unlit mesh, helpers.py:138),
  add_label, add_grid, gui add_image (video panel, app.py:148), dropdowns, sliders, folders, buttons,
  `client.atomic()` camera updates (app.py:338).
- `scripts/peter_gt/view_gt.py` (141 l): a separate standalone script, does not reuse tools/viewer; static
  point clouds + frustums drawn as line segments (view_gt.py:53-59), set/cloud dropdowns (view_gt.py:70-74).

## 2. Feature matrix (Y = present, P = partial, - = absent)

| feature | BR | BH | BVR |
|---|---|---|---|
| robot URDF visual meshes | Y urdf_mesh.py:171 | - | - |
| collision geometry | - (visual_geoms only, urdf_mesh.py:102) | - | - |
| skeleton (spheres+bones) | Y skeleton.py:48 | Y layers/skeleton.py | Y scene.py:233-271 |
| skinned human mesh | P via BH SkinnedMeshMode (CPU LBS) | Y client skinning mesh.py:221 | Y client skinning scene.py:225 |
| playback play/pause/scrub/speed | P blocking `play(fps)`, no GUI (trajectory_player.py:67) | - | Y play toggle, frame slider, fps slider, wall-clock clock (app.py:473-495) |
| layers + visibility toggles | P `set_mode_visible` (scene.py:128), no GUI | Y Layer protocol + checkboxes | Y dict + checkboxes |
| forces / contact arrows | P ForceVectorsOverlay (not wired, test-only) | - | Y per-joint/per-frame, root, sensor, plate (app.py:37-43) |
| joint torques | - | - | Y sphere size by torque (app.py:29-32) |
| frames / axes | Y FrameAxesOverlay | Y FramesLayer (contact frames) | Y contact frames + connectors |
| keypoints | - | - | P (human point clouds, contact frames) |
| camera / video overlay | - | - | Y frustums, camera path, video image, onboard camera mode |
| point clouds | - | P (edit mode only) | Y scene + per-frame human clouds |
| ground / grid | Y GridOverlay | - | Y add_grid, gravity |
| themes | P Theme dataclass, 7 fields | P color constants | P constants, viser default lighting (app.py:~544) |
| interactive gizmos | Y IK targets (targets.py) | Y joint posing gizmo | - |
| editing tools | - | Y segment, frames, shape sliders, persistence | - |
| multiple characters | - (one model per Visualizer) | - (one body) | Y people per clip, several body sources |
| record / export video | - | - | - |
| offline / headless render | - (MockBackend records calls only) | - | - |
| plots / time-series panels | - | - | - |
| batch display | P batch_index only (render_modes/base.py:20) | - | - |

## 3. Duplication, uniqueness, cleanest design

Duplicated in all three: skeleton as icosphere-per-joint + cylinder-per-bone with a shortest-arc z→dir
quaternion (BR skeleton.py; BH layers/base.py:82 `bone_transform`; BVR helpers.py:91 `_wxyz_shortest_arc_z`),
xyzw→wxyz conversion (BR helpers.py:11; BH layers/base.py:76; BVR inline), top-4 skin-weight renormalization
for add_mesh_skinned (BH mesh.py:13 `_top4_normalized_weights`; BVR helpers.py:110 `_top4_skin_weights`),
bind-pose bone extraction, color constants, visibility checkbox wiring.
Duplicated BR/BH: two "layer" protocols that differ only in what they receive (`update(data)` vs
`update(session, kind)`). BH SkinnedMeshMode duplicates MeshLayer skinning but slower (CPU).

Unique: BR — URDF mesh loading from IRModel, IK target gizmos, backend/mode protocols, mock backend tests.
BH — editing (segmentation painting, frame placement, posing gizmo), picking, persistence, change-kind
observer. BVR — playback clock, clip switching with lock, camera/video/frustum stack, force/torque display,
multi-person, point clouds, `/clip` root-frame teardown, `atomic()` batching.

Cleanest: BH's `Layer` (layers/base.py:46) + `StudioSession` observer is the cleanest abstraction: a layer
owns one node group, builds once, and decides per change kind whether to rewrite vertices in place or
rebuild (base.py:1-6 docstring, mesh.py:49-68). BVR has the best *mechanics* (root frame for teardown,
atomic frame push, wall-clock playback, client skinning) but no abstraction: `ClipScene` is a 990-line god
object with per-feature node dicts and `if` chains in `apply_frame`. BR has the most interface but least
function: the backend Protocol abstracts viser away for exactly one backend, and the mode Protocol is
fine but bound to `(Model, Data)`.

## 4. Time handling and performance

- BR: no time model. `TrajectoryPlayer.show_frame(k)` takes `traj.q[0,k]` → `forward_kinematics` per frame
  → per-node `set_transform` (trajectory_player.py:55-65, scene.py:157). FK runs on every frame show, batch 0
  only, `play` blocks the caller with `time.sleep` (trajectory_player.py:67-83). No `atomic()`. Force arrows
  are re-added whenever length changes (force_vectors.py:~135).
- BH: no time; state pushed per user action. MeshLayer with skinning sends only bone poses on "pose"
  (mesh.py:55-59) — the fast path; fallback rewrites `handle.vertices` (mesh.py:63-66). SkinnedMeshMode
  (BR plug-in) re-creates the mesh node per update (skinned_mesh.py:72) — the slow path.
- BVR: everything preloaded into `ClipData` as numpy (all frames) at clip load; per frame `apply_frame`
  writes bone wxyz/position for each skinned body (scene.py:~640-643), each skeleton sphere/bone position
  individually (scene.py:~693-708), toggles per-frame human point-cloud nodes by visibility (pre-created
  per frame, scene.py:~616-622), and removes + re-adds arrows (scene.py:~723). All inside one
  `server.atomic()`. Playback thread advances by wall clock, skipping frames if posing is slow (app.py:483-493).
  Limits: one websocket message per sphere/bone/bone-of-skinned-mesh per frame (O(people x joints));
  per-frame point-cloud nodes multiply memory on the client; loading is eager (whole clip in RAM + GPU skinning prep).

## 5. Coupling

- BR: modes take `(Model, Data)` everywhere (render_modes/base.py:50-66); `Scene.update_from_q` calls
  `forward_kinematics` (scene.py:157-162); URDFMeshMode reaches into `model` for the stored `io.IRModel`
  (urdf_mesh.py:102,143); Visualizer knows `IKResult` and IK targets (visualizer.py:164-205) → viewer
  depends on tasks. The backend is generic, the modes are robot-specific.
- BH: layers take `StudioSession` (layers/base.py:51-54), which wraps a `ParametricBody`; layers read
  `session.vertices`, joint positions, segmentation. Body-specific; not reusable for robots.
- BVR: `ClipScene` reads `ClipData` field by field; loaders know BVR npz layouts, BEDLAM, SMPL-X/MHR
  (loading.py:271-959). Imports `tools.geometry.surface` (loading.py:19). Fully domain-bound.
- To separate: (a) a generic core that knows only nodes, transforms, timeline and layers, fed by arrays
  `(T, ...)`; (b) domain adapters that turn Model+trajectory → per-frame body poses (BR), ParametricBody →
  skin weights + bone poses (BH), ClipData → layers (BVR). FK should run once per trajectory (batched over T),
  not per displayed frame.

## 6. Problems

- Dead / unwired in BR: `build_joint_panel` (panels.py:16) has no caller; `register_mode` and
  `get_renderer` (render_modes/__init__.py:13, renderers/__init__.py:13) have no caller outside their files;
  `ForceVectorsOverlay` is used only in a test, not reachable from `Visualizer`; `seek_frame` is an alias
  of `show_frame` (trajectory_player.py:85).
- Over-abstraction in BR: `RendererBackend` Protocol + `MockBackend` (250 l) for one backend; the Protocol
  lacks methods ViserBackend actually uses (`add_mesh_trimesh`, `add_arrow`, `add_grid`,
  `add_transform_control`), so callers probe with `hasattr` (urdf_mesh.py:169).
- Under-abstraction in BVR: 990-line `ClipScene`, 1114-line loader, feature flags as string keys in a dict
  (app.py:79-87).
- Defaults: BR dark gray background, gray links, blue joints (themes.py:16-19), no lighting config; BVR
  notes viser default lighting and a 24em sidebar hack (app.py:~540-544). No shared palette.
- Missing vs owner's list: BR has no GUI timeline, no torques, forces not wired, no human mesh, no
  multi-character; BH has no time at all; BVR lacks robots (URDF meshes), gizmos and reuse. None has
  recording/export, headless rendering, or time-series plots.
- view_gt.py duplicates frustum drawing that BVR scene.py does with add_camera_frustum.

## 7. Minimal core concepts for a unified viewer (from what the three actually do)

1. `Viewer`: owns the server, one root node per scene (BVR `/clip` trick), GUI sidebar, `atomic()` push.
2. `Timeline`: T frames + fps; play/pause, scrub slider, speed, wall-clock clock (BVR app.py:473-495);
   static (T=1) case covers BH studio and BR single pose.
3. `Layer`: owns a node group; `build(viewer)`, `set_frame(k)`, `set_visible`, optional GUI controls;
   gets its data preloaded as `(T, ...)` arrays at build time (BVR model) — no Model/Data in the protocol.
4. Generic layers in BR core: `MeshLayer` (static or rigid per-body poses), `SkinnedMeshLayer`
   (client skinning, bone poses `(T,J,7)`), `SkeletonLayer` (joints `(T,J,3)` + parents, ideally batched
   spheres/cylinders), `ArrowsLayer` (forces/torques `(T,N,3)` at anchors, via add_arrows), `FramesLayer`,
   `PointCloudLayer`, `CameraLayer` (frustums + image), `GridLayer`.
5. Adapters: robot (Model + q trajectory → batched FK once → MeshLayer/SkeletonLayer/Arrows from tau or f_ext),
   human (BH body → skin weights + bone poses), BVR (ClipData → layers) stay in their packages.
6. Interaction as optional controllers (gizmo/IK targets, picking) on top of layers, not in the core.
Drop: RendererBackend Protocol (keep viser direct; add a second backend only when one exists), mode registry.

## Open questions
- Should the per-frame skeleton use batched meshes (viser `add_batched_meshes`) to cut messages? None of the
  three uses it; not verified that the installed viser version has it.
- Headless/offline rendering and video export are absent everywhere; viser alone cannot do it server-side.
- Line refs marked `~` were inferred from grep offsets inside `apply_frame` (scene.py:602 + offset), not read in full.
