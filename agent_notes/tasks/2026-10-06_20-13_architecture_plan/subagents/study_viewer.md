# Study: rendering library and refined design for the unified viewer

Inputs: `subagents/scan_viewers.md` (inventory of BR/BH/BVR viewers), a local probe of the installed
viser (`subagents/probe_viser.py`, run in the BVR env with `uv run --no-sync`), greps of the reference
design notes, and three web searches (mostly unproductive; see "Sources").

Source tags used in cells: **L** = verified on this machine by the probe; **W** = web search this
session; **K** = model knowledge (cutoff mid-2026), NOT re-verified for late 2026; **R** = repo file.

## 0. Facts established this session

- Installed viser is **1.0.30** (L). Present on `SceneApi`: `add_mesh_skinned`,
  `add_batched_meshes_simple`, `add_batched_meshes_trimesh`, `add_batched_glb`, `add_batched_axes`,
  `add_line_segments`, `add_arrows` (no single `add_arrow`), `add_camera_frustum`, `add_point_cloud`,
  `add_transform_controls`, `add_gaussian_splats`, `add_spline_catmull_rom`/`_cubic_bezier`, `add_glb`,
  `add_label`, `add_image`, `add_box`, `add_cylinder`, `add_icosphere`, `add_3d_gui_container`,
  lights (`add_light_directional/hemisphere/point/spot/rectarea/ambient`, `configure_default_lights`),
  `set_environment_map`, `configure_fog`, `on_pointer_event`. No `add_capsule` (L).
- GUI: `add_plotly`, `add_uplot`, `add_image`, `add_markdown`, `add_html`, `add_progress_bar`,
  `add_tab_group`, `add_upload_button`, `add_modal`, `configure_theme` (L).
- Server: `atomic`, `flush`, `get_scene_serializer`, `request_share_url`, `on_client_connect`;
  client: `get_render` (L). `viser.extras`: `ViserUrdf`, `Record3dLoader` (L).
- Signatures (L, `inspect`): `add_batched_meshes_simple(name, vertices, faces, batched_wxyzs,
  batched_positions, batched_scales, batched_colors, batched_opacities, lod, ...)` → per-instance
  color, scale and opacity exist (torque coloring, contact on/off without re-creating nodes);
  `add_batched_axes(..., batched_wxyzs, batched_positions, batched_scales, ...)`;
  `add_arrows(name, points, colors, shaft_radius, head_radius, head_length, ...)`;
  `add_mesh_skinned(name, vertices, faces, bone_wxyzs, bone_positions, skin_weights (V,B), ...)`.
- viser selects the top-4 bone weights itself (`viser/_scene_api.py:1693-1700`); I saw no
  renormalization in those lines, so BH/BVR's top-4 helpers reduce to an optional renormalize.
- Serializer: `viser/infra/_infra.py:52 StateSerializer` with `insert_sleep(duration)` (:76),
  `serialize() -> bytes` (:82) and `show(height, dark_mode)` (:150, notebook/inline embed).
- Reference projects: curobo and mjlab use viser (mjlab via `mjviser`), crocoddyl and drake use
  meshcat, PyposeWarp mentions rerun (R: `BetterRobot/references/design/{curobo,mjlab,crocoddyl,
  drake,PyposeWarp}.md`). No BetterHuman reference mentions any of them (R grep, empty).
- Consequence for the scan's open question: batched meshes exist in the installed version, so the
  O(people x joints) message cost in BVR (`scan_viewers.md` §4) is fixable without switching library.

## 1. Comparison

Columns: Remote = viewable in a browser from a remote GPU box; Phone = usable on a phone browser;
API = weight of the Python API for "a few lines"; GUI = custom widgets driven from Python; Gizmo =
interactive transform handles with Python callbacks; Time = built-in timeline/playback; Skin = skinned
mesh with GPU/client skinning; Inst = instancing/batched nodes; Plots = time-series next to 3D;
Rec = record/export/embed; Headless = offline render to images/video without a browser.

| lib | Remote | Phone | API | GUI | Gizmo | Time | Skin | Inst | Plots | Rec | Headless | Maint. | License |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| viser 1.0.30 | Y websocket + `request_share_url` (L) | Y (K: React/three.js client, touch orbit) | light, imperative (K) | Y rich (L) | Y `add_transform_controls` (L) | N built-in; we write it (K) | Y `add_mesh_skinned` (L) | Y batched meshes/axes/glb (L) | Y plotly + uplot (L) | Y scene serializer → `.viser` file replayed by static client (L for API, K for format) | P `client.get_render` needs a connected browser (L/K) | active, nerfstudio group, paper 2025 (W arXiv 2507.22885) | Apache-2.0 (K) |
| rerun 0.27-0.28 | Y web viewer + gRPC (W ref.rerun.io web viewer 0.28.1) | P heavy wasm app (K) | light, log-style (K) | N no Python widgets; blueprints only (K) | N (K) | Y best-in-class multi-timeline (K) | N no skinning archetype known; per-frame vertices (K, W no hit) | P instanced archetypes (K) | Y native time-series views (K) | Y `.rrd` files, embeddable web viewer (K) | N (K) | very active; 0.26 perf, 0.27 frame hierarchies (W changelog) | MIT/Apache-2.0 (K) |
| meshcat (+Drake) | Y browser (K) | P (K) | light (K) | P Drake adds sliders/buttons (K) | N (K) | P animation export only (K) | N (K) | N (K) | N (K) | Y static html animation (K) | N (K) | slow upstream; Drake fork active (K) | MIT / BSD (K) |
| Foxglove | Y (K) | P (K) | MCAP/ws protocol, not a scene API (K) | P panels, extensions in TS (K) | N (K) | Y (K) | N (K) | P (K) | Y (K) | Y MCAP (K) | N (K) | commercial; core app no longer open source (K) | proprietary app + MIT SDK (K) |
| polyscope | N desktop/OpenGL (K) | N | light (K) | Y imgui (K) | P (K) | N (K) | N (K) | P (K) | P imgui plots (K) | P screenshots (K) | Y EGL headless (K) | active, academic (K) | MIT (K) |
| Open3D vis | P WebRTC mode (K) | N | medium (K) | P (K) | N (K) | N (K) | N (K) | N (K) | N (K) | P (K) | Y offscreen (K) | slowing (K) | MIT (K) |
| PyVista/trame | Y trame (K) | P (K) | medium-heavy VTK (K) | Y trame/vuetify (K) | P VTK widgets (K) | N (K) | N (K) | P glyphs (K) | P (K) | Y html/gltf export (K) | Y offscreen (K) | active (K) | MIT (K) |
| MuJoCo/mjviser, Newton | Y via viser (R mjlab.md) / N native GLFW (K) | as viser | tied to sim model (K) | as viser | as viser | sim-step driven (K) | N (K) | as viser | as viser | as viser | Y MuJoCo offscreen renderer for MJCF only (K) | active (K) | Apache-2.0 (K) |
| Genesis viewer | N desktop (K) | N | tied to sim (K) | N (K) | N (K) | N (K) | N (K) | Y (K) | N (K) | Y video recorder (K) | Y (K) | active (K) | Apache-2.0 (K) |
| aitviewer | N desktop moderngl (K) | N | medium (K) | Y imgui (K) | P (K) | Y SMPL sequences (K) | Y CPU/GPU SMPL (K) | P (K) | N (K) | Y video export (K) | Y headless (K) | low activity since 2024 (K) | MIT (K) |
| Blender (bpy) | N | N | heavy (K) | N | N | Y keyframes (K) | Y armatures (K) | Y (K) | N | Y high-quality video (K) | Y (K) | very active (K) | GPL (K; our scripts only call it) |
| custom three.js/R3F | Y | Y | we build it | Y | Y | Y | Y | Y | Y | Y | P (Playwright) | ours to maintain | ours |

Reading of the table. Only viser and a custom three.js app cover remote + phone + Python widgets +
gizmos + client skinning + instancing together. Rerun wins on timeline, logging and plots but has no
Python GUI, no gizmos and no skinning (K; the 0.27/0.28 changelog hits did not mention skinning).
Desktop viewers (polyscope, aitviewer, Genesis, Open3D) fail the remote/phone requirement. Headless
offline rendering is the one requirement viser cannot meet alone.

## 2. Recommendation: stay on viser, plus two thin optional exporters

1. **Interactive viewer = viser, used directly** (no backend Protocol). Every feature in the brief
   maps to an existing viser call (§0), the three current viewers already use viser, and two reference
   projects (curobo, mjlab) do too. A custom three.js front end would duplicate viser's client.
2. **Playback is ours**: a ~150-line `Timeline` on top of viser GUI (BVR's wall-clock clock,
   `scan_viewers.md` §1 BVR app.py:473-495). This is the one gap rerun would fill, and it is small.
3. **Recording**: (a) `.viser` scene file via `server.get_scene_serializer()` with `insert_sleep(dt)`
   per frame → shareable, playable in a browser with no Python (L for API; K for exact usage);
   (b) MP4 via `client.get_render(h, w)` per frame while a browser tab is connected (L for API).
4. **Optional, later**: `export/rerun.py` (log the same `(T, ...)` arrays as rerun entities, ~150 l)
   only if the owner wants rerun's timeline/plot tooling for long BVR runs; `export/offline.py`
   with pyrender/EGL or Blender only if true headless video (no browser) becomes a real need. Both
   read the same layer arrays, so they add no coupling to the core.

What would change my mind:
- viser development stops or a release breaks `add_mesh_skinned`/batched meshes → re-evaluate a
  custom three.js client (keep layer arrays, swap the transport).
- Owner needs logging of millions of frames with random access across runs → make rerun the
  primary log viewer, keep viser for interactive tools.
- Headless video in batch jobs (no browser anywhere) becomes routine → promote `export/offline.py`.
- A measured websocket ceiling below ~30 fps for 4 people + 2 robots (not yet measured; see §5).

## 3. Refined design for BetterRobot

### 3.1 Package layout (`src/better_robot/viewer/`, target ~1,600 lines total)

| file | job | lines |
|---|---|---|
| `__init__.py` | exports `Viewer`, `Timeline`, `Layer`, built-in layers, `show_robot` | 30 |
| `viewer.py` | `Viewer`: server, root node per scene, layer registry, sidebar, `atomic` push | 220 |
| `timeline.py` | `Timeline`: T, fps, play/pause/scrub/speed/loop, wall-clock thread, callbacks | 150 |
| `layer.py` | `Layer` base: name, root node, `build`, `set_frame`, `set_visible`, `gui`, `dispose` | 80 |
| `layers/mesh.py` | `RigidMeshLayer` (per-body meshes, poses `(T,B,7)`), `MeshLayer` (static) | 160 |
| `layers/skinned.py` | `SkinnedMeshLayer` (client skinning, bone poses `(T,J,7)`, top-4 weights) | 120 |
| `layers/skeleton.py` | `SkeletonLayer` (batched spheres + batched unit cylinders) | 120 |
| `layers/arrows.py` | `ArrowsLayer` (forces/contacts/torque axes, `(T,N,3)` origin+vector) | 90 |
| `layers/frames.py` | `FramesLayer` (batched axes `(T,F,7)`) | 50 |
| `layers/points.py` | `PointsLayer` (`(T,N,3)` or static, per-point colors) | 70 |
| `layers/camera.py` | `CameraLayer` (frustum `(T,7)` + intrinsics + optional video panel) | 90 |
| `layers/ground.py` | `GroundLayer` (grid, shadow plane) | 30 |
| `layers/plot.py` | `PlotPanel` (uplot series `(T,K)` with a cursor at the current frame) | 70 |
| `robot.py` | adapter: `Model` + `q (T,nq)` → FK once → layers; torque → colors/arrows | 180 |
| `tools/gizmo.py` | `PoseGizmo` (transform controls, `on_change` callback); IK target tool | 100 |
| `tools/picking.py` | ray/rect picking on CPU (moved from BH studio picking.py) | 120 |
| `export.py` | `save_viser(path)`, `record_mp4(path, fps)` | 80 |
| `theme.py` | palette, lights, background, GUI theme in one dataclass | 50 |
| `_math.py` | xyzw→wxyz, shortest-arc z→dir (batched), top-4 weights | 50 |

Imports: viewer depends on `data_model`, `kinematics` (only in `robot.py`) and `io` (meshes); the
core files (`viewer`, `timeline`, `layer`, `layers/*`) import only numpy/torch/viser. Drop the
current dependency on `tasks` (scan §5: Visualizer knows `IKResult`); the IK tool takes a callback.

### 3.2 Core types (signatures)

```python
class Timeline:
    def __init__(self, num_frames: int, fps: float = 30.0, loop: bool = True) -> None: ...
    frame: int                                   # current index, setter clamps and notifies
    def play(self) -> None: ...                  # starts wall-clock thread; skips frames if slow
    def pause(self) -> None: ...
    def set_speed(self, speed: float) -> None: ...
    def on_frame(self, callback: Callable[[int], None]) -> None: ...
    def build_gui(self, gui: viser.GuiApi) -> None: ...   # play toggle, slider, speed, fps label

class Layer:                                      # subclass; no Model/Data in the protocol
    name: str
    num_frames: int                               # 1 = static
    def build(self, scene: viser.SceneApi, root: str) -> None: ...   # create nodes once under root
    def set_frame(self, frame: int) -> None: ...  # write preloaded slice k into existing handles
    def set_visible(self, visible: bool) -> None: ...
    def build_gui(self, gui: viser.GuiApi) -> None: ...  # optional per-layer controls
    def dispose(self) -> None: ...

class Viewer:
    def __init__(self, port: int = 8080, theme: Theme | None = None, host: str = "0.0.0.0") -> None: ...
    timeline: Timeline                            # grows to max(layer.num_frames)
    def add(self, layer: Layer, group: str = "scene") -> Layer: ...  # group = toggle/teardown unit
    def remove_group(self, group: str) -> None: ...   # one scene.remove on the group root
    def add_plot(self, name: str, series: ArrayLike, labels: list[str]) -> PlotPanel: ...
    def show(self, frame: int = 0) -> None: ...   # push frame k for all layers inside server.atomic()
    def play(self, block: bool = True) -> None: ...
    def save(self, path: str | Path) -> None: ...     # .viser via scene serializer
    def record(self, path: str | Path, fps: float | None = None, size=(1080, 1920)) -> None: ...
    server: viser.ViserServer                     # escape hatch, documented
```

`Viewer.show(k)` is the only place that pushes a frame: `with server.atomic(): for layer in visible:
layer.set_frame(k)`. Invisible layers skip `set_frame`; on re-show they catch up to the current frame.

### 3.3 Built-in layers (data contract, all numpy float32, `T` may be 1)

- `RigidMeshLayer(meshes: list[trimesh.Trimesh], body_index: (M,), poses: (T,B,7), colors=None)`:
  one node per mesh, per frame only `wxyz/position` writes; with repeated primitives (collision
  spheres/capsules) it uses `add_batched_meshes_simple` with `(M,)` instance transforms.
- `SkinnedMeshLayer(vertices: (V,3), faces: (F,3), weights: (V,J), bone_poses: (T,J,7))`: viser
  `add_mesh_skinned`; per frame writes J bone poses (one message batch, no vertices).
- `SkeletonLayer(joints: (T,J,3), parents: (J,), radius, colors: (T,J,3) | None)`: two batched nodes
  (spheres, unit cylinders); per frame two array writes instead of 2J node updates.
- `ArrowsLayer(origins: (T,N,3), vectors: (T,N,3), scale, colors)`: `add_arrows` per frame (replace
  arrays, no remove/add), zero-length arrows for inactive contacts so N stays fixed.
- `FramesLayer(poses: (T,F,7), axes_length)`: `add_batched_axes`.
- `PointsLayer(points: (T,N,3) | (N,3), colors)`: fixed N per layer; padding with NaN-free hidden points.
- `CameraLayer(poses: (T,7), K: (3,3), image_size, video: VideoSource | None)`: frustum + GUI image.
- `GroundLayer(height=0.0, up="z")`, `PlotPanel(series: (T,K))` with a vertical cursor at frame k.

### 3.4 Keeping per-frame updates cheap

1. Precompute: adapters run FK/LBS-bone math once over all T on GPU, `.cpu().numpy()` once.
2. Batched nodes: skeleton, frames, collision spheres as batched meshes/axes: O(layers) messages per
   frame, not O(people x joints) (fixes scan §4 BVR limit).
3. Client skinning: send bone poses only (`(J,7)` per frame), never vertices.
4. One `server.atomic()` per frame; playback thread drops frames by wall clock (BVR app.py:483-493).
5. Fixed array sizes per layer (pad, hide) so no node is created or removed during playback.
6. Lazy per-frame sources (video frames, per-frame point clouds) read on demand via a small LRU,
   instead of one node per frame (scan §4: BVR pre-creates per-frame cloud nodes).

### 3.5 Robots

`robot.py` is the only viewer file that knows `Model`:
`robot_layers(model, q: (T,nq), *, visual=True, collision=False, skeleton=False, frames=(),
tau: (T,nv) | None = None, f_ext=None) -> list[Layer]`. It calls `forward_kinematics` once on the
whole `(T,nq)` batch, reads body poses `(T,B,7)`, builds meshes from the stored `io.IRModel` (current
`urdf_mesh.py` logic), and maps `tau` to per-joint colors (`|tau|/effort_limit` through a colormap) on
the skeleton spheres, or to `ArrowsLayer` along joint axes. Batch dims beyond T: `q (S,T,nq)` gives
S copies offset along x (side-by-side compare) or overlaid with alpha.

### 3.6 BetterHuman and BVR plug in without BetterRobot knowing them

- BH ships `better_human.viz.human_layers(body, params_seq, *, mesh=True, skeleton=True,
  contacts=None) -> list[Layer]`: computes bone poses `(T,J,7)` and top-4 weights once, returns core
  layers. It subclasses `Layer` only for genuinely new visuals (e.g. segmentation parts).
- BVR keeps its loaders and writes `clip_layers(ClipData) -> dict[group, list[Layer]]`; its
  990-line `ClipScene` becomes a list of core layers plus a clip dropdown that calls
  `viewer.remove_group("clip")` then `viewer.add(...)`.
- Rule: the core accepts arrays and `Layer` subclasses, never imports BH or BVR.

### 3.7 Interactive tools sit on top

A tool is an object holding a `Viewer` and callbacks; it is not a layer and not in the core loop.
- `PoseGizmo(viewer, name, pose, on_change)` wraps `add_transform_controls`.
- IK target tool (in examples or `tasks`): gizmo `on_change` → `solve_ik` → `viewer.show_pose(q)`;
  this inverts today's viewer→tasks dependency.
- BH studio = a BH app built from core `Viewer` + BH layers + its editors (segment, frames, posing,
  persistence). Its `StudioSession` observer stays in BH; on change it rewrites the layer's T=1 arrays
  and calls `viewer.show(0)`. Picking moves to `viewer/tools/picking.py` (generic over vertices).

### 3.8 Export and recording

- `viewer.save("clip.viser")`: serializer captures the scene, then for k in range(T) push frame k and
  `insert_sleep(1/fps)`; output plays in the static viser client (shareable, no Python).
- `viewer.record("clip.mp4")`: needs one connected browser; loops frames, `client.get_render`,
  writes with imageio-ffmpeg (optional dependency, lazy import).
- Later, optional: `export_rerun(layers)`, `render_offline(layers)` reading the same arrays.

### 3.9 Theme

One `Theme` dataclass: light and dark palettes (background, ground, robot gray, human skin, joint,
contact, force, torque colormap), `configure_default_lights` off + one directional key light + a
hemisphere fill + `set_environment_map` for soft reflections, `gui.configure_theme(dark_mode=...,
control_layout="collapsible", brand_color=...)`, sidebar width from the theme (replaces BVR's 24em
hack). Shared palette replaces BR themes.py, BH base.py:16-35 constants and BVR constants.

### 3.10 User code

```python
# 1. view a robot pose
import better_robot as br
from better_robot.viewer import Viewer, robot_layers
model = br.io.load("panda.urdf")
viewer = Viewer()
for layer in robot_layers(model, model.q_neutral[None]):
    viewer.add(layer)
viewer.show()

# 2. play a trajectory, joints colored by torque, torque plot next to 3D
viewer = Viewer()
for layer in robot_layers(model, q_traj, skeleton=True, tau=tau_traj):   # (T,nq), (T,nv)
    viewer.add(layer)
viewer.add_plot("tau", tau_traj, labels=model.joint_names[1:])
viewer.play()

# 3. human clip: mesh + skeleton + contact forces
from better_human.viz import human_layers
from better_robot.viewer import ArrowsLayer
viewer = Viewer()
for layer in human_layers(body, params_seq):
    viewer.add(layer, group="person0")
viewer.add(ArrowsLayer("contacts", origins=cop, vectors=grf, scale=1e-3), group="person0")
viewer.play()

# 4. compare two trajectories side by side
viewer = Viewer()
for name, q, dx in [("ref", q_ref, 0.0), ("ours", q_ours, 1.0)]:
    for layer in robot_layers(model, q, offset=(dx, 0, 0), name=name):
        viewer.add(layer, group=name)
viewer.play()
```

## 4. What to delete and migration order

Delete (trash) from BR viewer: `renderers/` (RendererBackend Protocol, ViserBackend 375 l,
MockBackend 158 l), `render_modes/` registry and `RenderContext`, `scene.py` modes dict,
`trajectory_player.py` (blocking `play`), `panels.py` (no caller), `helpers.py` (moves to `_math.py`),
`Visualizer.add_ik_targets/add_ik_result` (move to IK tool). Keep and port: URDF mesh building
(urdf_mesh.py), skeleton geometry, ForceVectors logic into `ArrowsLayer`.
From BH: `viz/skinned_mesh.py` (CPU LBS re-upload), the `Layer` protocol in `studio/layers/base.py`,
duplicated helpers (`bone_transform`, xyzw→wxyz, `_top4_normalized_weights`). Keep editors.
From BVR (owner's repo, read-only for us; migrate when the owner chooses): `ClipScene` node dicts,
`apply_frame` if-chains, helpers duplicated in `helpers.py:91,110`; keep loaders and `ClipData`.

Order:
1. Core: `viewer.py`, `timeline.py`, `layer.py`, `theme.py`, `_math.py` + skeleton, arrows, points,
   ground layers. Check: replay a BVR clip's skeleton+forces from saved arrays at 30 fps.
2. `robot.py` + mesh layers; port examples 01/IK; remove old BR viewer in the same change.
3. `SkinnedMeshLayer`; BH `human_layers`; delete BH `skinned_mesh.py`.
4. BH studio onto core `Viewer` (largest step; editors unchanged).
5. Export (`save`, `record`), plots, camera layer.
6. BVR `clip_layers` adapter; retire `ClipScene` and `view_gt.py`.

## 5. Risks and open questions

- Not measured: websocket throughput for batched updates (e.g. 4 people x 55 joints + 2 robots at
  30 fps over a laptop VPN, and on a phone). Measure before freezing §3.4; a 50-line bench script.
- `add_mesh_skinned` keeps 4 weights per vertex (L, `_scene_api.py:1694`); max bone count in the
  client shader was not checked. MHR has many joints; check its bone count fits.
- `.viser` file size for long clips: each frame's arrays are serialized; a 10-minute clip with
  skinned meshes may be large. Unmeasured.
- `client.get_render` needs a browser connected; true headless video stays open (§2 item 4).
- Batched-mesh per-instance colors: the parameter exists (L); whether writing `batched_colors`
  on an existing handle per frame is cheap (vs re-sending geometry) is not checked.
- BVR browses 18,076 scenes with lazy per-scene load (BVR `CLAUDE.md`, viewer paragraph); the
  `remove_group`/`add` clip swap must stay as fast as today's `/clip` root removal.
- Viser version pinning across BR/BH/BVR envs (BVR has 1.0.30; BR/BH env versions not checked).
- Open for owner: rerun exporter wanted at all? Offline renderer (pyrender vs Blender) wanted?
  Should the Viewer support several timelines (e.g. two clips at different fps) or resample to one?
- Late-2026 state of rerun (skinning, Python callbacks) and viser release notes could not be
  confirmed by web search; cells tagged K may be stale.

Sources: arXiv 2507.22885 (Viser paper, https://arxiv.org/abs/2507.22885v1);
rerun changelog (https://rerun.io/changelog); rerun web viewer 0.28.1
(https://ref.rerun.io/docs/js/0.28.1/web-viewer); rerun URDF howto (https://rerun.io/docs/howto/urdf).
