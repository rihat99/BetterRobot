# P0 — BetterHuman studio audit (mars_aux, 2026-10-07 17:05)

Driven every handler on a real viser server (127.0.0.1:8091, SMPL-X, CPU): no tool raises; `tests/viz` 63 passed
(58 s, fake server). Every owner complaint traces to a concrete cause. Driver script:
`/tmp/claude-1003/-home-rikhat-akizhanov-better/d9220856-907e-4a64-b1a1-f6cd72ec6355/scratchpad/drive.py`.
Paths relative to `/home/rikhat.akizhanov/better/dev/BetterHuman/src/better_human/viz/studio/`.

## Ranked defects
1. **S1 parts mesh appears on its own.** `layers.py:153-156`, `78-81`: `PartsLayer.rebuild` and
   `SegmentationLayer.rebuild` call dispose()+build() directly; new viser handles are visible by default and the
   viewer never re-hides them. Trigger: every shape-slider tick (`layers.py:262-263`), every segmentation edit
   (`269-271`). Measured: parts 0/55 → 55/55 visible after `random_shape` while "Parts" checkbox read False.
   Meshes also land at stale poses because `Viewer._push` skips layers it thinks are hidden.
   Fix: viewer-mediated rebuild that preserves visibility.
2. **S1 three sets of mesh switches disagree with the scene.** Studio Layers tab (Mesh/Skeleton/Frames/Parts,
   `app.py:28-33,132-136`); "Color by segmentation" hides the mesh (`app.py:169-172`); BR `Viewer._add` auto-adds
   a "human" folder with one checkbox per layer, always `initial_value=True`
   (`BetterRobot/src/better_robot/viewer/viewer.py:122-124`) → segmentation/parts/edit show True while hidden.
   That folder is deleted and recreated on every `StudioLayers.rebuild` (opacity, radius, joint select, edit-vertex,
   body reload; `layers.py:217-220,238-257`). Fix: one visibility control per layer, owned by the viewer; option to
   suppress/sync the auto checkbox; segmentation as a colour mode of the mesh, not a second layer.
3. **S1 authored frames never drawn.** Frames live in `session.frame_set` (`state.py:120,219`); the scene's frames
   layer draws model frames baked into the body (`body.frame_poses()` → `core/body.py:325`) and exists only when
   `body.contact_frames is not None` (`layers.py:190-192`) — for SMPL-X neutral there is no `frames` layer at all,
   so the checkbox does nothing. Loading contacts works (status line) but nothing appears. `reload()` doesn't feed
   frames into `bh.load` (`app.py:83-90,289-294`). Fix: FramesLayer driven by `session.frame_set`.
4. **S2 frames "don't load".** `data_dir="studio_data"` relative to cwd (`app.py:52`, `__main__.py:16`). Frames
   Load reads a free-text box defaulting to "frames" (`frames_editor.py:43-45,130-135`) → "no saved frame set
   'frames'"; no dropdown of saved sets (Segment tab has one, `segment_editor.py:59`). Fix: resolve data dir against
   repo root; saved-sets dropdown.
5. **S2 segmentation "doesn't load".** Default is per-joint argmax (55 parts, `state.py:119-126`); authored
   built-ins `families/assets/{smpl,smplx}_vert_segmentation.json` only via "Load built-in". MHR has no built-in
   (`segment_editor.py:17`).
6. **S2 dark theme hard-coded.** `app.py:67` `replace(DARK, up="+y")`; BR Viewer defaults LIGHT (`viewer.py:80`);
   `layers.py:176`, `viz/human_layers.py` default DARK.
7. **S2 slow full rebuilds.** Opacity slider and joint select tear down the whole "human" group incl. GUI folder
   (0.13–0.21 s each); a shape tick 2.1 s (55 part meshes + mass update). Fix: in-place opacity; rebuild parts
   lazily only while visible.
8. **S3 minor.** Status line not cleared by `select_joint(None)`; Timeline folder useless for a single-frame studio;
   `docs/studio.md:10-11` doesn't mention the auto "human" folder.

## Old files — nothing lost
- `studio_data/segmentations/` never existed in git history. Frames were only ever
  `studio_data/frames/smplx_neutral/{contacts,better_contacts}.json` (identical in the main checkout).
- `docs/segmentation/` = 10 PNGs. Commit 7005732 moved `smpl/config/*_vert_segmentation.json|npy` to
  `families/assets/` (mano json and smplx_mini npy deleted; recover with `git show 7005732^:<path>`).
- Expected layout: `<data>/segmentations/<key>/*.json`, `<data>/frames/<key>/<name>.json`, `<data>/poses/<key>/*.json`,
  key = `smplx_neutral` etc. (`persistence.py:31-42`).

## Model chooser
`HumanStudio.__init__` fixes family/gender/lod once. Dependent: body label (`app.py:104`), shape-slider groups
(`105-115`), density widget, pose dropdown, `model_key`/paths, editors' built-in map, gizmo (joint count?).
`session.load_body` + "body" change already rebuilds layers and refreshes editors (reload 6.2 s). Missing: a
family/gender/LOD dropdown and rebuilding the Body tab widgets.

## Part A (partial — contracts-vs-code comparison not done, out of budget)
- BH uses BR: `core/`, `families/` → Model, ModelBuilder, build_model, forward_kinematics_state, KinematicState,
  lie.so/se; `core/mass.py` → spatial.Inertia, spatial.mesh.orient_faces_by_component; `terms.py` → optim.TermFn,
  dynamics.state.inverse_dynamics; `viz/` → viewer (Viewer, Layer, PointsLayer, SkeletonLayer, DARK), viewer.tools
  (PoseGizmo, ray_hit, vertices_in_rect, vertices_in_sphere, CameraFrustum).
- No references to removed BR APIs outside `trash/`; no TODO/FIXME.

## Opinion
Keep the domain core (`state.py`, `persistence.py`, `posing.py`, `gizmo.py`, editors' logic — tested, working);
rebuild only the UI shell (`app.py`, `layers.py`, ≈600 lines) on BR `better_robot.viewer`. Start in BR: Viewer
needs (a) auto per-layer checkbox that respects `set_visible` / can be suppressed, (b) viewer-mediated layer
rebuild preserving visibility, (c) in-place opacity, (d) theme argument. Then the studio = model-chooser tab,
one visibility panel owned by the viewer, a mesh layer with colour modes (skin / segmentation / parts), a frames
layer driven by `session.frame_set`.
