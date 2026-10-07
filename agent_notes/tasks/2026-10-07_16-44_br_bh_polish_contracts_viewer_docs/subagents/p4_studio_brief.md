# P4 brief — rebuild the HumanStudio UI shell on BetterRobot's viewer

Read `context.md`, then `p0_bh_studio.md` (defects + opinion; the owner approved "rebuild the UI shell, keep the
domain core"), the "BH viz adaptation" notes at the end of this file, `p3a_state.md` (current BR viewer API:
Layers panel, `Viewer.rebuild`, `Layer.listed/visible/apply_theme`, `set_step`, `nsteps`, `skeleton_spec`,
joint shapes, `PoseGizmo` worker thread + `wait()`), BR `src/better_robot/viewer/{viewer.py,layer.py,theme.py}`,
and BH `docs/contracts/bh-addendum.md` (H-STUDIO rules). Repos: BH `/home/rikhat.akizhanov/better/dev/BetterHuman`
(the work), BR read-only except where noted. Scope: `src/better_human/viz/**`, `tests/viz/**`, `docs/studio.md`,
`src/better_human/viz/studio/__main__.py`. Keep `state.py`, `persistence.py`, `posing.py`, `gizmo.py` and the
editors' logic (`segment_editor.py`, `frames_editor.py`) — change them only where the new shell needs a hook, and
list each such change.

Owner's words: "HumanStudio is just broken. Multiple buttons to turn off and on mesh. Parts mesh appearing
randomly. Segmentation and frames not loading. Option to choose model inside studio. Light theme. Make every tool
actually work. More efficient, user friendly, convenient." Decisions: model chooser = smpl, smplx, MHR (gender /
LOD sub-options); LIGHT theme, navy; the viewer owns visibility.

## Target shell (`app.py` + `layers.py` rewritten; ≈ same size or smaller)
- Sidebar: **Model** tab (family dropdown smpl/smplx/mhr, gender dropdown, LOD slider for MHR, "Load" button;
  reload rebuilds body-dependent widgets: shape sliders by group, density, pose dropdown; status line), **Pose**
  tab (existing posing tools + gizmo), **Shape** tab (sliders, random, reset), **Segment** tab (existing editor:
  built-in / saved dropdown, edit tools, save), **Frames** tab (existing editor + a saved-sets dropdown instead of
  the free-text box; Load/Save), plus the viewer's own **Layers** folder — the ONLY visibility control. Delete the
  studio's Layers tab and the "color by segmentation" hide-the-mesh hack.
- Layers: `BodyMeshLayer` with a colour mode (`skin` | `segmentation` | `parts`) switched by a dropdown on the
  Segment tab — one mesh, recoloured in place, no second/third mesh layer; `SkeletonLayer` from BR with
  `skeleton_spec`; `FramesLayer` driven by `session.frame_set` (vertex positions + parent-joint orientation) —
  authored frames must be visible as soon as they are loaded or edited; `PointsLayer` for selections. Opacity
  and marker radius update in place (no group teardown). Rebuilds go through `viewer.rebuild(layer)`.
- Data dir resolved against the repo root (`persistence.py` default), not the cwd; `--data-dir` still wins.
- No Timeline (single step) — BR already hides it. No per-group folders. No DARK anywhere in the studio.
- `GizmoController` callbacks run on the gizmo worker thread: guard session edits with one `threading.Lock` in
  the session (document it).

## Verification (all without a visible browser)
- `tests/viz`: port the existing tests to the new shell (same assertions where the subject survives; dropped tests
  listed with reason) and add: parts/segmentation never become visible on their own after a shape change (the
  p0 defect #1 repro); one checkbox per layer and `visible` matches the scene after toggling; frames layer shows
  `len(frame_set)` markers after loading `studio_data/frames/smplx_neutral/contacts.json`; model switch
  smpl→smplx→mhr rebuilds without stale handles (scene node names unique, no exceptions).
- Scripted real-server drive (`subagents/p0_assets/drive.py` style, port 8095): exercise every button/slider once,
  assert no traceback, time the slow ones (shape tick must be < 0.5 s on CPU for smplx; report).
- Screenshots via BR's new `viewer.capture.HeadlessClient` if P3b has landed (check `src/better_robot/viewer/capture.py`
  exists); else skip and say so. Save to the task folder `vis/p4/`.
- Full BH suite (`CUDA_VISIBLE_DEVICES="" BETTERHUMAN_MODELS_DIR=$PWD/models uv run pytest tests -q -p no:cacheprovider`,
  known failure test_reduction only), `uv run ruff check src tests`, update `docs/studio.md` to the new UI.
Final message ≤ 30 lines: what the shell is now, files rewritten/kept, domain-core hooks added, tests dropped,
timings, screenshot paths, gate tails. If budget runs low: keep the tree runnable (old shell or new, not half),
write `p4_state.md` with the exact remaining list.

## BH viz adaptation notes (from the P3a adaptation agent, 2026-10-07 19:50)
- `StudioLayers.rebuild` still calls `viewer.remove_group("human")` and re-adds every layer → recreates every
  checkbox on each opacity/radius change. Replace with `Viewer.rebuild(layer)`.
- Studio's own layer toggles (app.py ~133) duplicate the viewer's Layers checkboxes; viewer clicks don't update
  `StudioLayers.visible` → stale visibility after rebuild. Remove the studio's toggles.
- Gizmo callbacks run on the worker thread concurrently with viser GUI callbacks; no session lock exists.
- `DARK` remains the default in `viz/human_layers.py` (lines 23, 69, 89, 103, 144) and `StudioLayers(theme=DARK)`;
  examples 01–04 use `replace(DARK, up="+y")` → switch all to LIGHT.
- BR `RobotView.spheres` (the model) is now `RobotView.sphere_model`; BH did not use it.
