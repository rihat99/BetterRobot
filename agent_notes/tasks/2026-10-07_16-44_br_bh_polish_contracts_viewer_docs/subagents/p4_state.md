# P4 state — HumanStudio shell (stopped at the 50k-token budget, 2026-10-07)

Tree is runnable: the OLD shell with targeted fixes, not a half rewrite. Paths under BH `src/better_human/viz/`.

## Done
- `studio/layers.py`: Segmentation/Parts layers get `stale` + `set_visible` override (rebuild on show);
  `StudioLayers._refresh` routes shape/segmentation rebuilds through `viewer.rebuild(layer)` when visible, else
  marks stale (fixes p0 defect #1); `visible` is now a read-only property from `layer.visible` (viewer owns it);
  `rebuild()` (body reload, opacity, radius, edit) carries visibility across `remove_group`; theme default LIGHT.
- `studio/app.py`: studio Layers tab toggles and "Color by segmentation" removed; tab renamed "Display"
  (opacity, joint radius, explode); `data_dir=None` -> `persistence.DEFAULT_ROOT`.
- `studio/persistence.py` (domain-core hook): `DEFAULT_ROOT = <repo>/studio_data`. `studio/__main__.py`: `--data-dir` default None.
- `human_layers.py`: DARK -> LIGHT everywhere. `docs/studio.md` updated.
- Tests: 2 added in `tests/viz/test_studio_layers.py`; `test_segmentation_rebuild_on_edit` now shows the
  segmentation layer first (hidden layers rebuild lazily).
- SMPL-X shape tick (FakeServer, CPU, excl. app `_update_mass`): 0.04 s parts hidden, 0.12-0.21 s parts visible.

## Remaining (exact)
1. Model tab: family (smpl/smplx/mhr), gender, MHR LOD slider, Load; rebuild shape sliders / density / pose dropdown on reload.
2. Mesh colour mode dropdown (skin | segmentation | parts) on Segment tab, one mesh recoloured in place; drop
   separate segmentation layer.
3. FramesLayer driven by `session.frame_set` (vertex position + parent-joint orientation); feed on load/edit.
4. Frames tab saved-sets dropdown (replace free-text name, `frames_editor.py:43-45,130-135`).
5. In-place opacity / joint radius / edit overlay (no `remove_group`); body reload may keep remove_group.
6. `threading.Lock` in `StudioSession` guarding gizmo-worker edits (`gizmo.py` callbacks); document it.
7. Tests: one checkbox per layer + `visible` matches scene after toggling; frames layer marker count after loading
   `studio_data/frames/smplx_neutral/contacts.json`; smpl->smplx->mhr switch with unique node names.
8. Real-server drive on 127.0.0.1:8095 (every control once, timings); screenshots via BR `viewer/capture.py`
   (file exists) into `vis/p4/`.
9. Out of P4 scope: examples 01-04 still `replace(DARK, up="+y")`.

## P4-2a (2026-10-07, uncommitted on BH 83cc96e)
- Model tab (first): Family (smpl/smplx/mhr) / Gender (disabled for mhr) / LOD 0-6 (disabled unless mhr) / Load.
  `HumanStudio.load_model(family, gender, lod)`: set_mode("pose"), density back to uniform, `session.load_body`,
  then `self.tabs.remove()` + `_build_tabs()` (all tabs are body-dependent). Gizmo already clears on "body".
- Frames layer: `layers.frame_poses(session) -> (1, F, 7)` (vertex pos + resolved parent-joint quat); BR `FramesLayer`
  named "frames" replaces HumanView's model-frames layer (default set = body.contact_frames, so not distinct).
  `StudioLayers.refresh_frames()` on "frames" notify (count change -> `viewer.rebuild`); pose/shape update in place.
- Frames tab: "Saved" dropdown (persistence list for model_key) + Load; "Save as" text only for Save; refreshed after
  Save, and rebuilt on model switch.
- Lock: existing `StudioSession._lock` renamed `session.lock` (RLock: nested mutations and .value sets re-fire
  on_update); every GUI on_click/on_update + click/rect pointer callbacks wrapped by `HumanStudio._locked`;
  gizmo drags via `GizmoController._drag_locked`.
- Tests (67 pass): frames layer count after loading `studio_data/frames/smplx_neutral/contacts.json` and +1 on add;
  smpl -> smplx -> mhr switch (key, family, slider counts, unique layer names). FakeTabs gained `remove()`.
- Not tested: saved-sets dropdown refresh; FakeGui.control() returns first match, so stale handles after a switch.

## Remaining for P4-2b
1. Mesh colour mode dropdown (skin | segmentation | parts), one mesh recoloured in place.
2. In-place opacity / joint radius / edit overlay (no remove_group).
3. Real-server drive on 127.0.0.1:8095 (every control once incl. Model Load, timings; check real viser
   add_batched_axes with 0 frames and tab-group remove()).
4. Screenshots via BR `viewer.capture` into `vis/p4/`.
5. docs/studio.md: Model tab, frames layer, saved sets, lock (not updated in P4-2a).

## P4-2b (2026-10-07, uncommitted on BH 83cc96e, on top of P4-2a)
- Done: `layers.BodyMeshLayer` ("mesh", `mode` skin|segmentation|parts) is the only mesh layer; SegmentationLayer and
  PartsLayer kept as drawing providers inside it (never added to the viewer). viser `add_mesh_simple` has no vertex
  colours, so a mode switch = `viewer.rebuild(mesh)` (same layer object/checkbox, children replaced). Explode kept.
  Segment tab: "Colour" dropdown -> `StudioLayers.set_mode`. HumanView no longer used by the studio.
- In place: opacity sets live handle `.opacity`; radius writes `skeleton.radius`/`bone_radius` + `viewer.show()`;
  edit overlay and body switch use `StudioLayers._renew(name)` (dispose, re-run `__init__` on the same object,
  `viewer.rebuild`). No `remove_group` left in BH viz. `StudioLayers.rebuild()` removed.
- Tests (tests/viz 69 pass): mode switch keeps one "mesh" layer + visibility; hidden parts stay hidden (p0 #1);
  body reload keeps layer identities/mode/visibility; opacity/radius/edit keep identities. Old segmentation/parts
  visibility asserts rewritten for the single layer.
- examples 01-04: DARK -> LIGHT. ruff clean.
- NOT done: real-server drive + screenshots. Script scratchpad/drive_p4.py (in-process HumanStudio on :8095 +
  HeadlessClient) hung > 600 s with no output (stdout piped through tail; cause unknown: HumanStudio init on a
  real server, mhr load, or HeadlessClient connect). Next: run unpiped with prints per step. docs/studio.md not
  rewritten (budget).

## P4-3 (2026-10-07, uncommitted on BH, on top of P4-2b)
- Real-server drive: `subagents/drive_studio.py` (in-process HumanStudio + real viser on 127.0.0.1:8095, log to file,
  faulthandler 30 s watchdog per step, HeadlessClient screenshots). Final run: 53/53 steps OK, no traceback, port freed.
  Log: scratchpad p43/drive6.log. Covers model smplx->smpl->mhr->smplx, betas slider/random/reset, density apply,
  pose save/load/reset, opacity/radius/explode, colour skin->segmentation->parts->skin, built-in load, part add +
  assign + save, frames load `contacts` + add + save as `drive_frames` + reload, gizmo select + drag, delete all frames.
- Bug fixed: Model **Load** crashed on real viser (`RuntimeError: Cannot assign to '_tab_labels' on a removed
  GuiTabGroupHandle`; viser's tab-group remove() flags itself removed before removing its tabs). Fix app.py
  `load_model`: remove each tab, then the group. Regression: tests/viz/test_studio_app.py
  `test_model_switch_on_real_viser_server` (real ViserServer on :8097).
- Timings (CPU): init smplx 2.37 s; model switch smpl 0.17 / mhr 0.68 / smplx 1.50 s; shape tick skin 0.05 s,
  parts 0.12-0.28 s; mode switch seg 0.02 / parts 0.10-0.18 / skin 0.002 s; density apply 1.3 s; gizmo drag 7 ms.
- Screenshots: vis/p4/studio_{skin,segmentation,parts,frames_skeleton}.png (y-up, body framed, light bg OK).
  Copied segmentation + frames_skeleton to BH docs/images/. docs/studio.md rewritten (new UI, ~105 lines).
- Note: HeadlessClient reads the Viewer's port, so an injected server needs the same `port=` passed to HumanStudio.
