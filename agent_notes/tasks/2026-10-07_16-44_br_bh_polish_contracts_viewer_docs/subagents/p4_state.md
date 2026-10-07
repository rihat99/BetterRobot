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
