# P3a brief — viewer polish, batch 1 (BetterRobot)

Read `context.md`, then `p0_viewer.md` in full and `p0_bh_studio.md` §"Ranked defects" 1–2 and "Opinion".
Repo `/home/rikhat.akizhanov/better/dev/BetterRobot`. Scope: `src/better_robot/viewer/**`, `tests/viewer/**`,
`examples/viewer/**`, `examples/07_branching_tree_body.py` (viewer bits only). Do NOT touch docs (another phase),
nor the rasterizer/`record` path yet (batch 2) — but do not break it either (`tests/viewer/test_render.py` must
still pass). BetterHuman's studio uses `SkeletonLayer`, `Theme`, `DARK`, `PoseGizmo`: keep those importable;
where you change their API, list the BH call sites (grep `/home/rikhat.akizhanov/better/dev/BetterHuman/src`)
in your final message — do not edit BH.

## Owner's wishes (verbatim)
"light theme everywhere in viser, our navy blue color. When there is no trajectory we don't need play and
speed. Different joints may have different shapes, not all spheres — revolute can be a cylinder, better shows the
axis of rotation. Sidebar design can be improved. Lights should be viser defaults for now. Aesthetic, elegant,
user friendly, no unnecessary things. Adaptability and modularity. The gizmo IK example lags and gets stuck."
Owner decisions: rename the time axis `frame`→`step` everywhere in the viewer (`num_steps`, `set_step`,
`Timeline` slider label "Step"); "frame" means a robot frame only. Navy: `#1f3a68` (primary), use it as the
viser `brand_color`; keep LIGHT as the default theme, keep DARK available.

## Work items (in this order; run `tests/viewer` after each)
1. Gizmo lag (p0_viewer §4): latest-wins worker inside `PoseGizmo` — one daemon thread + a Condition holding a
   single pending pose; `on_update` overwrites the pending pose, the worker solves only the newest; `on_drag_end`
   triggers a final solve. The user callback runs on the worker thread, never on viser's pool. Test with the fake
   server: 50 rapid updates → the callback runs ≤ a handful of times and the last pose wins. Update
   `examples/viewer/03_ik_gizmo.py` (no shared mutable `state["q"]` races; `torch.set_num_threads` is NOT
   allowed — process-wide setting — so keep the solve small instead). Measure before/after with the scratch
   approach described in p0_viewer §4 (port 8094, stop it after).
2. Lights: remove the custom key/hemisphere lights and `key_light_direction`; call
   `scene.configure_default_lights(enabled=True)` (no shadows). Update `tests/viewer/fake_server.py`.
3. Theme: `Theme` gets `brand_color` (default navy), `control_width`; `apply` uses `configure_theme(dark_mode,
   brand_color, control_layout, control_width, show_logo=False, show_share_button=False)`. Ground/skeleton/frame
   default colours re-tuned for a light background (muted greys, navy accents, no saturated primaries).
4. Rename `frame`→`step`: `Layer.set_frame`→`set_step`, `num_frames`→`num_steps`, `Timeline` internals and GUI
   labels, `Viewer.show(step)`, `record`/`render_frames` parameter names (keep `frames=` where it genuinely means
   robot frames, e.g. `robot_layers(frames=[...])`). Update every caller in src/tests/examples.
5. Timeline only when `num_steps > 1`: build the Timeline folder lazily on the first multi-step layer; a
   single-step scene shows no Play/Step/Speed/Loop at all. `Viewer.play()` on a single-step scene just serves.
6. Sidebar: replace "one folder per group + one checkbox per layer" with one "Layers" folder holding one checkbox
   per layer (label = `group/name` when grouped), created by the viewer and kept in sync with `set_visible`
   (defect #2 in p0_bh_studio). Add `Viewer.rebuild(layer)` that disposes + builds a layer and re-applies its
   current visibility and step (defect #1). Add `Layer.visible` property backed by the viewer. Drop the
   `GroundLayer` isinstance special case (use `fits_camera`/a `in_layer_list: bool` attribute).
7. Joint shapes by kind (p0_viewer §5): `SkeletonLayer` takes full joint poses `(T, njoints, 7)` plus
   `joint_kinds`/`joint_axes` from the model (helper `skeleton_spec(model)`); draws per kind: revolute/helical →
   cylinder along the axis, prismatic → thin box along the axis, spherical → sphere, free_flyer → small triad,
   planar/translation → small cube, fixed → nothing; bones stay as now. One batched mesh per kind. Keep the
   torque colouring path (`tau=`) working.
8. `RobotView.update` isinstance chain → each layer exposes `update(state)`-style method or the view keeps typed
   attributes; pick the simpler and say which.

## Verification
- `tests/viewer` green; full suite green (`CUDA_VISIBLE_DEVICES="" UV_CACHE_DIR=/tmp/betterrobot-uv-cache uv run
  pytest tests -q -m "not bench and not cuda" -p no:cacheprovider`); `uv run ruff check src/better_robot/viewer tests/viewer examples`.
- Screenshots for the orchestrator: run `examples/viewer/01_robot_pose.py` and `02_play_trajectory.py` on port
  8094 and capture with the headless approach from p0_viewer §3 (`subagents/p0_assets/headless.py`, chrome
  wrapper `subagents/p0_assets/chrome.sh`); save PNGs to the task folder `vis/p3a/` (gitignored). Use the GPU
  flags only if `python3 ~/.claude/skills/autoresearch/scripts/gpu_free.py --top 1 --need-gb 2` says FREE;
  otherwise CPU. Stop every server/browser you start.
- Write `subagents/p3a_state.md` (what changed, API diffs, BH call sites affected, screenshot paths).

Final message ≤ 30 lines: items done/undone, API changes (old → new), BH call sites to update, gate tails,
gizmo timing before/after, screenshot paths. If budget runs low: finish the item you are on, write p3a_state.md
with the exact remaining list, stop.
