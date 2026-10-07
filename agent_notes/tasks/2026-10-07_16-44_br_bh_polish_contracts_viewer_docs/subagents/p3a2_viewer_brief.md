# P3a-2 brief — viewer polish, batch 1 continued (items 4–6 of p3a_viewer_brief.md)

Read `context.md`, then `p3a_viewer_brief.md` (the original brief; owner wishes and scope), then `p3a_state.md`
(what the previous worker finished: gizmo worker, default lights, navy theme — and its exact remaining list with
grep patterns and BH call sites). Repo `/home/rikhat.akizhanov/better/dev/BetterRobot`. Same scope rules as the
original brief (viewer src/tests/examples only; keep `tests/viewer/test_render.py` passing; do not edit BH).

Do ONLY these three items, in order, running `tests/viewer` after each:
4. Rename the time axis `frame`→`step` across the viewer (`Layer.set_frame`→`set_step`, `num_frames`→`nsteps` (counts follow the Model style `njoints`/`nframes`, so `nsteps`, never `num_steps`),
   `Timeline` internals + GUI label "Step", `Viewer.show(step)`, `record`/`render_frames` parameter names where
   they mean time; keep `frames=` where it means robot frames, e.g. `robot_layers(frames=[...])`). Update every
   caller in src/tests/examples (`grep -rn "set_frame\|num_frames\|frame=" src/better_robot/viewer tests/viewer examples`).
   Docstrings too. List the BH call sites you did NOT change in the final message.
5. Timeline only when `nsteps > 1`: build the Timeline folder lazily when the first multi-step layer is added;
   a single-step scene shows no Play/Step/Speed/Loop. `Viewer.play()` on a single-step scene just serves. Test it.
6. Sidebar: replace "one folder per group + one checkbox per layer" with one "Layers" folder holding one checkbox
   per layer (label `group/name` when grouped), created by the viewer and kept in sync with `set_visible`.
   Add `Viewer.rebuild(layer)` (dispose + build, re-apply visibility and current step) and a `Layer.visible`
   property. Drop the `GroundLayer` isinstance special case in `Viewer._add` (a `Layer.listed: bool = True`
   attribute, False for GroundLayer). Tests for sync and rebuild with the fake server.

Then: full suite (`CUDA_VISIBLE_DEVICES="" UV_CACHE_DIR=/tmp/betterrobot-uv-cache uv run pytest tests -q -m
"not bench and not cuda" -p no:cacheprovider` — failures outside viewer/examples belong to other agents, report
them only), `uv run ruff check src/better_robot/viewer tests/viewer examples`. Append a "P3a-2" section to
`p3a_state.md`: API diffs old→new, BH call sites to update, what remains (items 7–8 are NOT yours).
Final message ≤ 25 lines. Do not spawn agents. Do not commit. No servers needed.
