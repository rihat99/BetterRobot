# STATE — BetterRobot / 2026-10-07_02-26_p3_viewer (updated 2026-10-07 05:40 Dubai)

## Pointers
- Plan: /home/rikhat.akizhanov/better/dev/BetterRobot/agent_notes/tasks/2026-10-07_02-26_p3_viewer/PLAN.md
- Design: .../agent_notes/tasks/2026-10-06_20-13_architecture_plan/DESIGN.md (D4, Q6, Q7) and
  .../subagents/study_viewer.md §3–§5
- Previous tasks (done): P1 optim `.../2026-10-06_23-45_p1_optim/REPORT.md`, P2 kin/dyn
  `.../2026-10-07_00-39_p2_model_kin_dyn/REPORT.md`
- Owner is asleep: work automatically, ask nothing, record assumptions. Never push/merge; commit on `dev`.
- Never bind servers to 0.0.0.0 here; viewer default host 127.0.0.1. Telegram pings do not work.

## Status (commits 54ce744 … 9f85980 on dev)
- M1 core + simple layers, M2 robot adapter + mesh layers, M3 tools/plot/examples, M4 rasterizer +
  .viser export, M5 skinned/camera layers, M6 old viewer trashed + docs: DONE.
- Code: `src/better_robot/viewer/` (viewer.py, timeline.py, layer.py, theme.py, _math.py, robot.py,
  layers/{skeleton,arrows,points,frames,ground,mesh,plot,skinned}.py, tools/{gizmo,picking}.py,
  render/{__init__,rasterizer}.py, CLAUDE.md). Tests `tests/viewer/` (19, fake server + 1 loopback).
- Numbers: 0.23 ms/frame Python for 5 batched layers (skeleton 24 joints, arrows, frames, 1000 points);
  Panda+G1 visual+collision+skeleton 4.2 ms/frame (94 mesh nodes, 2 writes each); rasterizer
  160 ms per 640x360 Panda frame (136k faces) on CPU with 192 threads, 487 ms on 1 thread; 300-frame
  Panda `.viser` = 1.7 MB; GIF of 300 frames 45 s. Full suite 1743 passed (known optim failure deselected).
- Artifacts: `artifacts/panda.gif`, `artifacts/panda.viser` (gitignored).

## Next steps
1. M7: mars_aux review of core/rasterizer (context file `subagents/context_review_viewer.md`), fix majors.
2. REPORT.md, final commit, drop the index line; then P4 BetterHuman.
3. GPU baselines (P2 leftover) only on a FREE GPU.

## Assumptions (made off-hours)
- Default host 127.0.0.1 instead of the design's 0.0.0.0 (session rule; explicit `host=` exposes).
- Offline recording without new dependencies (own torch rasterizer + PIL); MP4 only via `ffmpeg` on PATH.
- Old BR viewer deleted (approved roadmap); old examples 01/02/04/05 keep the old `tasks.solve_ik` but use
  the new viewer; `(S, T, nq)` side-by-side batches not implemented (use groups + `offset=`).
- PlotPanel marks the current frame with a point per series (uPlot has no programmatic vertical cursor).
- Composite/mimic specifics untouched; per-mesh nodes (not per-body frames) for rigid meshes.
