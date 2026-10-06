# STATE — BetterRobot / 2026-10-07_02-26_p3_viewer (updated 2026-10-07 02:35 Dubai)

## Pointers
- Plan: /home/rikhat.akizhanov/better/dev/BetterRobot/agent_notes/tasks/2026-10-07_02-26_p3_viewer/PLAN.md
- Design: .../agent_notes/tasks/2026-10-06_20-13_architecture_plan/DESIGN.md (D4, Q6, Q7) and
  .../subagents/study_viewer.md §3 (layout, signatures, layer contracts), §4 (what to delete), §5 (risks)
- Previous tasks (done): P1 optim `.../2026-10-06_23-45_p1_optim/REPORT.md`, P2 kin/dyn
  `.../2026-10-07_00-39_p2_model_kin_dyn/REPORT.md` (last code commit 72de682)
- Owner is asleep: work automatically, ask nothing, record assumptions. Never push/merge; commit on `dev`.
- Never bind servers to 0.0.0.0 here; viewer default host 127.0.0.1. Telegram pings do not work.
- Env: viser 1.0.26, trimesh, PIL; no pyrender/imageio/matplotlib/ffmpeg → offline renderer is our own torch
  rasterizer + PIL (PNG/GIF); MP4 only if ffmpeg appears on PATH.

## Status
- Task folder created, PLAN.md written. No code yet. Next: M1 core.

## Next steps
1. Read the old viewer once (`src/better_robot/viewer/`, 2,463 lines: keep urdf_mesh.py mesh building and skeleton
   geometry) and BVR's viewer for feature parity (`/home/rikhat.akizhanov/better/BetterVideoReconstruction`, read-only).
2. M1 core + simple layers with fake-scene unit tests; commit. Then M2 robot adapter, M3 tools, M4 export/offline
   renderer, M5 plots/camera/skinned, M6 delete old viewer, M7 review + report.
3. GPU baselines (P2 leftover) only on a FREE GPU.

## Assumptions (made off-hours)
- Default host 127.0.0.1 instead of the design's 0.0.0.0 (session rule; explicit `host=` exposes).
- Offline recording without new dependencies (own rasterizer); MP4 deferred to ffmpeg availability.
- Old BR viewer deleted in M6 per the approved roadmap; BH studio untouched until P4.
