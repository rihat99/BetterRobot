# PLAN — P3 viewer (BetterRobot core, robot adapter, offline recording)

Task id `2026-10-07_02-26_p3_viewer`, branch `dev`, repo `/home/rikhat.akizhanov/better/dev/BetterRobot`.
Owner asleep: no intake interview possible; the plan follows the approved design
(`.../2026-10-06_20-13_architecture_plan/DESIGN.md` D4, roadmap P3, owner decisions Q6, Q7) and the design study
`.../2026-10-06_20-13_architecture_plan/subagents/study_viewer.md` §3–§5 (package layout, signatures, layer data
contracts, deletion list). Deviations are listed under Assumptions in STATE.md.

## Objective
One modular viewer core in BetterRobot on viser (installed 1.0.26): `Viewer`, `Timeline`, `Layer`, built-in layers
fed by preloaded `(T, ...)` arrays, a robot adapter, tools on top (gizmo, picking), export (`.viser`) and recording
without a browser (Q7). BetterHuman layers and the BVR port follow in P4 / later; the core imports neither.

## Facts (checked 2026-10-07 02:30)
- Environment: viser 1.0.26, trimesh 4.11.5, PIL 12.2; no pyrender, imageio, matplotlib, ffmpeg. No new
  dependencies without asking → offline recording = own torch rasterizer writing PNG sequences / GIF through PIL;
  MP4 needs ffmpeg (owner's call, documented).
- Old BR viewer: 2,463 lines under `src/better_robot/viewer/` (backend protocol, render modes, scene, trajectory
  player, panels, overlays); examples 01–08 use it. BH studio under `/home/rikhat.akizhanov/better/dev/BetterHuman/src/better_human/viz`.
- Rule from this session: never bind a server to a network address in tests or examples run here; the library
  default host is `127.0.0.1` (user passes `host="0.0.0.0"` to expose; the design's default was 0.0.0.0).

## Milestones and checks
M1 Core (`viewer.py`, `timeline.py`, `layer.py`, `theme.py`, `_math.py`) + `SkeletonLayer`, `ArrowsLayer`,
   `PointsLayer`, `FramesLayer`, `GroundLayer`. Check: unit tests with a fake scene API (no server) for frame
   pushes, visibility, group removal, timeline clamping/looping; one real-server smoke test on 127.0.0.1 that a
   50-frame skeleton+arrows scene builds and `show(k)` runs for all k under 2 ms/frame of Python time.
M2 Robot adapter `robot.py` (`robot_layers(model, q (T,nq), visual=, collision=, skeleton=, frames=, tau=)`) +
   `RigidMeshLayer` (URDF meshes via the existing `urdf_mesh.py` logic, batched primitives for collision spheres)
   + `MeshLayer`. Check: Panda and G1 trajectories from `examples/optim2/02_trajectory.py` play; FK is called once
   for the whole clip; torque colouring and `(S, T, nq)` side-by-side.
M3 Tools: `PoseGizmo` (transform controls → callback), IK target tool as an example on optim2 (`solve_ik`-style
   loop with `terms.frame_target`), picking (rays/rect on CPU, ported from BH studio). Check: examples 01/02 rewritten
   on the new viewer; the old viewer's examples 01–08 all run on the new API.
M4 Export and recording: `save(path.viser)` through viser's serializer with per-frame sleeps; offline renderer
   (`render/`): a small torch z-buffer rasterizer for meshes, lines (skeleton, arrows, frames) and points reading
   the same layer arrays, camera from the theme or a `CameraLayer`, PNG/GIF through PIL, MP4 through `ffmpeg` on PATH
   if present. Check: a 30-frame Panda clip renders headless in CI-like conditions; image regression test on one
   frame (hash tolerance); `.viser` file loads in the static client (manual, note only).
M5 Plots (`PlotPanel` with cursor), `CameraLayer` (frustum + image), `SkinnedMeshLayer` (client skinning, bone
   poses only) — the skinned layer is tested with a synthetic 2-bone mesh here and used by BH in P4.
M6 Delete the old viewer (renderers/, render_modes/, scene.py, trajectory_player.py, panels.py, overlays/ …) with
   `trash`, move URDF mesh building and skeleton geometry into the new files, update `docs/concepts` viewer chapter,
   `viewer/CLAUDE.md`, contract tests (layer dependencies: viewer imports `io`, `data_model`, `kinematics` only;
   nothing imports `tasks`). Check: full suite green except the known failure; `grep -rn "render_modes\|RendererBackend"`
   empty.
M7 Review (mars_aux, clean context) of core + renderer; REPORT.md; commit; drop the index line.

## Rules
- Viser only through the core files; layers import numpy/torch/viser only; the core never imports BH/BVR/`tasks`.
- No new dependencies. Servers bind to 127.0.0.1 in tests/examples; tests use a fake scene API where possible.
- Deleting the old viewer is in the approved roadmap ("delete the three old viewers"); BH studio changes are P4.
- Commit on `dev` after each milestone; never push. Measure before claiming: frame push time, websocket message
  count per frame, `.viser` size for a 300-frame clip.
