# Context: adversarial review of the new viewer (core, robot adapter, offline rasterizer)

Repo: `/home/rikhat.akizhanov/better/dev/BetterRobot` (git worktree, branch `dev`, HEAD 9f85980). Read-only task:
you do not edit files, commit, or push. Run Python with `CUDA_VISIBLE_DEVICES="" uv run python ...` from the repo
root; scratch scripts go to `/tmp/claude-1003/-home-rikhat-akizhanov-better/d9220856-907e-4a64-b1a1-f6cd72ec6355/scratchpad/review_viewer/`
(create it). No new dependencies. Never bind a server to 0.0.0.0: a real server must use `Viewer(port=<free port>)`
(default host 127.0.0.1) and `viewer.server.stop()` at the end.

## What to review (commits 54ce744..9f85980, `git log --oneline 4503509..HEAD -- src/better_robot/viewer`)
- `src/better_robot/viewer/viewer.py` (`Viewer`: groups, `add`, `remove_group`, `set_visible`, `show`, `save`,
  `record`, `play`), `timeline.py` (`Timeline`: wall-clock thread, loop, speed, GUI), `layer.py` (`Layer`,
  `Geometry`), `theme.py`, `_math.py` (quaternion helpers, `compose_poses`, `rotate`, `quat_z_to`,
  `instance_mesh`, unit meshes).
- `layers/`: skeleton, arrows, points, frames, ground, mesh (`MeshLayer`, `RigidMeshLayer`, `SpheresLayer`),
  plot (`PlotPanel`), skinned (`SkinnedMeshLayer`, `CameraLayer`).
- `robot.py`: `RobotView` / `robot_layers` / `robot_meshes` / `_torque_colors` / `_load_geom`.
- `tools/gizmo.py`, `tools/picking.py`; `render/__init__.py` (`render_frame`, `frame_camera`, `record`),
  `render/rasterizer.py` (`Camera`, `rasterize`, `_project`, `_block_pixels`).
- Tests `tests/viewer/` (fake server in `fake_server.py`), docs `docs/concepts/viewer.md`,
  `src/better_robot/viewer/CLAUDE.md`. Design intent: `agent_notes/tasks/2026-10-06_20-13_architecture_plan/subagents/study_viewer.md` §3.

## Claims to attack
1. Quaternion conventions: BetterRobot poses are `[x, y, z, qx, qy, qz, qw]`; viser wants `wxyz`. Check every
   conversion (`xyzw_to_wxyz`, `PoseGizmo.pose`, `FramesLayer`, `RigidMeshLayer`, `SkinnedMeshLayer`,
   `CameraLayer`, `compose_poses` vs `better_robot.lie.se3.compose`, `rotate` vs the torch rotation) with your own
   random tests. Is `quat_z_to` correct for anti-parallel and near-parallel directions (no NaN, unit norm)?
2. Rigid meshes: `RigidMeshLayer` world pose = body pose ∘ geom origin; compare against the old implementation's
   logic (trashed copy: `trash/render_modes/urdf_mesh.py` uses `lie.se3.compose(body_world, local_pose)`) and
   against `KinematicState.frame_pose` for a frame attached to the same body. Body index == joint index
   assumption in `robot.py` (`state.joint_pose` indexed by `body_name_to_id`): is it valid for every model
   (free-flyer G1, programmatic builder models, mimic joints)? Run Panda and G1.
3. Skeleton: `parents - 1` mapping drops the universe; bones to the universe become roots. Fixed joints get
   small spheres. Torque colours: `_torque_colors` with infinite/zero effort limits, multi-dof joints
   (free-flyer: 6 v), `tau` of shape `(nv,)` vs `(T, nv)`.
4. Viewer loop: `show(k)` pushes each visible layer once in one `server.atomic()`; hidden layers skipped and
   caught up; static layers pushed once; `show()` with no argument force-pushes; `timeline.frame` setter clamps
   and notifies; playback thread drops frames by wall clock and stops at the end when `loop=False`; `pause()`
   from the GUI thread cannot deadlock (`join` on the current thread is skipped); `_set_speed` while playing.
   Try to break it with a fake server (see `tests/viewer/fake_server.py`) and with a real loopback server.
5. Rasterizer: depth test is perspective-correct (`1/z` interpolation) and double-sided; occlusion correct for
   intersecting triangles; triangles partly behind the camera are dropped (acceptable? note it); large triangles
   (ground) binned into 16-px tiles; small ones into 4/8-px anchored blocks — is a triangle with `extent <= 4` whose
   bbox spans two 4-px blocks still fully covered (block anchored at `floor(lo)`, size 4: covers `[lo, lo+4)`; bbox
   width up to 4 ⇒ `hi <= lo + 4`, pixel centres `< hi` ... check the boundary with explicit tests)? Flat shading,
   colours per face. Compare a rendered Panda frame with the viser client's view only if you have a browser
   (probably not: say so). Measure the time per frame (CPU) and memory for a 1920x1080 frame of the Panda scene.
6. Export: `Viewer.save` records all frames with `insert_sleep(1/fps)` into `.viser`; file size for 300 frames
   (`agent_notes/tasks/2026-10-07_02-26_p3_viewer/artifacts/panda.viser`, 1.7 MB) — does it include the mesh
   geometry once and per-frame pose messages? Can it be loaded by `viser`'s static client (parse the msgpack
   header with `msgspec` to count messages at least)?
7. Design: dead code, duplicated helpers, anything over-engineered or missing from the plan
   (`agent_notes/tasks/2026-10-07_02-26_p3_viewer/PLAN.md` M1–M6), names vs `docs/conventions/naming.md`,
   layer rule (`tests/contract/test_layer_dependencies.py`: viser only in `viewer.py`, trimesh only in
   `robot.py`), docstrings matching shapes. Thread safety of `Viewer.show` called from the timeline thread while
   the user calls `viewer.add` from the main thread.

## Report (final message, ≤ 60 lines)
Ranked findings: severity (major/minor/nit), file:line, what is wrong, how you showed it (command + numbers),
proposed fix. Then one line per claim above: confirmed / refuted / not checked. Then anything you could not run.
