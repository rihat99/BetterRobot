# STATE — BetterRobot / 2026-10-07_10-52_br_docs_rewrite (updated 2026-10-07 13:05 Dubai)

## Pointers
- Plan: /home/rikhat.akizhanov/better/dev/BetterRobot/agent_notes/tasks/2026-10-07_10-52_br_docs_rewrite/PLAN.md
- Subagent briefs/notes: `subagents/` (context.md, nav.md, figures.md, writers_brief.md, docstrings_brief.md,
  a2_inventory.md = Data->state mapping, a2b_state.md, a2c_state.md, and a2d_state.md when A2d finishes).
- Worktrees: BR `/home/rikhat.akizhanov/better/dev/BetterRobot`, BH `/home/rikhat.akizhanov/better/dev/BetterHuman` (both `dev`).
- Rules: commit on dev only, never push/merge; trash not rm; servers 127.0.0.1 only; mars_* subagents only;
  parity tests ported never weakened. Owner present (working hours); decisions taken: remove ALL legacy,
  optim2->optim, MkDocs Material (deps added), figures+screenshots, public docstrings rewritten.

## Status (BR HEAD 0e81bbc; BH 3df98a0+lock)
- DONE+committed through 0e81bbc: legacy trashed; optim2->optim; MkDocs scaffold + API ref; docstrings; figures;
  Pinocchio parity on state API; ALL tests/examples/benchmarks/README on the state API; all 31 docs pages written
  (snippets pass, strict build clean).
- UNCOMMITTED on disk (viewer tests 23 passed, docs viewer snippets pass): viewer fixes — record camera sampled over
  the clip (`up` threaded through render.record/render_frames), group `offset` honoured offline (`_Shifted` view in
  viewer.py), `GroundLayer(plane=None)` follows `Theme.up`, `Layer.fits_camera` replaces the class-name check;
  2 tests in tests/viewer/test_core.py; docs/concepts/viewer.md notes updated.
- Optim claim "windowed term linear in a time-free variable is rejected" NOT reproduced (scratchpad repro: 7 linear
  forms linearize fine); the check at problem.py:576 fires only for ops without a second derivative — correct.
- RUNNING: A2e (mars_top) removing the Data API from src per subagents/a2e_brief.md (also adds
  check_quaternion_norm, restores the pytest.raises half, ports the last Data-mentioning tests, updates CLAUDE.md/README,
  resolves architecture.md TODO(api)). Hand-back arrives as a message; a2e_state.md if it stops early.

## Next steps
1. Collect A2e; gates: BR suite, BH suite, grep Data/create_data/StaleCache/KinematicsLevel/JointMimic empty
   (design-decisions.md excepted), ruff on touched files, mkdocs strict; commit (viewer fixes + A2e together or apart).
2. D: mars_aux review of docs vs code — re-check writers' unverified claims: LM time linear in knots, cu129/Blackwell
   note, error table in load-a-robot, Pinocchio tolerances by dtype in testing.md, JointComposite nq/nv,
   JointRevoluteUnbounded, frame_placements only on model.values, design-decisions "why" sections vs DESIGN.md,
   hard-constraints "infeasible element runs all outer rounds".
3. REPORT.md (pages + line counts, figures, removed line counts, review findings, Material MkDocs-2.0 banner is
   informational), mark STATE done, drop the BR index line, final commit. Optional: panda_ik.png camera.
