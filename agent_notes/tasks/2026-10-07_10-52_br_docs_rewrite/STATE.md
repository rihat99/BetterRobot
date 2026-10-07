# STATE — BetterRobot / 2026-10-07_10-52_br_docs_rewrite (updated 2026-10-07 13:05 Dubai)

## Pointers
- Plan: /home/rikhat.akizhanov/better/dev/BetterRobot/agent_notes/tasks/2026-10-07_10-52_br_docs_rewrite/PLAN.md
- Subagent briefs/notes: `subagents/` (context.md, nav.md, figures.md, writers_brief.md, docstrings_brief.md,
  a2_inventory.md = Data->state mapping, a2b_state.md, a2c_state.md, and a2d_state.md when A2d finishes).
- Worktrees: BR `/home/rikhat.akizhanov/better/dev/BetterRobot`, BH `/home/rikhat.akizhanov/better/dev/BetterHuman` (both `dev`).
- Rules: commit on dev only, never push/merge; trash not rm; servers 127.0.0.1 only; mars_* subagents only;
  parity tests ported never weakened. Owner present (working hours); decisions taken: remove ALL legacy,
  optim2->optim, MkDocs Material (deps added), figures+screenshots, public docstrings rewritten.

## Status (BR HEAD 52245da; BH 3df98a0+lock)
- Committed through 52245da: everything in the earlier status plus viewer fixes, check_quaternion_norm, new panda_ik.png.
- UNCOMMITTED on disk: (a) A2e-2 removed the Data API from src (trashed kinematics/{forward,_fk_matrix}.py,
  dynamics/{rnea,crba,aba,centroidal,_execution}.py, data_model/{data,_kinematics_level}.py, joint_models/mimic.py;
  root __all__ = 21 names). (b) README.md rewritten, README snippets now run in tests/docs/test_snippets.py.
  (c) src/better_robot/{data_model,kinematics,dynamics}/CLAUDE.md on the state API. (d) Docs review fixes
  (3 mars_aux reviewers, 31/31 pages read, ~130 claims, 15 wrong/stale fixed): load-a-robot error table (IRError),
  testing tolerances, viewer sampling, conventions naming, extending joint table, fit-a-clip LM, differentiate,
  use-with-betterhuman, lie-and-spatial (hat_so3 names; Inertia.from_* now take dtype/device), first-model, IK.
  (e) architecture.md TODO resolved. Optim "windowed linear" claim NOT reproduced (no bug).
- RUNNING: A2e-3 (mars_top) porting tests to the post-removal src per subagents/a2e3_brief.md (10 collection
  errors + 15 failures listed in a2e2_state.md; also edits docs/reference/roadmap.md inventory line).

## Next steps
1. Collect A2e-3; fix docs/reference/roadmap.md "center_of_mass ... velocity" sentence (now center_of_mass(model, state));
   gates: BR suite, ruff touched files, mkdocs strict, grep gate (a2e3_brief), BH suite; commit.
2. REPORT.md: pages + line counts, figures, removed line counts (git diff --shortstat cb58627^..HEAD), review
   findings, code fixes, not-done list (gpu-and-batching numbers come from dirty: True JSONs; Material MkDocs-2.0
   banner informational); mark STATE done; drop BR index line; final commit; TG ping.
