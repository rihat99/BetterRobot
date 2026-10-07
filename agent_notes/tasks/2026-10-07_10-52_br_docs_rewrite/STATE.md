# STATE — BetterRobot / 2026-10-07_10-52_br_docs_rewrite (updated 2026-10-07 13:05 Dubai)

## Pointers
- Plan: /home/rikhat.akizhanov/better/dev/BetterRobot/agent_notes/tasks/2026-10-07_10-52_br_docs_rewrite/PLAN.md
- Subagent briefs/notes: `subagents/` (context.md, nav.md, figures.md, writers_brief.md, docstrings_brief.md,
  a2_inventory.md = Data->state mapping, a2b_state.md, a2c_state.md, and a2d_state.md when A2d finishes).
- Worktrees: BR `/home/rikhat.akizhanov/better/dev/BetterRobot`, BH `/home/rikhat.akizhanov/better/dev/BetterHuman` (both `dev`).
- Rules: commit on dev only, never push/merge; trash not rm; servers 127.0.0.1 only; mars_* subagents only;
  parity tests ported never weakened. Owner present (working hours); decisions taken: remove ALL legacy,
  optim2->optim, MkDocs Material (deps added), figures+screenshots, public docstrings rewritten.

## Status (BR HEAD 672c9bc; BH 3df98a0+lock)
- DONE+committed: legacy optim/residuals/tasks/examples 01-07/Sphinx trashed; optim2->optim both repos;
  MkDocs scaffold (mkdocs.yml, API ref, glossary/roadmap/changelog, tests/docs/test_snippets.py); docstrings
  optim/terms/lie/spatial/collision/viewer; figures (8 svg, 5 png incl. banded_sparsity); Pinocchio parity
  suite on the state API; state API additions (joint_jacobian, *_time_variation, centroidal_momentum);
  docs pages: index, getting-started (8), concepts architecture/model-and-state/kinematics/dynamics/io/
  optimization/time-and-structure, guides load-a-robot/differentiate/benchmark/use-with-betterhuman/
  write-a-term/own-the-loop, contributing (3).
- UNCOMMITTED on disk (green, 338 passed in their folders): A2c port of tests/kinematics + contract +
  test_exceptions_wired + test_skeleton_signatures to the state API (trashed test_cache_invariants.py).
- DONE on disk, uncommitted: concepts joints-bodies-frames, lie-and-spatial, collision (writer finished).
- RUNNING at compact time (hand-backs arrive as messages): A2d (ports tests/{dynamics,data_model,io,viewer,
  autograd,bench}, examples 07/08, benchmarks, README snippets); writer for concepts joints-bodies-frames,
  lie-and-spatial, collision; writer for concepts viewer, design-decisions + guides hard-constraints, fit-a-clip.
- src STILL HAS the Data API (nothing removed from src yet). Removal is the next code step (A2e).

## Next steps
1. Collect A2d + the two writers; run `uv run pytest tests -q -m "not bench and not cuda"` and
   `uv run mkdocs build --strict`; commit.
2. A2e (mars_top): remove Data API from src per a2_inventory.md/a2_state.md step 4 (data.py,
   _kinematics_level, Data entry points in kinematics/forward.py, jacobian.py, dynamics/{rnea,crba,aba,
   centroidal}.py keeping raw kernels, StaleCacheError, JointMimic + its guards/test); ADD
   `check_quaternion_norm: bool = False` to `forward_kinematics_state` and restore the pytest.raises half of
   `test_fk_compile::test_free_flyer_quaternion_norm_check_is_opt_in` + port the 2 quaternion tests in
   test_exceptions_wired; port tests/contract/test_boundary_validation_count.py, check test_naming/test_docstrings;
   update src CLAUDE.md files, README; gates: BR suite, BH suite, grep Data/create_data empty, mkdocs strict.
3. Fix-ups: docs/concepts/architecture.md TODO(api) comment (root __all__ after removal); API ref pages follow
   __all__ automatically; figures: panda_ik camera off-centre (optional), figures.md banded line.
4. Code fixes owed (from docstring agents): Viewer.record frames camera on frame 0 only; add(offset=) ignored
   by offline render; GroundLayer ignores Theme.up; optim linearize rejects a windowed term linear in a
   time-free variable without jac (problem.py:576) — fix or document.
5. D: full gates, mars_aux review of docs vs code (claims flagged "unverified" by writers are listed in their
   hand-backs — re-check: LM time linear in knots, cu129/Blackwell note, error table in load-a-robot,
   Pinocchio tolerances by dtype in testing.md), REPORT.md, mark done, drop index line, commit.

## Compact message
BR docs rewrite: legacy removal + MkDocs docs mostly written and committed (672c9bc); Data API removal from
src (A2e) and final gates/review/report remain; three subagents still running at compact.
