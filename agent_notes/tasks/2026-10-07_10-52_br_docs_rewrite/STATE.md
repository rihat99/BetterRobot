# STATE — BetterRobot / 2026-10-07_10-52_br_docs_rewrite (updated 2026-10-07 11:40 Dubai)

## Pointers
- Plan: /home/rikhat.akizhanov/better/dev/BetterRobot/agent_notes/tasks/2026-10-07_10-52_br_docs_rewrite/PLAN.md
- Worktrees: BR `/home/rikhat.akizhanov/better/dev/BetterRobot`, BH `/home/rikhat.akizhanov/better/dev/BetterHuman` (both `dev`).
- Rules: commit on dev only, never push/merge; trash not rm; servers 127.0.0.1 only; GPU only when FREE;
  mars_* subagents only; parity tests ported, never weakened. New deps approved: mkdocs-material, mkdocstrings.

## Status
- A1 DONE: BR cb58627 (legacy optim/residuals/tasks + examples 01-07 + Sphinx tree trashed; optim2 -> optim;
  examples/01-06 are the former optim2 examples; MkDocs deps in dev extras; figures in docs/figures, screenshots
  in docs/images — see subagents/figures.md; fixes still owed: architecture.svg edge optim/terms, panda_ik camera,
  banded_sparsity.png missing). BH 3df98a0 + lock commit (rename). BR suite green except
  tests/contract/test_roadmap_stub_inventory.py (needs docs/reference/roadmap.md); ~90 pre-existing ruff errors
  in untouched files (io/parsers/urdf.py, benchmarks/run.py, tests/spatial).
- RUNNING (11:35): A2 Data-API removal (mars_top; inventory -> subagents/a2_inventory.md); MkDocs scaffold +
  API reference + tests/docs (mars_top_opus); docstrings optim+terms, lie+spatial+collision, viewer.

## Next steps
1. Collect A2 -> commit; then docstrings for data_model/io/kinematics/dynamics (after A2).
2. Collect scaffold -> commit; launch writers (getting started, concepts x2, guides+contributing).
3. Figures fix-up (light agent), archify map, D: build --strict, snippet tests, mars_aux review, REPORT.md.

## Code fixes owed (found by docstring agents, not yet done)
- viewer.py: `Viewer.record` frames the camera on frame 0 only (render.record samples the clip).
- viewer.py + render: `add(offset=)` moves only the viser node; offline render ignores it.
- layers/ground.py: `GroundLayer` plane does not follow `Theme.up`.
