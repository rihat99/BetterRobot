# Shared context for every subagent of task 2026-10-07_16-44_br_bh_polish_contracts_viewer_docs

Two PyTorch libraries, both on their `dev` worktrees:
- BetterRobot (BR): `/home/rikhat.akizhanov/better/dev/BetterRobot` — batched kinematics, dynamics, Lie/spatial
  algebra, nonlinear least squares (`optim`), robot terms (`terms`), collision, io (URDF/MJCF/programmatic),
  viser-based viewer (`viewer`). Agent guide: `CLAUDE.md`; docs site: `docs/` (MkDocs Material, `mkdocs.yml`).
- BetterHuman (BH): `/home/rikhat.akizhanov/better/dev/BetterHuman` — SMPL-family body models built on BR
  (`src/better_human`), a viser "studio" app (`src/better_human/viz/studio`), docs as loose `docs/*.md`.
  Models (weights) in `models/`; set `BETTERHUMAN_MODELS_DIR=$PWD/models`.
- Read-only references: `/home/rikhat.akizhanov/better/BetterRobot/references/design/*.md` (summaries of pinocchio,
  brax, drake, crocoddyl, theseus, warp, pyroki, jaxlie, ... plus BEST_PRACTICES.md).
  Never modify anything there, nor the main checkouts `/home/rikhat.akizhanov/better/{BetterRobot,BetterHuman}`.

Commands (always CPU, always through uv):
```bash
cd /home/rikhat.akizhanov/better/dev/BetterRobot
CUDA_VISIBLE_DEVICES="" UV_CACHE_DIR=/tmp/betterrobot-uv-cache uv run pytest tests -q -m "not bench and not cuda" -p no:cacheprovider
uv run ruff check src tests
uv run mkdocs build --strict
cd /home/rikhat.akizhanov/better/dev/BetterHuman
CUDA_VISIBLE_DEVICES="" BETTERHUMAN_MODELS_DIR=$PWD/models uv run pytest tests -q -p no:cacheprovider
```
Known pre-existing BH failure: `tests/core/test_reduction.py::test_corrective_slice_is_exact_when_dropped_joint_is_at_rest` (1 ulp).

Hard rules:
- Never delete with `rm`; use `python3 ~/.claude/skills/autoresearch/scripts/trash.py <paths...>` (moves into the
  repo's `trash/`).
- Never commit, push, merge, or touch git state (the orchestrator commits). Never bind a server to 0.0.0.0.
- No new dependencies. Never skip/xfail/weaken a test; a test whose subject is gone is dropped and listed.
- Do not edit files outside the scope your brief names. Put open questions in your final message.
- Your final message is read by an orchestrator with no access to your transcript: concrete paths, numbers,
  file:line evidence. Keep it under the length your brief states.

Task folder (notes go here, nowhere else): `/home/rikhat.akizhanov/better/dev/BetterRobot/agent_notes/tasks/2026-10-07_16-44_br_bh_polish_contracts_viewer_docs/`
(`subagents/` for your output files).
