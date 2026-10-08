# Shared context for every subagent of task 2026-10-08_15-45_br_bh_now_items

You are a subagent. Do your brief only, put open questions in your final message, never ask the user.

## Repos (git worktrees on branch `dev`; work ONLY here)
- BetterRobot (BR): `/home/rikhat.akizhanov/better/dev/BetterRobot` — PyTorch-native robotics library
  (`src/better_robot/{lie,spatial,data_model,kinematics,dynamics,terms,optim,collision,io,viewer}`). Head `05c46d2`.
- BetterHuman (BH): `/home/rikhat.akizhanov/better/dev/BetterHuman` — SMPL-family human bodies on top of BR
  (`src/better_human/{core,families,io,terms.py,viz}`; `viz/studio` = HumanStudio). Head `2b8da78`.
- Read first: each repo's `CLAUDE.md`, BR `src/better_robot/*/CLAUDE.md`, BR `docs/contracts/00-index.md`
  (design contracts `C-*`), BR `docs/concepts/*.md`, BH `docs/concepts/*.md`, BH `docs/contracts/`.
- Main checkouts `/home/rikhat.akizhanov/better/{BetterRobot,BetterHuman}` are READ-ONLY. Never edit them.
- Reference projects (read-only): `/home/rikhat.akizhanov/better/BetterRobot/references/{design/*.md,kin_dyn,optim,sim}`
  (pinocchio, drake, mujoco_warp, newton, warp, brax, crocoddyl, theseus, jaxopt, optax, pyroki, curobo, pypose, jaxlie,
  IsaacLab, mjlab, opensim-core, frax; `design/*.md` are digests), and
  `/home/rikhat.akizhanov/better/BetterHuman/references/{design/*.md,parametric_body,retarget}` (smplx, STAR, SUPR,
  MHR, momentum, anny, SOMA-X, GarmentMeasurements, mink, GMR, soma-retargeter).
  BetterVideoReconstruction (BVR, read-only): `/home/rikhat.akizhanov/better/BetterVideoReconstruction` (its viewer:
  `tools/viewer`).
- Prior architecture plan (what the owner wanted and why): BR
  `agent_notes/tasks/2026-10-06_20-13_architecture_plan/` (PLAN.md, REPORT.md, `site/*.html`), and the last task's
  report `agent_notes/tasks/2026-10-07_16-44_br_bh_polish_contracts_viewer_docs/REPORT.md`.

## Commands
- BR tests: `CUDA_VISIBLE_DEVICES="" UV_CACHE_DIR=/tmp/betterrobot-uv-cache uv run pytest tests -q -m "not bench and not cuda" -p no:cacheprovider`
  (expected: all pass, ~1580). Lint: `uv run ruff check src tests` (82 pre-existing findings in untouched files; do not
  add new ones). Docs: `uv run mkdocs build --strict`.
- BH tests: `CUDA_VISIBLE_DEVICES="" BETTERHUMAN_MODELS_DIR=$PWD/models uv run pytest tests -q -p no:cacheprovider`
  (one known pre-existing failure: `tests/core/test_reduction.py::test_corrective_slice_is_exact_when_dropped_joint_is_at_rest`).
  Lint: `uv run ruff check src tests` (clean).
- Always `uv run`, never system python. Run commands from the worktree you are working in (absolute paths).
- GPU: only if `python3 ~/.claude/skills/autoresearch/scripts/gpu_free.py --top 1 --need-gb 8` says FREE; set
  `CUDA_VISIBLE_DEVICES` explicitly. Otherwise CPU.
- Servers bind to 127.0.0.1 only. Headless Chromium for screenshots:
  `BETTER_ROBOT_CHROME=~/.cache/ms-playwright/chromium_headless_shell-1243/chrome-headless-shell-linux64/chrome-headless-shell`
  with `LD_LIBRARY_PATH=/tmp/claude-1003/-home-rikhat-akizhanov-better/d9220856-907e-4a64-b1a1-f6cd72ec6355/scratchpad/libasound/x/usr/lib/x86_64-linux-gnu`;
  screenshot helper `/tmp/claude-1003/-home-rikhat-akizhanov-better/d9220856-907e-4a64-b1a1-f6cd72ec6355/scratchpad/shot_studio.mjs <url> <out.png>` (env `CHROME=`).

## Rules
- Never `rm`; trash with `python3 ~/.claude/skills/autoresearch/scripts/trash.py <paths>`.
- Do not commit, push, merge or change branches. The orchestrator commits.
- No new dependencies. Never skip, xfail or weaken a test. Keep diffs minimal; match file style; comments say why.
- Type hints and docstrings on public signatures (first docstring line on the `"""` line). Document shapes `(B, nq)`.
- Write your deliverable where the brief says (task folder `agent_notes/tasks/2026-10-08_15-45_br_bh_now_items/`);
  final message = a short summary (≤25 lines) + absolute paths. Nothing is ground truth: check code and run it.
