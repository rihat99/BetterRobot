# Context for subagents — BetterRobot docs rewrite + legacy removal

Repos (git worktrees on branch `dev`): BetterRobot `/home/rikhat.akizhanov/better/dev/BetterRobot` (BR),
BetterHuman `/home/rikhat.akizhanov/better/dev/BetterHuman` (BH, editable dependency on BR). Run from the repo
directory with `CUDA_VISIBLE_DEVICES="" uv run ...` (CPU only; no GPU jobs). BH models: set
`BETTERHUMAN_MODELS_DIR=/home/rikhat.akizhanov/better/dev/BetterHuman/models`.

Rules: never `rm` — move files away with `python3 ~/.claude/skills/autoresearch/scripts/trash.py <paths>` (lands
in `<repo>/trash/`); do not commit (the main agent commits); never bind a server to anything but `127.0.0.1`;
never skip, xfail or loosen a test to get green — port it or report the number; no new dependencies except
those the owner approved for this task (mkdocs-material, mkdocstrings[python]); match the repo's ruff config
(line length 120, `uv run ruff check` + `ruff format`); comments explain why, never narrate; docstrings: first
line on the opening `"""`, tensor shapes documented like `q (B, nq)`.

Owner decisions (fixed): the docs describe the new design only; ALL legacy is removed from `dev`:
old `better_robot.optim`, `residuals`, `tasks`, examples 01–07, the mutable `Data` API (`Model.create_data`,
`forward_kinematics(model, data)`, `rnea`/`crba`/`aba`/`ccrba` on `Data`), Warp leftovers, unused placeholders.
`optim2` is renamed to `optim` (BR and BH). The only kinematics/dynamics API is the state API:
`better_robot.kinematics.forward_kinematics_state(model, q) -> KinematicState`, Jacobians on that state,
`better_robot.dynamics.state.{inverse_dynamics, mass_matrix, forward_dynamics, bias_forces, centroidal_*,
center_of_mass}` (read `src/better_robot/dynamics/state.py` and `kinematics/__init__.py` for the real names).
Docs toolchain becomes MkDocs Material + mkdocstrings; the Sphinx tree under `docs/` is replaced.

Where things are: `src/better_robot/{data_model,io,lie,spatial,kinematics,dynamics,collision,optim2,terms,viewer}`,
agent guides `src/better_robot/CLAUDE.md` and `src/better_robot/<pkg>/CLAUDE.md`, design history
`agent_notes/tasks/2026-10-06_20-13_architecture_plan/DESIGN.md`, phase reports `agent_notes/tasks/*/REPORT.md`
(P0 measure, P1 optim, P2 model/kin/dyn, P3 viewer; BH P4 in the BH repo). Examples on the new API:
`examples/optim2/*.py`, `examples/08_dynamics.py`, `examples/04_branching_tree_body.py`, BH `examples/0*.py`.
Tests: `tests/` (pytest; `-m "not bench and not cuda"`), contract tests in `tests/contract/`, Pinocchio parity in
`tests/test_pinocchio/` (tolerances there are gates: keep them exactly when porting).
