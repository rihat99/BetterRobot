# Context for subagents — architecture plan task (2026-10-06)

You are one of several agents helping the main agent write a **redesign plan**
(no code changes yet) for two libraries owned by one person.

## Workspace (all paths absolute)

- `/home/rikhat.akizhanov/better/dev/BetterRobot` — BetterRobot, branch `dev`
  (git worktree). PyTorch robotics library: lie, spatial, data_model
  (Pinocchio-style Model/Data), kinematics, dynamics, collision, residuals,
  optim, tasks, io, viewer. ~24k lines in `src/better_robot`. Agent guide:
  `CLAUDE.md`; docs in `docs/concepts`, `docs/conventions`.
- `/home/rikhat.akizhanov/better/dev/BetterHuman` — BetterHuman, branch `dev`.
  Parametric human bodies (SMPL family, MHR) on top of BetterRobot. ~7k lines
  in `src/better_human`. Has `viz/` and `viz/studio/` (viser).
- `/home/rikhat.akizhanov/better/BetterVideoReconstruction` (BVR) — downstream
  research project that uses both. Read-only. `tools/human_optim/`,
  `tools/optim.py`, `tools/viewer/` are the relevant parts.
- Reference material (read-only, never modify):
  `/home/rikhat.akizhanov/better/BetterRobot/references/design/*.md` (one
  summary per external library: theseus, pypose, pyroki, jaxopt, optax,
  crocoddyl, curobo, warp, mujoco_warp, newton, pinocchio, drake, brax, frax,
  ...; plus `BEST_PRACTICES.md`), extracted sources next to it in
  `references/{kin_dyn,optim,sim}/`, and
  `/home/rikhat.akizhanov/better/BetterHuman/references/design/*.md`
  (smplx, MHR, momentum, mink, GMR, soma-retargeter, ...).

## What the owner wants (their words, condensed)

- Architecture is "far from perfect". No backward-compatibility constraint;
  API is not fixed; any module may be restructured.
- Dislikes the **torch / warp duality** — redundant. Decision taken: users
  always pass and get torch tensors with autograd; what sits underneath is
  open, but the end state must have ONE implementation per algorithm.
- "Hates" the current **optim**: "not usable at all, lacks versatility,
  modularity, customization to different projects". Wants an elegant solution:
  easy compatibility with torch optimizers, easy to add own optimizers, easy
  least squares, Jacobians handled elegantly without over-engineering.
  The design will be judged on four use cases: (1) human motion fitting as in
  BVR (long trajectories, staged fits, many weighted losses, contacts/forces,
  mixed Adam/LM), (2) robot IK and trajopt at large GPU batch, (3) human→robot
  retargeting, (4) solver as a differentiable layer inside NN training.
- Kinematics and dynamics must be **very fast**; wants very good profiling and
  benchmarking. Same for BetterHuman.
- **Visualizer**: BetterRobot and BetterHuman each have an old one; BVR has
  another with features the owner likes (trajectory playback, layers,
  skeleton, mesh, forces, torques). Wants one modular, elegant viewer that
  lives inside BetterRobot and serves robots and humans; BetterHuman and BVR
  add layers on top. Currently viser; alternatives are open.

## Rules for you

- Read-only on all source trees. Write only where your brief says (if
  anywhere). Do not spawn agents. Do not install packages.
- Be skeptical of docs and notes: check claims against the code. Cite
  `path:line` for every important claim.
- You have a small budget (about 50k tokens of work). Read selectively:
  list files and sizes first, read the key ones, grep the rest.
- Return raw findings, concrete and specific (names, line counts, numbers).
  No praise, no generic advice. Put open questions at the end.
