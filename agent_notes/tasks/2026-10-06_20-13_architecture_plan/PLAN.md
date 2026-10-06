# PLAN — architecture plan for BetterRobot + BetterHuman

Task id: `2026-10-06_20-13_architecture_plan`. Branch: `dev` (owner said to
work in the dev branches; no separate task branch). Notes live here in the
BetterRobot dev worktree; the task also covers
`/home/rikhat.akizhanov/better/dev/BetterHuman`.

## Objective

Produce a redesign plan, not code. Explain the current state of both repos
and propose the target architecture for: backend (remove torch/warp
duality), optim (new design), Jacobians, performance + benchmarking,
BetterHuman structure, one shared visualizer.

## Deliverable

1. HTML explainer (explain-html skill), one or a few pages: current status,
   problems, target design, phased roadmap. Easy to read, with figures.
2. Interactive architecture diagrams (archify skill): current and target.
3. `REPORT.md` in this folder pointing at both.

## Intake answers (owner, 2026-10-06)

- Backend: torch tensors + autograd are the public API; the core underneath
  is open; end state has one implementation per algorithm.
- Optim is judged on four use cases: human motion fitting (BVR), robot
  IK/trajopt at large batch, human→robot retargeting, solver as a
  differentiable layer in NN training.
- Visualizer lives inside BetterRobot; BetterHuman and BVR add layers.
  Rendering library (viser or other) is open.
- Numbers for the status page: run existing benchmarks on CPU and on the
  least-loaded GPU, labeled "shared GPU, indicative".

## Steps

1. Scan (6 parallel agents): BR optim/residuals/tasks; BVR optim usage;
   three viewers; BR warp lanes + kin/dyn; BetterHuman; benchmark run.
2. Studies (parallel): optim designs in references + web; backend options;
   viewer libraries; BR core (lie/spatial/data_model/io) quality.
3. Synthesis by main agent: target architecture + roadmap.
4. Adversarial review of the plan (mars_aux, clean context); fix.
5. Build diagrams and HTML; render and inspect.
6. REPORT.md, cleanup, commit on `dev`.

## Assumptions

- BVR is read-only, used as a source of requirements.
- No source changes in this task.
