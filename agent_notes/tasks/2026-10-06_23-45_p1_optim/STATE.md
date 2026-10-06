# STATE — BetterRobot / 2026-10-06_23-45_p1_optim (updated 2026-10-07 02:00)

- Plan: /home/rikhat.akizhanov/better/dev/BetterRobot/agent_notes/tasks/2026-10-06_23-45_p1_optim/PLAN.md
- Design: /home/rikhat.akizhanov/better/dev/BetterRobot/agent_notes/tasks/2026-10-06_20-13_architecture_plan/DESIGN.md (D3, section 6)
- Owner asleep since ~23:30 on 2026-10-06; instruction: work automatically, ask nothing. Record assumptions here.
- Background HTTP server for plan pages on 127.0.0.1:8765 (never bind 0.0.0.0).

## Done (all committed on dev)
- M1: src/better_robot/optim2/ (problem, linear, manifold, kernels, solvers, phases) + src/better_robot/terms/ (robot.py, ik.py).
- M2: time axis. `Problem(time=(...))`, `residual(window=k)`, coloured autodiff, block-tridiagonal BandedNormal,
  `terms.finite_difference`. Joint Jacobians memoized on KinematicState.
- M3 (part): `LBFGS` (batched), `Problem.gradient`, `solve_layer` (implicit, via one differentiable Newton step).
- Tests: tests/optim2/ (15). Full suite: only the 2 known pre-existing failures.
- Numbers (subagents/*.json): IK batch 1024 f32 1 thread: new LM 95.5%/72.5% in 2.5 s vs old 79%/66% in 3.7 s;
  4 restarts 98.8%/98.2%. Panda trajopt: 0.086 s vs old 0.3 s, same result. G1 300-frame tracking (10,500 unknowns):
  8.8 s for 30 LM steps, banded; dense impossible.

## Next steps
- Adversarial review of optim2 + terms by a clean-context mars_aux; fix findings.
- Examples for the four use cases (IK, trajopt, human-style clip fit with phases + Adam + L-BFGS, IK layer in NN training).
- Not done from the design: unroll mode of solve_layer, augmented Lagrangian recipe, arrowhead (shared-over-time
  variables under the banded solver; they work today through the dense path only), BVR person-fit port,
  obstacle trajopt (needs collision, P2).
- Then REPORT.md. Carried from P0: GPU baselines when a GPU is FREE; BVR clip reference.

## Assumptions (made off-hours)
- Temporary package names `optim2` and `terms`; old optim/residuals/tasks are not deleted without the owner.
- IK recipe default: hard joint limits, no rest/limit regularizers.
- Term names have their own namespace; variables, data and cached values share one.
- float32 LM stops on gradient 1e-7 only (ftol/xtol off) because looser values stopped problems early.
