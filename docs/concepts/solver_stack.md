# The residual / cost / solver stack

Every optimisation problem in BetterRobot — IK, trajopt, retargeting —
is built from the same four-layer stack:

```
Optimizer  →  LeastSquaresProblem  →  CostStack  →  Residual[s]
   (LM, GN, Adam,        (carries x0,        (weighted        (returns r(x),
    LBFGS, MultiStage)    bounds, x → state)  named terms)    optionally J(x))
```

## Residuals

A `Residual` is a pure function of `ResidualState` returning a vector
`r(x)` plus optional `.jacobian(state)`. Residuals know their `dim`
(stable across calls, even when no contact is active — see
{doc}`/design/09_COLLISION_GEOMETRY`). Default Jacobian strategy is
AUTO: prefer analytic, fall back to central FD.

Concrete residuals shipped today: `Pose`, `Position`, `Orientation`,
`JointPositionLimit`, `Rest`, `Velocity`, `Acceleration`,
`TimeIndexed`, `ContactConsistency`, `JointVelocityLimit`,
`JointAccelLimit`, `ReferenceTrajectory`. Stubs: `Jerk`, `Yoshikawa`,
collision, nullspace.

## Costs

`CostStack` is an ordered dict of `(name, residual, weight, kernel)`
quadruples. Weights and kernels are runtime-mutable so a
`MultiStageOptimizer` can re-tune mid-solve via snapshot/restore.

Robust kernels: `l2`, `huber`, `cauchy`, `tukey`. They reweight
`r → ρ(r)` per residual, not per cost stack — different terms can use
different kernels.

## Problem

`LeastSquaresProblem` packs `x0`, `lower`, `upper`, a state factory
(`x → ResidualState`), and the cost stack. Two key extras for trajopt:

- `gradient(x)` — matrix-free product `Jᵀ r` without forming `J`
- `jacobian_blocks(x)` — per-residual blocks, so block-sparse solvers can
  exploit structure

## Optimizer

Pluggable Protocol. Built-in: `LevenbergMarquardt`, `GaussNewton`,
`Adam`, `LBFGS`, `MultiStageOptimizer`, `LMThenLBFGS`. Each consumes a
`LinearSolver` (Cholesky / LSTSQ / CG / SparseCholesky) and a
`DampingStrategy` (Constant / Adaptive / TrustRegion).

## Why this matters for users

Two facts you should remember:

1. **Same stack everywhere.** Once you understand the residual chain,
   IK, trajopt, and retargeting are just different residual mixes.
2. **Everything is runtime-pluggable.** Custom residuals, kernels, and
   linear solvers are `Protocol` objects you can drop in without
   subclassing. See {doc}`/conventions/15_EXTENSION`.

```{seealso}
{doc}`/design/07_RESIDUALS_COSTS_SOLVERS` for the normative spec,
including matrix-free trajopt memory wins and the `ResidualSpec` mini-DSL.
```
