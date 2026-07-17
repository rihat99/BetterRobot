# M2b Results — Batched Second-Order Solvers

**Status:** completed and verified on branch `dev` on 2026-07-17, with P2
explicitly deferred because its prescribed fixture does not prove pose
unreachability.

## Outcome

M2b adds named-block `LevenbergMarquardt` and `GaussNewton` solvers with a
frozen `init_state` / pure `update` / detached `run` lifecycle, tensor-only
per-element state, arbitrary leading batch axes, grouped robust objectives,
scaled Madsen--Nielsen damping, isolated factorization failures, and
projected active-set bounds with KKT statuses. The owner-authorized
recommended bounded method was adopted; full reflective TRF remains a SciPy
reference.

## Delivered

- Public `LevenbergMarquardt`, `GaussNewton`, `LMState`, and `LMStatus`
  exports for M2a `Problem`/`Values`. GN is a fixed-damping preset of the same
  guarded update.
- Per-element damping, gain, accept/reject, active-set, factorization,
  convergence, iteration, and status tensors for `B...` batches. Terminal
  elements ride along without moving.
- Grouped robust IRLS models and true grouped-`rho` actual decrease, per-block
  scales, Madsen--Nielsen accept/reject updates, and one candidate residual
  evaluation per step.
- Projected active-set restricted normal systems, a projected-gradient
  safeguard, feasible retraction, projection-consistent prediction, and
  `CONVERGED` / `STALLED_AT_BOUNDS` / `MAXITER` / `FAILED` reporting.
- `cholesky_ex` info-mask/zero-step handling in named-block LM. A failed
  element escalates strictly positive damping, receives one actual attempt at
  `mu_max`, and cannot poison a valid neighbor.
- Prevalidated internal residual/Jacobian/retract/difference paths so public
  M2a validation remains strict while `update` stays structurally sync-free.
- Detached `run` by default plus an explicit `create_graph=True`
  `init_state`/`update`/`finalize` oracle for small unrolled differentiation
  tests. Warm starts retain damping only after exact batch-shape, dtype, and
  device checks and always refresh target-dependent artifacts.
- Batched/ridge-aware standalone `Cholesky` and `LSTSQ`, solver-quality tests,
  a tracked SciPy TRF benchmark, capture-eligibility documentation, and a
  regenerated lock file with SciPy as a development dependency.

## Verification evidence

These focused suites overlap; counts are not summed into a fabricated total.

| Check | Result |
|---|---|
| Pattern, bounds, damping, differentiation, and CPU compile-proxy slice | 43 passed |
| RobotConfig q-to-v bound mappings | 11 passed across revolute trajectories, prismatic, helical, translation, planar translation, nested composites, and free-flyer rejection |
| Kernel consistency + linear-solver contract | 36 passed |
| M2a prevalidated-seam regression slice | 85 passed |
| Optimization/public/hot-path slice | 209 passed |
| Solver-quality probes P1, P3--P6 | 7 passed; P6 completed in 16.1 s |
| Tracked SciPy convergence benchmark | 2 passed |
| Documentation/API checks | 62 passed; prior strict-offline build passed; post-closeout HTML build succeeded with only four expected unreachable-inventory warnings |
| Hygiene | scoped Ruff, `uv lock --check`, and post-report `git diff --check` passed |
| Full repository suite | 1,173 passed, 1 skipped, 3 intentionally deselected by `not bench and not cuda` |

The CPU, float32 convergence reference recorded:

| Case | BetterRobot | SciPy TRF |
|---|---|---|
| P1 bounded interior | 40 iterations, cost `1.306e-12`, position error `1.58e-6 m` | 20 Jacobian iterations, cost `1.096e-10` |
| P3 feasible regularized | 4 iterations, cost `1.96226e-6`, position error `1.15e-4 m` | 5 Jacobian iterations, cost `1.96220e-6` |

The analytic coupled-bound toy shows why active restriction is material:
unrestricted-then-clamped LM lands near `[1, 1]` at cost `1.0`; the restricted
system reaches `[1, 1.6]` at cost `0.1` with active mask `[True, False]` and
`STALLED_AT_BOUNDS`. P6 passed the committed one-`B=128`-call versus 128
`B=1` status/cost/task-space parity protocol; its wall time is not a speedup
benchmark.

No CUDA graph or GPU result is claimed. The compile test is a CPU structural
proxy; M6 owns real warmup/capture/replay certification. CI remains
manual-only at the owner's request and is not cited as verification.

## Removed consumer-facing symbols

- `better_robot.optim.solvers.CG` and deep module
  `better_robot.optim.solvers.cg`
- `better_robot.optim.solvers.SparseCholesky` and deep module
  `better_robot.optim.solvers.sparse_cholesky`
- `better_robot.optim.strategies.TrustRegion` and deep module
  `better_robot.optim.strategies.trust_region`

Read-only consumer greps found no BHF/BVR imports of these symbols. Neither
BetterHumanForce nor BetterVideoReconstruction was modified.

## Deviations and caveats

1. **P2 is deferred, not forced green.** FK at one Panda configuration pushed
   outside selected limits proves that configuration infeasible, not that a
   redundant arm has no other feasible IK solution for the same pose. A
   diagnostic solve found another feasible near-zero-cost solution. A future
   P2 needs a geometrically certified unreachable target.
2. **P8's plan formula was false.** The repository convention is
   `weight(s) = 2 d rho/ds`, because `L2.rho(s)=s/2` and `L2.weight(s)=1`.
   The plan and tests were corrected instead of testing `weight=d rho/ds`.
3. **Convergence is stricter for bounded problems.** Accepted-step
   `xtol`/`ftol` termination is unbounded-only; every bounded success requires
   projected-gradient KKT. Tolerance-only unbounded `CONVERGED` remains
   `implicit_valid=False` unless final KKT passes. This avoids turning
   numerical stagnation into a constrained-success lie.
4. **A small unrolled graph oracle was added despite the original “do not add
   `create_graph` machinery” note.** The binding M2a differentiation contract
   required graph-preserving step evidence. `run` remains graph-free; stable
   implicit backward and active-set stability remain M6 work.
5. **Warm start does not literally skip initialization.** It rebuilds static
   and target-dependent evaluation artifacts, then carries forward compatible
   damping. Reusing stale residuals/Jacobians would be incorrect after target
   changes.
6. **Named-block bound-active damping is assembled inside LM.** Restricting
   the free subspace changes the diagonal system, so the step owns that
   assembly. The standalone linear-solver contract still supports
   `solve(A, b, ridge=None)`.
7. **Standalone Cholesky retains its batched least-squares fallback for the
   legacy surface.** Named-block LM deliberately has no rescue: failed
   elements take a zero step and escalate damping with fixed tensor work.
8. **Finite free-flyer translation boxes fail fast.** World-axis state bounds
   are not axis-aligned in a right-local `SE(3)` tangent. Euclidean and
   supported `RobotConfig` coordinates, including knot-major trajectories,
   are mapped explicitly.
9. **Capture readiness remains conditional.** Masked layouts still move
   static indices to the active device, and custom manifolds/residuals without
   the private prevalidated path own their sync-freedom. M6 must hoist stable
   device layouts and test actual CUDA replay.
10. **The SciPy comparator is not pure fp32.** SciPy drives parameters in
    NumPy float64 while BetterRobot callbacks evaluate residuals/Jacobians in
    torch float32; the benchmark records both dtypes.
11. **The callable-matvec linear-solver form is deferred.** M2b implements
    and tests the dense `solve(A, b, ridge=None)` contract. The protocol
    documents callable matvec support as the M5 extension point; claiming it
    in the current runtime API would be false.
12. **Some requested survey traces were not persisted.** Final P1/P3 metrics,
    formula tests, the active-set ablation toy, and mixed-batch behavior are
    recorded, but per-iteration active-count/`mu` trajectories and a separate
    safeguard-selection ablation are not.
13. **BHF/BVR were intentionally untouched.** Consumer migration remains M4,
    and CI remains stopped/manual-only until the owner revisits it.
