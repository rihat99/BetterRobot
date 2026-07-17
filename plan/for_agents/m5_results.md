# M5 results — sparse trajectory structure

**Status:** implementation is complete on branch `dev` (2026-07-17), while
milestone performance acceptance is incomplete/deferred. The mandatory
canonical Phase-C scaling sweep was not run.

## Outcome

- Added an explicit temporal `VarSpec` annotation and checkable residual
  patterns, with symbolic bandwidth analysis cached on `Problem`.
- Added exact dense, block-banded, and matrix-free normal operators. LM routes
  through dense, banded Cholesky, or fixed-work normal CG without constructing
  a global identity/Jacobian/Hessian on the structured paths.
- Added dual legacy/named-block temporal adapters for velocity, acceleration,
  reference trajectories, time-indexed terms, and contact consistency, with
  arbitrary leading batches and analytic dense/temporal block parity.
- Rebased `solve_trajopt` onto a temporal named `q` block, sanitized state
  bounds, hemisphere-aligned ingress, active soft costs, route diagnostics,
  and explicit rejection of unsupported legacy optimizer/constraint/B-spline
  requests.
- Added a fresh-process SMPL-scale CPU benchmark harness with RSS/time caps,
  DNF reporting, route/status evidence, slope calculation, and a normal-suite
  T=50 structured smoke test.

## Verification

- Optim/residual/task focused gate: **771 passed**.
- Solver/LM focused gate: **91 passed**; the SMPL-scale block solve reached a
  maximum relative residual of `2.54e-7`.
- Core structured tests: **41 passed**; dense LM regressions: **51 passed**;
  trajopt task/trajectory tests: **27 passed**.
- Benchmark pytest gate: **4 passed** (structured T=50 child smoke, pending
  baseline schema, selector validation, and conservative process-failure
  classification). In separate direct harness checks, isolated one-update T=50
  dense and structured runs both succeeded, selected the requested routes, and
  agreed in final cost to approximately `8e-10`.
- Full repository CPU gate: **1,494 passed, 2 skipped, 3 deselected** with
  `-m "not bench and not cuda"`. Scoped Ruff, `git diff --check`, roadmap
  inventory, front-page test, generated API reference, Sphinx HTML build, and
  offline lock validation passed. The docs build had only four expected
  offline intersphinx warnings.

## Deviations and deferred work

1. The Phase-A synchronous review stop used the owner's explicit delegated
   unattended-work authorization to accept recommended decisions; it was not
   inferred from silence. The durable decision record is
   `plan/design/m5_sparse_trajectory_structure_design.md`.
2. `TemporalPattern` lives in the residual layer so residual declarations do
   not import the higher-level optimizer package; optimizer analysis consumes
   that protocol without reversing the repository dependency DAG.
3. Schur elimination is deferred because there is no second in-tree production
   trajectory-plus-shared-variable caller. A speculative public abstraction was
   not added.
4. Robot B-splines remain rejected. The existing component-space utility is not
   quaternion-, bounds-, or replacement-safe and was not advertised as a
   manifold trajectory parameterization.
5. The original SMPL benchmark requested `JointPositionLimit`, but the current
   residual returns a non-flat trajectory output and its SMPL `dq/dv`
   projection has zero tangent support because the movable joints have
   `nq != nv`. It was replaced, as approved in Phase A, by a benchmark-only
   nonzero tangent envelope with checkable diagonal temporal structure.
6. A requested dynamic-dimension fallback test cannot be constructed because
   `Residual.dim` is a positive static contract validated at `Problem`
   construction. The contract was not weakened to manufacture that case.
7. The canonical CPU sweep (T=50/125/250/500, five LM updates, warmups and
   repetitions) was not run. The committed baseline therefore remains
   `PENDING_MEASUREMENT`; no slopes, T=500 acceptance, or dense DNF outcome are
   fabricated. GPU measurement is deferred to M6.
8. BHF, BVR, and every external consumer remained untouched. External parity,
   migration, and shim deletion are not claimed. CI remains manual-only by
   owner request.

## Implementation commits

- `6b5e2dd` — approve the Phase-A sparse trajectory design
- `2861169` — add structured trajectory linear algebra
- `7d38aef` — add temporal residual block adapters
- `b6b129d` — reject empty temporal event shapes
- `340c599` — route trajopt through named blocks
