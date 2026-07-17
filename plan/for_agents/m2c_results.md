# M2c Results — First-Order Path, Phases, and Task Rebase

**Status:** complete for the BetterRobot repository on branch `dev`
(2026-07-17). External BHF/BVR import verification is explicitly deferred
because those repositories were kept out of scope.

## Outcome

M2c finishes the named-block task-facing optimization path. BetterRobot now
has a batched, manifold-correct, matrix-free Adam; immutable functional
phases; a named-block `solve_ik`; explicit external IK targets; physical
per-block LM step caps; strict rejection of unsafe non-knot robot trajopt;
and an optimizer-owned canonical `CostStack`. Both roadmap acceptance
problems run through public configuration rather than solver subclasses.

## Delivered

- Public `Adam`, `AdamState`, and `AdamStatus` with reduced tangent moments,
  arbitrary leading batch axes, feasible manifold retraction, per-element
  statuses, compatible warm starts, and no Jacobian assembly. State cost,
  gradient norm, and status describe the returned point after every update.
- Robust `Problem.objective()` and tangent `gradient()` use grouped `rho`
  consistently for first-order work, including scalar objective terms.
- Public `Phase`, `PhaseResult`, and `run_phases()` with weight overrides,
  mask elimination, lazy providers, `on_start`, fresh state per phase, mixed
  LM/Adam stages, exception isolation, and validation even at zero iterations.
- `solve_ik` rebuilt as a `RobotConfig` block plus built-in pose, position,
  orientation, limit, and rest residuals. Pose and active rest targets are
  declared `Problem` parameters and differentiable reads. Batches return
  per-element diagnostics; unbatched calls retain Python scalars.
- Honest IK configuration validation. Unknown selections and non-default
  method-specific fields that would otherwise be ignored fail at the facade.
  Named-block L-BFGS remains an actionable `NotImplementedError`; the staged
  supported preset is `lm_then_adam`.
- LM `block_step_limits` cap named reduced-tangent block norms before both LM
  and projected-gradient retractions. Prediction uses the physical step.
- Public acceptance tests for coupled robot-configuration plus SE(3) camera
  extrinsics, and for ICP-style translation/rotation/scale with independent
  caps, relative damping, and a caller-owned external stopping loop.
- Broad robot-trajopt safety gate: only `KnotTrajectory` is accepted.
  Floating-base, bounded, and multistage B-spline paths fail with an M5
  explanation; the unsafe chain-rule subclass was removed. Evidence is in
  `plan/design_notes/m2c_bspline_trajopt_evidence.md`.
- Canonical `better_robot.optim.{CostStack,CostItem,CostKind}` in
  `optim/cost_stack.py`, with identity-preserving root and `costs.stack`
  compatibility exports. Documentation now states that this legacy stack
  allocates dense concatenations and does not provide phase snapshots.
- Updated API pages, task/solver/performance documentation, layer and public
  import contracts, hot-path lint, roadmap inventory, and the inherited M4
  migration ledger.

## Verification evidence

Overlapping focused suites are listed separately and are not summed.

| Check | Result |
|---|---|
| M2c focused acceptance/API/task slice | 119 passed |
| Final Adam/phase/hot-path correction slice | 69 passed |
| Solver-quality facade probes, including one `B=128` call vs sequential calls | 9 passed |
| Full repository CPU gate | 1,260 passed, 1 skipped, 3 deselected with `-m "not bench and not cuda"` |
| Source hygiene | Ruff check and format-check passed for all 40 changed Python files; `git diff --check` passed |
| Dependency lock | `UV_CACHE_DIR=/tmp/betterrobot-uv-cache uv lock --check --offline` passed; no lock change was required |
| Documentation | HTML build succeeded; four warnings were only unreachable external intersphinx inventories in the offline environment |

The full suite includes the pinocchio-parity and contract tests. CUDA tests
were intentionally deselected: this environment exposes no usable NVIDIA
device, so M2c makes no GPU or CUDA-graph claim. CI remains manual-only at the
owner's request and is not cited as evidence.

## Deviations and deferred work

1. **Phases are functional, not mutable snapshot/restore.** `Problem` and
   `VarSpec` are immutable, so each phase builds a cheap local problem view
   and discards it. This gives the required raise-path isolation without a
   mutable snapshot API. Phase masks intersect the base mask and cannot
   unfreeze a permanently fixed coordinate.
2. **The recommended B-spline drop was applied broadly.** Rather than retain
   separate silent failure modes, every non-`KnotTrajectory` robot solve is
   rejected until M5 supplies manifold interpolation, retraction, bounds, and
   matching sparse chain rules. The probe observed free-flyer quaternion norms
   from `0.151953` to `1.145826`; bounds were absent and multistage replacement
   omitted `dq_dz`.
3. **BHF/BVR were neither modified nor inspected.** The migration ledger is
   explicitly inherited and unverified. Consumer import checks requested by
   the original plan are not claimed and move to M4 when the owner reopens
   that scope.
4. **The `costs.stack` shim remains.** The canonical implementation moved to
   `optim`, but BetterRobot's remaining legacy trajopt/direct callers still
   justify the identity shim. `costs.factory` and old `Data` aliases had
   already been removed in M1 and were not recreated.
5. **Batched L-BFGS was not aliased to Adam.** Both L-BFGS spellings fail
   honestly because per-element histories, line searches, and reset rules are
   not implemented. `lm_then_adam` is a distinct supported choice.
6. **Finite differences remain available as an explicit debug strategy.** It
   is graph-free and slower, but retaining the selection preserves a tested
   diagnostic path without claiming production autodiff performance.
7. **The ICP acceptance model uses three blocks.** Translation, SO(3), and
   log-scale are separate so their `0.20`, `0.50`, and `0.30` physical tangent
   caps are independently configurable; a combined SE(3)+scale block could
   not express those three norms honestly.
8. **IK targets are explicit stable parameters.** Pose and active rest targets
   are named context reads for future implicit differentiation. Tensor item
   weights and robust-kernel scales still lack a stable parameter-role binding;
   M6 must reject disconnected parameters rather than infer identity.
9. **Adam evaluates the trial gradient before returning an update.** This is
   an extra matrix-free VJP relative to a lagged-diagnostic loop, but keeps
   state cost/gradient/status consistent with returned values and permits
   immediate per-element convergence reporting.
10. **Provider caching remains evaluation-local.** Residuals within one
    residual or Jacobian evaluation share FK, while separate evaluations may
    recompute it. This follows the frozen M2a lifetime contract rather than
    restoring legacy cross-call cache behavior.
11. **Strict docs builds are network-limited.** A warnings-as-errors build
    reaches only the four unavailable Python/Torch/NumPy/Trimesh inventories;
    the ordinary offline HTML build has no local content warnings.

M5 owns structured/manifold trajectory work. M6 owns implicit differentiation,
CUDA capture, and GPU evidence. No default backend was changed.
