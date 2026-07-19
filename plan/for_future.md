# For future — deliberately outside this round

Ideas and unfinished threads for a later update. Each entry says what it is,
why it waits, and where to start. Nothing here may be started without an
owner decision. (Carried forward from the polish round 2026-07-18, updated
2026-07-19.)

## Differentiable optimization as a module (owner's stated direction)

Theseus's `TheseusLayer` is the reference: the whole solve as an `nn.Module`,
with backward modes (unrolled, implicit, truncated). The v2 rebuild prepares
this — variables as objects, detached `optimize()` with a guarded implicit
mode, `TorchOptimizer` for first-order — but the module wrapper, the
backward-mode selection, and learned residuals/priors inside a `Problem` are
a design task of their own. See `references/design/theseus.md` §5 and §8.
**Do not implement early.**

## Residual vectorization (Theseus `Vectorize`, simplified)

Group structurally identical residual instances (same class, same dim, same
schema) and evaluate them as one batched call instead of N Python calls.
Natural fit for our fixed-width residual blocks; Theseus needs 475 lines of
schema hashing — ours should be far smaller and on by default if it lands.
Measure first: it only matters for problems with many small residuals.

## Warp / GPU expansion

- **Full benchmark baseline:** the 144-selector definition exists
  (`benchmarks/baseline.py`); only SMPL B=1 Torch and four Warp FK cases were
  measured. Panda, remaining batches, IK modes, and graph-record costs are
  open.
- **Warp FK default-on decision:** CUDA-validated, 1.3–2.5× vs compiled Torch
  forward-only; backward (Torch recompute) unmeasured. Owner flips the
  default only with backward numbers.
- **Further kernels:** Lie library, Jacobian, pose-residual, dynamics
  (RNEA/ABA/CRBA), integrate/difference, whole-formulation kernels — all
  unstarted; each must land with parity + gradcheck + benchmark.
- **CUDA-graph capture:** the experimental `GraphExecutor` lives at commit
  `1c8ea5b` (`optim/_graph_executor.py`); resurrect only when a persistent
  captured solver driver becomes a real goal.

## Optimizer stack extensions

- **Matrix-free normal route (`NormalCG`):** deleted with no measured need;
  recover from git history when a trajectory too long for banded Cholesky
  actually appears.
- **Implicit differentiation v2:** banded/operator backward for long
  trajectories, gradient-quality diagnostics, implicit-vs-unrolled oracle
  tests, named parameter binding for weights/kernel scales.
- **Schur elimination** for trajectory + shared/nuisance blocks: waits for a
  second production caller (e.g. body-shape-plus-trajectory fitting).
- **Line-search L-BFGS beyond the adapter:** `TorchOptimizer` wraps
  `torch.optim.LBFGS` where its closure-based line search fits; a batched
  per-element line search does not exist in torch and gets built only against
  a demonstrated need.

## Dynamics and kinematics

- `compute_minverse`, `compute_coriolis_matrix`, centroidal derivatives,
  analytic RNEA/ABA derivatives (Carpentier–Mansard) where speed matters.
- Integrators: implement behind real simulation-adjacent demand — or never
  (the scope fence says no simulator).
- Offset contact points in `solve_contact_forces` (today wrenches apply at
  joint origins as `[force, torque=0]`; a real offset contributes `r × f`).
- Angular contact consistency (the removed `ContactConsistencyResidual`
  `angular` option) belongs with the offset-contact work.
- **Model-parameter ergonomics:** mass lives in column 0 of
  `ModelValues.body_inertias (nbodies, 10)` with no named accessor; fine for
  the raw seam, unfriendly for "differentiate w.r.t. link mass" users. A
  documented accessor or a small params view is worth designing when model
  identification becomes a real workload.

## Residuals

- **Yoshikawa manipulability:** add only with an explicit
  Jacobian-conditioning and singularity contract.
- **Nullspace regularization** (removed raise-only export): needs a
  posture-projection contract first.
- **Jerk smoothness** (removed raise-only export): needs a third-difference
  temporal declaration and a use case.
- **Acceleration limits:** first needs an owned source and unit/shape
  contract for acceleration bounds on `Model`/`ModelValues`.

## Collision

Still a stub package. Decision owed: port a real implementation (owner's
external code is the candidate source) or cut the package until it has one.
No Torch oracle exists, which also blocks collision kernels.

## Trajectory representations

Manifold-safe B-splines (quaternion-aware, bounds-aware) as a trajectory
parameterization. The Euclidean B-spline basis that ships is a numerical
utility only and is documented as such.

## Benchmarks against the field

The external-competitor definition (`tests/bench/external/definition.md`)
exists; no cuRobo/JAX-class environment was ever built. Decide whether public
claims need this before release.

## Infrastructure

- **CI:** still manual-only (`workflow_dispatch`) by owner request; a CPU
  gate + docs build on push is a one-file change when wanted.
- **GPU CI runner** for the CUDA suite (currently owner-run on the host
  GPUs).
- **Viewer extras** (recording, COM/path-trace/residual overlays) — rebuild
  only against real demand.
