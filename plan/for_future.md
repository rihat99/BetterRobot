# For future — deliberately outside the polish phase

Ideas and unfinished threads the owner may pick up in a later update. Each
entry says what it is, why it waits, and where to start. Nothing here is a
promise; nothing here may be started without an owner decision.

## Native neural-network integration (owner's stated direction)

Make optimization layers combine naturally with `nn.Module`s: residuals backed
by networks, learned priors inside a `Problem`, solver steps inside training
loops, implicit differentiation as the bridge. The polish phase prepares this
(torch-extension mindset, `torch.optim` adapter, slim implicit path, open
differentiable blocks); the actual feature — patterns, examples, perhaps a
`ResidualModule` — is a design task of its own. **Do not implement early.**

## Warp / GPU expansion (the unfinished GPU milestone)

- **Full benchmark baseline:** the 144-selector definition exists
  (`benchmarks/m6_baseline.py`); only SMPL B=1 Torch and four Warp FK cases
  were measured. Panda, remaining batches, IK modes, and graph-record costs
  are open.
- **Warp FK default-on decision:** CUDA-validated, 1.3–2.5× vs compiled Torch
  forward-only; backward (Torch recompute) unmeasured. Owner flips the
  default only with backward numbers.
- **Further kernels:** Lie library, Jacobian, pose-residual, dynamics
  (RNEA/ABA/CRBA), integrate/difference, whole-formulation kernels — all
  unstarted. Each kernel must land with parity + gradcheck + benchmark, per
  the standing kernel rules.
- **CUDA-graph capture productization:** the experimental `GraphExecutor`
  (429 lines + CUDA tests) was deleted in polish phase 1 with zero production
  callers; it lives at git tag/commit `1c8ea5b` (`optim/_graph_executor.py`).
  Resurrect it when a persistent captured solver driver becomes a real goal;
  the missing pieces were a fixed-trip inner/outer driver, an end-to-end
  captured-IK proof, and capture-speed benchmarks.

## Optimizer stack extensions

- **Matrix-free normal route (`NormalCG`):** deleted in polish phase 2 with no
  production caller and no measured need; recover from git when a trajectory
  problem too long for banded Cholesky actually appears.
- **Implicit differentiation v2:** banded/operator backward for long
  trajectories, gradient-quality diagnostics, implicit-vs-unrolled oracle
  tests, named parameter-binding for item weights/kernel scales.
- **Schur elimination** for trajectory + shared/nuisance blocks: waits for a
  second production caller (e.g. body-shape-plus-trajectory fitting).
- **Batched L-BFGS:** never existed (the deleted legacy one was scalar); add
  only against a demonstrated need the `torch.optim` LBFGS adapter can't meet.

## Dynamics and kinematics

- `compute_minverse` (ABA-factorization inverse), `compute_coriolis_matrix`,
  centroidal dynamics derivatives, analytic RNEA/ABA derivatives
  (Carpentier–Mansard) replacing the autograd-derived helpers where speed
  matters.
- Integrators (`semi_implicit_euler`, `symplectic_euler`, `rk4`): stubs were
  deleted from the public surface; implement behind real simulation-adjacent
  demand — or never (the scope fence says no simulator).
- Offset contact points in `solve_contact_forces` (today wrenches apply at
  joint origins, `[force, torque=0]`; a real contact offset contributes
  `r × f`).

## Residuals

- **Yoshikawa manipulability:** the placeholder export was deleted. Add a real
  residual only with an explicit Jacobian-conditioning and singularity
  contract.
- **Acceleration limits:** `JointAccelLimit` was deleted because neither
  `Model` nor `ModelValues` declares acceleration bounds. A future residual
  first needs an owned source and unit/shape contract for those limits.

## Collision

Still a stub package. Decision owed: port a real implementation (owner's
external code is the candidate source), or cut the package entirely until it
has one. No Torch oracle exists, which also blocks any collision kernel work.
The empty `SelfCollisionResidual` and `WorldCollisionResidual` exports were
deleted; a residual API belongs with the eventual package decision.

## Trajectory representations

Manifold-safe B-splines (quaternion-aware, bounds-aware) as a trajectory
parameterization. The Euclidean B-spline basis that ships is a numerical
utility only and is documented as such.

## Benchmarks against the field

The external-competitor definition (`tests/bench/external/definition.md`:
Panda target set, success oracle, budgets) exists; no cuRobo/JAX-class
environment or measurement was ever built. Decide whether public claims need
this before release.

## Infrastructure

- **CI:** still manual-only (`workflow_dispatch`) by owner request. Enabling a
  CPU gate + docs build on push is a one-file change when the owner wants it.
- **GPU CI runner** for the CUDA suite (28 tests currently run by hand on the
  host GPUs).
- **Viewer extras** (recording, COM/path-trace/residual overlays) — deleted as
  stubs during the redesign; rebuild only against real demand.
