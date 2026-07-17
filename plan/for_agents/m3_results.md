# M3 Results — Parametric Model Breadth

**Status:** complete for the BetterRobot repository on branch `dev`
(2026-07-17). No code or evidence was read from or written to any external
repository.

## Outcome

M3 makes one frozen topology usable with differentiable, independently
batched joint placements, body inertias, and frame placements. Public model
coordinates now enforce supported scalar mimic chains, while full joint-space
recursions remain the internal dynamics oracle. Source ordering can be kept
explicitly, human-oriented residuals and mesh inertia are available, and
`integrate`/`difference` run by semantic joint group instead of one Python
dispatch per joint.

## Delivered

- Landed the parity gate first: expanded Pinocchio/helper coverage across
  joint kinds, multi-axis batches, fp32/fp64 FK/RNEA, and loop-oracle cases.
- Defined one right-aligned execution batch across query tensors and all three
  v1 model-value tables. FK, frames, Jacobians, RNEA, ABA, CRBA, and centroidal
  passes use the canonical structure/value seam without hidden casts or
  transfers in joint loops.
- Added validated `Model.with_values(...)`, batched frame placements,
  floating-only dtype moves, integer-table preservation, and a differentiable
  synthetic shape-parameter prototype flowing through FK, named-block IK, and
  RNEA. Removed the dead `Data._model_id` identity plumbing.
- Added dual public/full coordinate layouts for mimic models, affine
  configuration/tangent expansion, projected Jacobians/forces/mass/centroidal
  maps, sign-aware reduced limits and capacities, chain/cycle handling, and a
  constrained ABA solve. Panda now exposes `nq=nv=8` with full counts 9.
- Added opt-in stable Kahn source-order preservation while retaining DFS as the
  default, plus batched multi-DOF `Model.q_permutation` gathers.
- Added `SwingTwistLimitResidual`, `JointRotationPrior`, and batched,
  differentiable, winding-invariant uniform-density `Inertia.from_mesh`.
- Added exact-class manifold groups in `ModelStructure`; built-in Euclidean,
  spherical, free-flyer, continuous, and planar operations are vectorized,
  while custom/composite joints keep exact per-joint fallback semantics.
- Added persistent design evidence for parity, ordering, mimic reduction,
  human support, and manifold-vectorization decisions.

## Verification evidence

Overlapping focused suites are listed separately and are not summed.

| Check | Result |
|---|---|
| Full repository CPU gate | 1,370 passed, 1 skipped, 3 deselected with `-m "not bench and not cuda"` |
| Contract + Pinocchio suites | 480 passed |
| Final mimic/vectorization/Panda focused slice | 38 passed |
| Batched-values/order/human/Warp focused slice | 56 passed |
| Source hygiene | Scoped Ruff and format checks passed; `git diff --check` passed |
| Dependency lock | `uv lock --check --offline` passed; no lock change was needed |
| Documentation | HTML build succeeded; four warnings were unreachable offline intersphinx inventories |
| CI | Remains manual-only by owner request and is not cited as evidence |

The advisory CPU benchmark used an Intel Xeon Platinum 8570, one Torch
thread, fp32, and a 24-joint SMPL-like trajectory with `T=200`. Median
`integrate` time changed from 4.8074 ms to 1.3391 ms (3.59x), and
`difference` from 5.3164 ms to 1.4981 ms (3.55x). The benchmark records both
distributions and has no ratio gate.

## Deviations and deferred work

1. **External consumer work was prohibited.** BHF, BVR, `better_human`, and
   every other external repository were neither inspected nor modified.
   Consumer shim deletion and whole-pipeline parity remain later-stage work.
   The shape-to-FK/IK/dynamics acceptance path uses a synthetic in-tree
   parameterization instead.
2. **Mimic ABA uses a projected solve.** Expanding into unconstrained ABA and
   reducing its result is mathematically wrong. Mimic models form reduced
   CRBA/RNEA equations and solve them; non-mimic models retain the existing
   articulated-body fast path.
3. **Warp honestly falls back for mimic models.** The frozen prototype kernel
   ABI was not expanded speculatively. Torch owns reduced-coordinate mimic
   execution, and no GPU claim is made.
4. **Pinocchio's default Panda loader is not a constrained-mimic oracle.**
   Frozen FK/parity fixtures use an explicitly unconstrained BetterRobot twin
   for full-space recursion checks, while dedicated `Gq/Gv` tests prove the
   constrained public result.
5. **DFS remains the default ordering.** Flipping it would invalidate existing
   indexing parity. Stable Kahn ordering is explicit through
   `preserve_joint_order=True`.
6. **Human residual Jacobians use truthful AD/FD fallback.** Swing/twist has a
   real pure-pi singularity and clamp/wrap cuts; the rotation prior requires
   Lie Jacobian factors. No globally analytic Jacobian is claimed.
7. **`inertia_from_vertex_parts` is deferred.** There is no second in-tree
   caller, and boundary-face ownership plus joint-relative frame semantics are
   underspecified. `Inertia.from_mesh` intentionally supports uniform density;
   synthetic closed meshes replace prohibited consumer-derived fixtures.
8. **Grouped manifold parity uses tight tolerances, not bit equality.** The
   measured maximum differences were `1.49e-8` in fp32 and `2.78e-17` in
   fp64 due to grouped reduction order. Right/local perturbation semantics are
   unchanged.
9. **Batch broadcasting is strict torch broadcasting.** Per-person values for
   `(B,T,...)` queries require an explicit singleton time axis `(B,1,...)`;
   the library does not guess axes or auto-unsqueeze.

No dependency or default backend changed. M3.5 now owns the newly requested
safe cleanup pass before M4.
