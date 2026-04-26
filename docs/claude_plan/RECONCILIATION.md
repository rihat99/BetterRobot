# Reconciliation with the gpt_plan review

A second strategic review of BetterRobot was authored under
`docs/gpt_plan/`. It read this folder's proposals and gave a sharp
critique. This file logs how the two were reconciled — what was
adopted, what was adopted with changes, and what was rejected.

The reconciliation is **opinionated**. The colleague's critique
caught real bugs and surfaced ideas this folder was missing; both
plans benefit from being read together. Where they disagreed, the
choice taken below is the one that survived the back-and-forth.

The corresponding meta-document on the gpt_plan side lives at
[`../gpt_plan/09_claude_plan_review.md`](../gpt_plan/09_claude_plan_review.md);
this file is its mirror image.

---

## Adopted directly (with edits to existing proposals)

### Critical correctness fixes

| Issue caught | Where | Fix |
|--------------|-------|-----|
| **SE3 exp left-Jacobian formula** had `V = a · I + b · W + c · W²`. The leading term should be just `I`, not `a · I`. The Taylor expansion of `c` at `θ → 0` was also wrong (missing the `1/6` leading term — would give `c → 0` instead of `c → 1/6`). | [03_replace_pypose.md](accepted/03_replace_pypose.md) | Formula corrected to `V = I + b · W + c · W²` with `b = (1−cos θ)/θ²`, `c = (θ−sin θ)/θ³`; Taylor of `c` is now `1/6 − θ²/120`. A "do not copy as-is" caveat banner is added at the top of the implementation sketch — the formula needs derivation discipline, not transcription. |
| **Lie types should live in `lie/types.py`, not `spatial/`** | [01_lie_typed_value_classes.md](accepted/01_lie_typed_value_classes.md) | `SO3` / `SE3` / `Pose` move from `spatial/` to `lie/types.py`. `spatial/__init__.py` re-exports them as a convenience. Reasoning: `spatial` already depends on `lie`, so siting the types in `spatial` would invert the layering story. |
| **`.tensor`, not `.data`** for the new typed wrappers — `Data` is a class name in this library; `data.data` next to `pose.data` reads badly. | [01_lie_typed_value_classes.md](accepted/01_lie_typed_value_classes.md) | `SE3.tensor` is the field. Existing `Motion.data` / `Force.data` / `Inertia.data` keep their name for now and pick up an alias; renaming is deferred to a separate cleanup. |
| **Cache invalidation cannot detect in-place tensor mutation** (`data.q[..., 0] += 1`). Only reassignment is caught. | [07_data_cache_invariants.md](accepted/07_data_cache_invariants.md) | An explicit "limitation" subsection (§7.E.0) documents this; an acceptance test asserts the limitation; user docs name reassignment as the supported pattern. |
| **D1 (centroidal) does not block D2 (RNEA)**. RNEA needs stable Lie/spatial conventions, not a finished centroidal map. | [14_dynamics_milestone_plan.md](accepted/14_dynamics_milestone_plan.md) | Sequence diagram updated; calendar shows D1 close-out lands in parallel with D2. |

### Adopted, no critique needed

| Idea | Where | Status |
|------|-------|--------|
| `KinematicsLevel` enum + `StaleCacheError` + `Data.require()` | [07_data_cache_invariants.md](accepted/07_data_cache_invariants.md) | This was already this folder's proposal; gpt_plan endorsed it almost wholesale. |
| `ReferenceFrame` enum replacing string literals | [04_typing_shapes_and_enums.md](accepted/04_typing_shapes_and_enums.md) | Same — already proposed; gpt_plan agreed. |
| `_typing.SE3` rename to `SE3Tensor` (+ companion aliases) | [04_typing_shapes_and_enums.md](accepted/04_typing_shapes_and_enums.md) | Already proposed; gpt_plan agreed and added a few more useful aliases. |
| Frozen FK regression oracle + Pinocchio cross-check | [12_regression_and_benchmarks.md](accepted/12_regression_and_benchmarks.md) | Already proposed; gpt_plan agreed. |
| Diátaxis docs site | [10_user_docs_diataxis.md](accepted/10_user_docs_diataxis.md) | Already proposed; gpt_plan agreed but suggested scheduling after the Lie/API decisions are stable. |
| Optional dep contract test | [15_packaging_extras_releases.md](accepted/15_packaging_extras_releases.md) | Already proposed; gpt_plan agreed and went *further* — see "adopted with changes" below. |
| One normative style guide | [13_style_reconciliation.md](accepted/13_style_reconciliation.md) | Already proposed; gpt_plan agreed. |

---

## Adopted with changes

### Top-level public API stays lean

| Earlier draft (this folder) | Adopted |
|-----------------------------|---------|
| Promote ~31 symbols into `__all__` (added `SE3`, `SO3`, `Pose`, `Motion`, `Force`, `Inertia`, `IKResult`, `IKCostConfig`, `OptimizerConfig`, `SolverState`, `ReferenceFrame`, `JointKind`, `ModelBuilder`). | **26 symbols** — only `SE3` and `ModelBuilder` promoted to top-level. Everything else stays one submodule down (`better_robot.lie.SO3`, `better_robot.spatial.Motion`, `better_robot.tasks.ik.IKResult`, …). A new contract test asserts submodule reachability. |

The principle: **promote on evidence, not aesthetics**. A symbol
earns top-level status when example code, tutorials, or user issues
demonstrate the qualified path is the friction. See
[Proposal 06 §6.A](accepted/06_public_api_audit.md).

### Backend abstraction: explicit objects, global is sugar

| Earlier draft | Adopted |
|---------------|---------|
| Process-global `current()` is the architectural core; every Lie / kinematics call routes through it. | Explicit `Backend` objects passed via `backend=` kwargs are the architectural core; `default_backend()` and `set_backend()` exist as **convenience sugar** but are documented as one-time configuration, not the runtime knob. Reason: process-global mutable state interacts badly with `torch.compile`, multi-backend test workflows, and nested libraries. |

A new helper, `get_backend(name)`, returns a backend by name without
mutating the default — the right primitive for tests. See
[Proposal 02](accepted/02_backend_abstraction.md).

### PyPose replacement: keep analytic & matrix-free paths first-class

| Earlier draft | Adopted |
|---------------|---------|
| Once Lie autograd is correct, "retire `kinematics.residual_jacobian`'s FD" and "`apply_jac_transpose` is no longer required". | Default flips to autodiff, FD becomes opt-in — but **analytic Jacobians and `apply_jac_transpose` stay first-class** for memory-efficient long-horizon trajopt. Autograd is the safety improvement; sparsity and matrix-free assembly are independently load-bearing. |

See the rewritten "What gets retired (and what stays first-class)"
section of [Proposal 03](accepted/03_replace_pypose.md). The new
[Proposal 16 §16.D](accepted/16_optim_wiring_and_matrix_free.md) is where
the matrix-free path is pinned as a typed contract.

### Trajectory: don't force `B = 1` for unbatched

| Earlier draft | Adopted |
|---------------|---------|
| `Trajectory.q` always has shape `(B, T, nq)`; an unbatched trajectory has `B = 1`. | Both `q.shape == (T, nq)` and `q.shape == (*B, T, nq)` are accepted. `traj.with_batch_dims(n)` normalises when an algorithm needs a concrete batch axis. Reason: BetterRobot's other public APIs already accept arbitrary leading shapes. |

See [Proposal 08 §8.A](accepted/08_trajectory_lock_in.md).

### Spatial value types: keep `Symmetric3` reachable, drop "fill `Force.cross_motion`"

| Earlier draft | Adopted |
|---------------|---------|
| Drop `Symmetric3` from `spatial/__init__.py`. Implement `Force.cross_motion` for symmetry with `Motion.cross_force`. | **Keep** `Symmetric3` in `spatial/__init__.py` (just don't promote to top-level). **Do not** implement `Force.cross_motion` — `Force × Motion` is not a standard spatial-algebra operation; the dual that exists is `Motion.cross_force`. The `NotImplementedError` message is upgraded to point at the correct dual and at a new §7.X duality paragraph in `03_LIE_AND_SPATIAL.md`. |

See [Proposal 05 §5.C, §5.D](accepted/05_value_types_audit.md). The
gpt-plan reviewer was right that "looks symmetric" is not a
sufficient reason to teach users a non-standard operation.

### CI gates: advisory-then-blocking ladder

| Earlier draft | Adopted |
|---------------|---------|
| Strict shape-annotation contract test on the public surface; CUDA benchmark gate fails PRs at >20% regression from day one. | Several gates start advisory and only flip to blocking after collecting variance data: shape-annotation coverage advisory until Stage 3 of the migration; CUDA bench nightly-only until one cycle of stable runner data; CPU bench advisory until two cycles of <5% noise floor. The contract bundle (correctness, layer DAG, hot-path lint, mypy strict) blocks from day one. |

See [Proposal 11 §11.F](accepted/11_quality_gates_ci.md) and
[Proposal 12 §12.C](accepted/12_regression_and_benchmarks.md). The
reviewer's point: hard gates *before* runner stability is measured
produce flaky CI that gets muted, defeating the gate's purpose.

### Optional deps: go further

| Earlier draft | Adopted |
|---------------|---------|
| Core deps include `yourdfpy` and `robot_descriptions`; `pin` is a dev dep. | Core deps drop **`yourdfpy`** (move to `[urdf]`), **`robot_descriptions`** (move to `[examples]`), and **`pin`** (move to `[test]`). A user typing `pip install better-robot` gets a small core; URDF support is `pip install better-robot[urdf]`. |

See [Proposal 15 §15.A](accepted/15_packaging_extras_releases.md). The
reviewer was right that `yourdfpy` and `robot_descriptions` did not
belong in the lean core install.

### `[human]` extra: declare seam, defer dependency

| Earlier draft | Adopted |
|---------------|---------|
| `pyproject.toml` declares a `[human]` extra pointing at `better_robot_human>=0.1`. | Phase 1: no `[human]` extra; document the seam, ship a placeholder test that constructs an SMPL-skeleton-topology IRModel via the programmatic builder. Phase 2 (when the package is published): add the extra. |

See [Proposal 09 §9.E](accepted/09_human_body_extension_lane.md). Adding a
real dependency on an unpublished package was the issue.

### Regression oracle metadata: drop git SHA

| Earlier draft | Adopted |
|---------------|---------|
| `fk_reference.npz` metadata includes `git_sha=...`. | Metadata records `oracle_version`, `generation_seed`, dependency versions, and `generated_at` — **no SHA**. The combination is enough to reproduce the oracle byte-for-byte; the SHA was decorative and would be useless after rebases. |

See [Proposal 12 §12.A](accepted/12_regression_and_benchmarks.md).

---

## Newly added — proposals shaped by gpt_plan ideas

The reviewer surfaced several concepts this folder did not cover
end-to-end. They were collected into two new proposals:

| New proposal | What it covers (gpt-plan-driven) |
|--------------|----------------------------------|
| [16 — Optimization wiring and matrix-free trajopt](accepted/16_optim_wiring_and_matrix_free.md) | Wire `OptimizerConfig` knobs (`linear_solver`, `kernel`, `damping`); separate active / weight / robust-kernel concepts; `ResidualSpec` for sparse / temporal structure; `LeastSquaresProblem.gradient(x)` as a typed matrix-free path; `TrajectoryParameterization` Protocol with `KnotTrajectory` / `BSplineTrajectory`; `MultiStageOptimizer` generalising `LMThenLBFGS`. |
| [17 — IO ergonomics and asset resolution](accepted/17_io_ergonomics_and_assets.md) | Fix the `kind="revolute_z"` vs `kind="revolute"` mismatch via named `add_revolute_z` / `add_free_flyer_root` builder helpers; `IRModel.schema_version`; `AssetResolver` Protocol with filesystem / package / cached-download implementations. |

`JointModel` dynamics hooks (`joint_bias_acceleration`,
`joint_motion_subspace_derivative`) — also a gpt-plan-flagged gap —
land in [Proposal 14](accepted/14_dynamics_milestone_plan.md) where they
belong (they are required before D2 RNEA is correct for non-trivial
joint kinds).

---

## Rejected (or held back)

A handful of gpt-plan suggestions were considered and not adopted,
either because the cost outweighs the benefit pre-1.0 or because
the ground truth turned out to be different. They are recorded
here so the next reviewer is not left wondering.

| gpt_plan suggestion | Why not adopted (or held back) |
|---------------------|-------------------------------|
| Introduce `RobotState` as a separate type from `Data` early. | Held back. The split is a real concern when trajopt expands, but doing it before [Proposal 16](accepted/16_optim_wiring_and_matrix_free.md) lands makes optimisers depend on a state surface we have not pinned. Revisit after 16.A–F land — likely a small follow-up that splits `Data` into `RobotState + AlgorithmCache`. Not blocking. |
| Promote `StateMultibody` "earlier than optimal-control features". | Already in [Proposal 14 D7](accepted/14_dynamics_milestone_plan.md). The gpt-plan ordering would land it before D2 RNEA; that is gold-plating — RNEA does not need a finished `StateMultibody`. Land it just before D7, as currently sequenced. |
| Per-`Model` backend selection. | Out of scope. The explicit `backend=` kwarg pattern from [Proposal 02](accepted/02_backend_abstraction.md) covers the use case; `Model.meta["backend"]` may *advise* a backend but does not bind one. A future proposal can promote this if real workloads show friction. |
| `[geometry]` extra separate from `[viewer]`. | Adopted partially: [Proposal 15](accepted/15_packaging_extras_releases.md) ships `[geometry]` and `[viewer]`, but they share `trimesh`. The reviewer's full split (geometry separate from viewer) is preserved in the names; the dep overlap is intentional and minor. |
| Add a `Conda-forge` channel before 1.0. | Out of scope. Recorded in [Proposal 15 §"Out of scope"](accepted/15_packaging_extras_releases.md). Conda comes after PyPI is stable. |
| Implement non-textbook spatial operations for symmetry (`Force.cross_motion`). | Rejected — see "Adopted with changes" above. |

---

## How to use this document

- When reviewing a proposal, read the gpt_plan counterpart if one
  exists. The two often disagree in detail; this document records
  which detail won.
- If a proposal here is later changed in a way that re-opens a
  question already settled by this reconciliation, update **this**
  file (or follow it with a v2 reconciliation log) so the trail
  stays visible.
- The reconciliation is non-normative — the proposals themselves
  are what get implemented. This file is just the audit trail.
