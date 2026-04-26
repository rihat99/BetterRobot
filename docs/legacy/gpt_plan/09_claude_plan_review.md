# Claude Plan Review

## Purpose

This file reconciles `docs/claude_plan` with the GPT plan. The Claude plan is high quality and often more operational than the first GPT pass: it names concrete tests, files, acceptance criteria, and CI gates. I would adopt many of those details. I would not adopt every direction as written.

The main difference in philosophy:

- Claude tends to make more things public and load-bearing immediately.
- I prefer the same long-term architecture, but with stricter boundaries around top-level API, backend global state, and hot-path object wrappers.

## Adopt Directly

### Data Cache Invariants

Adopt Claude proposal 07 almost wholesale.

Why:

- Stale `Data` caches are a real wrong-answer risk.
- `_kinematics_level` exists but is not yet a strong contract.
- A typed `StaleCacheError` is much better than crashing on `None` or silently using old FK output.

Add:

- `KinematicsLevel`.
- `StaleCacheError`.
- `Data.require(level)`.
- cache invalidation when assigning `q`, `v`, or `a`.
- typed accessors such as `data.joint_pose(...)` and `data.frame_pose(...)` that check cache presence.

Caveat:

- Assignment invalidation cannot detect in-place mutation of the tensor stored in `data.q`, such as `data.q[..., 0] += 1`. Do not overclaim. The docs should say BetterRobot detects reassignment and public API misuse, not arbitrary in-place tensor mutation.

### ReferenceFrame Enum

Adopt Claude proposal 04's `ReferenceFrame` enum.

Why:

- `reference="local_world_aligned"` is too easy to typo.
- A `str` enum gives nearly painless migration.
- This is a common robotics concept and deserves a named type.

Suggested policy:

- `ReferenceFrame.WORLD`
- `ReferenceFrame.LOCAL`
- `ReferenceFrame.LOCAL_WORLD_ALIGNED`

Keep accepting legacy strings for one pre-v1 migration window, then prefer enums in docs and examples.

### Shape Alias Cleanup

Adopt the renaming idea:

- `_typing.SO3` -> `SO3Tensor`
- `_typing.SE3` -> `SE3Tensor`
- add `ConfigTensor`, `VelocityTensor`, `JointPoseStack`, `FramePoseStack`, etc.

This avoids a future collision with real `SO3` and `SE3` classes.

I would adopt jaxtyping-style annotations for public API gradually, not as a single strict gate across the whole public surface on day one.

### Regression Oracles

Adopt Claude proposal 12's frozen FK oracle and Pinocchio cross-check direction.

Why:

- Lie and frame convention drift is subtle.
- Pure Torch Lie replacement needs a frozen numerical reference.
- Pinocchio parity is a strong confidence signal for users.

Adjustments:

- Keep FK oracle small and deterministic.
- Store generated metadata, but do not rely on a local git SHA as part of reproducibility.
- CUDA benchmark baselines should start advisory or nightly-only until runner stability is proven.

### Optional Dependency Hygiene

Adopt Claude proposal 15's optional import contract.

Core import should not pull:

- Warp,
- viewer packages,
- docs packages,
- MuJoCo,
- human-domain packages.

I would go further than Claude's sample extras and move `robot_descriptions` out of core dependencies too. It is useful for examples and tests, not for importing `better_robot`.

### User Docs Direction

Adopt the Diataxis idea from Claude proposal 10.

BetterRobot needs user docs, not only internal architecture specs:

- tutorials,
- how-to guides,
- concepts,
- generated API reference.

I would schedule this after the Lie/API decisions are made, otherwise the tutorials will churn immediately.

### Style Reconciliation

Adopt Claude proposal 13's core point: one normative style guide should replace two draft style files.

Adopt:

- PyTorch-first, not NumPy-first.
- scalar-last quaternion convention `[qx, qy, qz, qw]`.
- NumPy-style docstrings for public scientific APIs.
- dataclasses for configs.
- no Pydantic or custom config metaclass.
- no in-place `_into`/`out=` variants until benchmarks prove a need.

Do not force a large import-style churn just to satisfy a style guide. The current code uses relative imports heavily; that is acceptable inside a package.

## Adopt With Changes

### Lie Value Classes

Claude proposal 01 agrees with the main GPT plan: add thin value classes, not Tensor subclasses.

Adopt:

- frozen dataclasses,
- no `__torch_function__`,
- `@` for composition and point action,
- no `*`,
- no `~T`,
- storage remains raw tensors in `Model` and `Data`,
- typed wrappers appear at public boundaries.

Change:

- Put `SO3` and `SE3` in `better_robot.lie.types`, not primarily in `spatial/`.

Reason:

- `SO3` and `SE3` are Lie group types. Conceptually they belong in `lie`.
- `spatial` already depends on `lie`; putting Lie group classes in `spatial` makes the layering story less clear.
- `spatial` can still re-export them later for convenience if that proves useful.

Also prefer `.tensor` as the canonical field name, with `.data` only if matching existing spatial classes is judged more important.

### Backend Abstraction

Claude proposal 02 is right that `backends/` needs to become real before Warp lands. It is also right that direct `_pypose_backend` imports should disappear from the public functional Lie facade.

Adopt:

- backend protocols,
- capability checks,
- explicit tests for backend boundary,
- lazy optional backend imports,
- Torch backend as the reference implementation.

Change:

- Do not make a process-global `current()` the core execution model.
- Use explicit backend/config objects under the hood, with a global default only as a convenience.

Reason:

- Process-global backend state interacts badly with `torch.compile`, tests that compare two backends, nested libraries, and concurrent workloads.
- A global convenience is fine for users. It should not be the only architecture.

### PyPose Replacement

Adopt the goal from Claude proposal 03: remove PyPose from runtime Lie operations.

Important pushback:

- Do not copy the implementation sketch as-is.
- The SE3 exponential sketch appears to use `V = a * I + b * W + c * W^2`, where the standard left Jacobian form is `V = I + B * W + C * W^2`. That `a * I` term is a red flag.
- SE3 log/exp must be independently derived from a trusted reference and locked by gradcheck and finite differences.

Also change the post-PyPose promise:

- Do not retire analytic Jacobians just because Lie autograd becomes correct.
- Do not remove `apply_jac_transpose`; it is still valuable for memory-efficient long-horizon trajectory optimization.
- `loss.backward()` can become valid, but analytic and matrix-free paths should remain first-class for speed and sparsity.

### Public API Audit

Claude proposal 06 is right that the magic number 25 should not be sacred. The exact public surface should be audited by usefulness, not numerology.

Adopt:

- explicit expected public API test,
- per-symbol stability tiers,
- promote genuinely common result/config types when they become stable.

Push back:

- Do not immediately expand top-level `better_robot.__all__` to roughly 30-35 names.
- Do not export both `SE3` and `Pose` at top level in v1 unless user testing proves both names are worth the extra surface.
- Do not top-level export every value type just because it is ergonomic.

Preferred policy:

- Top-level API remains small.
- Rich submodule APIs are encouraged: `better_robot.lie.SE3`, `better_robot.spatial.Motion`, `better_robot.tasks.IKResult`.
- Promote to top-level only after examples show users repeatedly need it.

### Trajectory Shape

Claude proposal 08 is right to lock the trajectory contract before trajopt grows.

Adopt:

- `Trajectory` validates `t`, `q`, optional `v`, `a`, `tau`.
- time axis convention is `(B..., T, D)` or `(B, T, D)` depending on existing batching rules.
- `Trajectory.to(...)`, `slice`, and basic resampling should exist.
- manifold interpolation should be explicit for quaternion and SE3 blocks.

Push back:

- Do not require a single trajectory to be represented as `B = 1`.
- BetterRobot already supports arbitrary leading shapes and unbatched examples. Forcing `B=1` everywhere adds ceremony without clear benefit.

Preferred policy:

- Accept `q.shape == (T, nq)` for one trajectory.
- Accept `q.shape == (B..., T, nq)` for batched trajectories.
- Normalize internally where an algorithm needs a concrete batch axis.

### Human Body Lane

Claude proposal 09 is directionally right: human and biomechanical modeling should remain a domain layer, not core BetterRobot.

Adopt:

- anatomical joints use the `JointModel` extension seam,
- muscles are actuators, not joints,
- OpenSim/SMPL importers emit IR,
- no core imports of SMPL, OpenSim, or human-domain dependencies.

Change:

- Do not add a real `[human]` dependency on an unpublished external package yet.
- Document the seam and add placeholder tests/builders first.

## Push Back Or Reject

### Force.cross_motion

Claude proposal 05 suggests implementing `Force.cross_motion` for symmetry.

I would not adopt this until the operation's convention is derived and named unambiguously.

Reason:

- The current code explicitly raises `NotImplementedError` because `Force x Motion` is not a standard operation in the same way `Motion x Force` is.
- Adding a method because it looks symmetric risks teaching users a nonstandard spatial algebra operation.

Better direction:

- Keep `Motion.cross_force` as the dual action.
- Add a tested named operation only if a concrete dynamics algorithm needs it.
- Document the duality in `spatial.ops` rather than inventing symmetry.

### Dropping Symmetric3 From spatial.__init__

Claude proposal 05 says `from better_robot.spatial import Symmetric3` should fail or warn.

I would not spend migration cost on this now.

Reason:

- `Symmetric3` is small, coherent, and already part of the spatial value-type vocabulary.
- It should not be top-level `better_robot.Symmetric3`, but hiding it from `better_robot.spatial` is not worth a breaking API decision unless there is evidence of misuse.

Preferred:

- Keep it as a submodule/type for advanced users.
- Do not promote it to top-level.
- Mention it as an internal-helper-ish type in docs.

### Backend As One-File Warp Swap

Claude says Warp becomes a "one-file landing" after backend abstraction.

This is too optimistic.

Warp needs:

- tensor/array bridge,
- stream semantics,
- autograd wrappers,
- graph capture policy,
- kernel cache,
- dtype/device checks,
- fallback behavior,
- benchmark and numerical parity tests.

The backend protocol can make Warp isolated. It cannot make it one-file.

### Strict Shape Annotation Gate Immediately

Claude's shape-annotation contract is good, but enforcing every public tensor parameter immediately is too much churn.

Preferred:

- add aliases now,
- use annotations on new public APIs,
- migrate touched functions,
- add advisory coverage report first,
- later make it blocking.

### Hard CUDA Benchmark Gate Early

Benchmarks are necessary, but a hard CUDA gate before stable runners creates noise.

Preferred:

- unit and contract tests block every PR,
- CPU microbenchmarks can be advisory at first,
- CUDA benchmarks run nightly or on protected branches,
- hard budget gates come after variance is measured.

### D1 Must Block D2 Dynamics

Claude proposal 14 says D1 center-of-mass must close before D2 RNEA.

I disagree.

RNEA does not need centroidal map completion. Current code already has partial RNEA independent of centroidal work. The better dependency graph is:

- RNEA can proceed after Lie/spatial conventions are stable.
- CRBA and ABA follow RNEA.
- Centroidal can proceed in parallel once FK and inertia transforms are stable.
- Derivatives depend on the implemented dynamics algorithms.

## Updates To GPT Plan Direction

### Architecture

Keep the GPT plan's architecture:

- functional tensor kernels stay canonical for hot paths,
- value classes are thin wrappers,
- backend does not leak,
- `Model`/`Data` stay foundational.

Add from Claude:

- explicit acceptance criteria,
- cache invariants,
- enum cleanup,
- optional import tests,
- docs site plan,
- regression and benchmark artifacts.

### Migration Order

The revised first wave should be:

1. Normative style and contract cleanup.
2. `ReferenceFrame`, `KinematicsLevel`, `StaleCacheError`.
3. Pure Torch Lie backend.
4. `SO3` and `SE3` value classes.
5. Public API audit.
6. Optional dependency split.
7. Regression oracle and benchmark scaffolding.
8. State, trajectory, and optimization matrix-free paths.

### Tests Worth Adding First

- `tests/data_model/test_cache_invariants.py`
- `tests/contract/test_optional_imports.py`
- `tests/contract/test_backend_boundary.py`
- `tests/contract/test_reference_frame_enum.py`
- `tests/lie/test_torch_backend_gradcheck.py`
- `tests/lie/test_value_class_equivalence.py`
- `tests/kinematics/test_fk_regression.py`
- advisory `tests/bench/bench_lie.py`

## Bottom Line

Claude's best contributions are operational specificity: exact tests, acceptance criteria, and concrete file changes. The GPT plan should absorb that.

The ideas I would not carry forward unchanged are:

- putting Lie types primarily in `spatial`,
- using a global backend selector as the architectural core,
- assuming PyPose replacement makes analytic/matrix-free Jacobian paths obsolete,
- expanding top-level public API too aggressively,
- forcing `B=1` for single trajectories,
- implementing nonstandard spatial operations for symmetry,
- making CUDA benchmark gates hard before runners are stable.
