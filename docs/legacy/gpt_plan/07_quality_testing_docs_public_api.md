# Quality, Testing, Docs, And Public API

## Current State

The project already has valuable quality infrastructure:

- public API contract tests,
- dependency DAG tests,
- PyPose import isolation test,
- deprecated naming contract tests,
- Pinocchio comparison tests,
- skeleton signature tests,
- style guidance documents.

The next step is to turn these into a coherent release-quality policy.

## Public API Policy

The 25-symbol top-level API ceiling is useful. Keep the ceiling, but do not treat the exact current list as sacred before v1.

Recommended:

- `better_robot.__all__` remains small.
- `better_robot.lie`, `better_robot.spatial`, `better_robot.io`, etc. expose richer submodule APIs.
- New `SO3` and `SE3` can initially live under `better_robot.lie`.
- Decide before v1 whether `SO3` and `SE3` deserve top-level names.

The current tests can be updated when docs intentionally change the contract.

## Docstring Style

The style docs disagree between Google-style and numpydoc-style recommendations. Pick one canonical style.

Recommended for BetterRobot:

- Public numerical APIs use numpydoc-style sections: `Parameters`, `Returns`, `Raises`, `Notes`.
- Small internal helpers use one-line docstrings or short paragraphs.
- Shapes, frames, units, dtype/device behavior, and differentiability are mandatory for public functions.
- Avoid long historical explanations in code docstrings; put those in design docs.

This matches scientific Python expectations and the current code's direction.

## Typing

Add or tighten:

- `py.typed` in the package.
- Pyright or mypy config in `pyproject.toml`.
- Tensor shape aliases with names that do not conflict with future classes:
  - `SO3Tensor`,
  - `SE3Tensor`,
  - `QTensor`,
  - `VTensor`.
- Protocols for extension points.

When `SO3` and `SE3` classes land, rename the current `_typing.SE3 = torch.Tensor` aliases.

## Linting

Current Ruff config is minimal. Add gradually:

- import sorting,
- unused arguments policy,
- bugbear-like rules if available,
- stricter naming for public API,
- no print statements in library code,
- no optional dependency imports outside approved files,
- no direct `.item()` or `float(tensor)` in batched solver paths unless documented.

Keep line length flexible for math-heavy code, but avoid unreadable expressions.

## Testing Strategy

Use the existing docs as categories:

- unit tests for primitives,
- integration tests for workflows,
- contract tests for public API and layering,
- oracle tests against Pinocchio and PyPose references where appropriate,
- gradient tests,
- property tests for Lie groups and manifolds,
- benchmark tests for hot paths.

## Tests To Add Soon

### Lie

- class/function equivalence,
- pure Torch vs PyPose/Pinocchio value checks,
- ambient gradients vs finite differences,
- tangent Jacobians vs finite differences,
- invalid shapes and bad quaternion norms.

### Backend

- no Warp import unless backend requested,
- capability fallback,
- Torch compile compatibility,
- CPU/CUDA parity,
- dtype rejection.

### Model/Data

- model topology invariants,
- data cache reset and stale-cache behavior,
- `Model.to` moves every tensor,
- `ModelTensors` mirrors `Model`.

### Optimization

- batched solver policy,
- robust kernel effect,
- matrix-free gradient correctness,
- line search and damping state transitions,
- `solve_ik` config wiring.

### Dynamics

- RNEA, CRBA, ABA against Pinocchio,
- inverse/forward dynamics consistency,
- derivative finite differences.

### IO/Collision/Viewer

- optional dependency import hygiene,
- builder docs execute,
- parser IR invariants,
- SDF analytic cases,
- mock viewer backend logs expected calls.

## Benchmarks

Add a benchmark suite, not just performance assertions in unit tests.

Benchmark groups:

- SO3/SE3 exp/log/compose,
- FK for Panda and humanoid-sized models,
- Jacobians for same,
- RNEA/CRBA/ABA,
- IK solve,
- trajectory residual gradient,
- collision residual.

Record:

- device,
- dtype,
- batch shape,
- horizon,
- model size,
- backend,
- commit hash.

Do not fail every PR on strict microsecond thresholds at first. Start by collecting numbers, then enforce budgets once stable.

## Packaging

Recommended changes:

- Add `py.typed`.
- Move optional dependencies into extras.
- Add `test`, `dev`, `viewer`, `urdf`, `mjcf`, `geometry`, `warp` extras.
- Add import-time tests for core package.
- Add version policy before v1: breaking changes allowed with changelog entries.

## Docs Source Of Truth

The docs should remain authoritative, but they must not freeze bad early choices.

Recommended process:

1. Change the design doc first for architectural decisions.
2. Add migration notes when breaking code.
3. Add contract tests for decisions that should stay stable.
4. Keep status docs for known gaps and temporary workarounds.

For the Lie refactor, update:

- `docs/design/03_LIE_AND_SPATIAL.md`,
- `docs/status/PYPOSE_ISSUES.md`,
- `docs/conventions/17_CONTRACTS.md`,
- public API tests if `SO3` or `SE3` become public.

## Error Handling

Keep the exception hierarchy. Add:

- `BackendCapabilityError`,
- `UnsupportedDtypeError` or use current `DtypeMismatchError` consistently,
- `InvalidLieValueError` or reuse `QuaternionNormError` plus `ShapeError`.

Error messages should include:

- argument name,
- observed shape/device/dtype,
- expected contract,
- remediation when obvious.

## Deprecation Policy

Before v1, breaking changes are allowed but should still be documented. After v1:

- one minor release deprecation window for public APIs,
- deprecation warnings with target removal version,
- migration examples,
- no deprecation requirement for private modules.
