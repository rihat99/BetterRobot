# Coding Style

> **Status:** normative. Single source of truth for in-tree code style.
> The drafts under `docs/style/` are archived; this document supersedes them.

This is the operational form of the principles in
[00_VISION.md](../design/00_VISION.md), the naming policy in
[13_NAMING.md](13_NAMING.md), and the contracts in
[17_CONTRACTS.md](17_CONTRACTS.md). When tooling (`ruff`, `mypy`,
`pyright`) and this document disagree, the tooling wins — open a PR to
fix the doc.

## 1 · Guiding principles

1. **Boring, predictable code beats clever code.** Library users will
   read our source when something goes wrong. Optimise for that reader.
2. **Fail loud at boundaries.** Validate inputs at public API entry
   points; trust internal callers.
3. **Composition over inheritance.** Prefer `Protocol`s and dependency
   injection over deep class hierarchies.
4. **Explicit over implicit**, especially around units, frames, and
   tensor shapes.
5. **Consistency beats personal preference.** If a convention is
   established, follow it even if you'd choose differently in a green
   field.

## 2 · Formatting and linting

`ruff` is the source of truth. Configure once; do not argue.

- Line length: **100 characters**.
- Indentation: **4 spaces**, never tabs.
- Quotes: **double quotes** for strings, single only inside double-quoted strings.
- Trailing commas in multi-line collections and call sites.
- Imports sorted by `ruff` (isort-compatible): stdlib → third-party →
  `better_robot.*` → relative-equivalent. Blank lines between groups.

Required `ruff` rule sets (see `pyproject.toml`):
`E`, `F`, `W`, `I`, `N`, `UP`, `B`, `RUF`, `SIM`, `PTH`, `NPY`, `PLR0913`.

```bash
uv run ruff format .
uv run ruff check --fix .
uv run mypy src/better_robot/ --strict
uv run pytest
```

A `pre-commit` hook runs `ruff format`, `ruff check`, and the fast
contract subset on every commit. Install once:

```bash
uv run pre-commit install
```

## 3 · Naming

PEP 8, no exceptions. The full table:

| Kind | Convention | Example |
|---|---|---|
| Modules and packages | `lower_snake_case` | `kinematics`, `joint_models` |
| Classes, exceptions, type aliases | `CapWords` | `Model`, `IKResult`, `SE3Tensor` |
| Functions, methods, variables | `lower_snake_case` | `forward_kinematics`, `q`, `q_dot` |
| Constants | `UPPER_SNAKE_CASE` | `EPS`, `MAX_ITERATIONS` |
| Type variables | `CapWords`, short | `T`, `StateT` |
| Private (module/class-internal) | `_leading_underscore` | `_validate_joint_limits` |
| "Really private" (name-mangled) | `__double_underscore` | rare; usually a smell |

Naming guidance:

- Storage attributes follow **`<entity>_<quantity>_<frame>`** —
  `joint_pose_world`, `frame_velocity_local`. See
  [13 §1.1](13_NAMING.md).
- Functions named after published algorithms keep their canonical
  acronym (`rnea`, `aba`, `crba`). Functions named by what they
  compute use full English (`forward_kinematics`). See [13 §1.2](13_NAMING.md).
- **No unit suffixes** in identifiers (`_m`, `_rad`). Reasoning:
  - The library is SI-only on the public surface (
    [17 §1](17_CONTRACTS.md)).
  - The `<entity>_<quantity>_<frame>` convention already encodes the
    semantic disambiguation.
  - `joint_velocity_world` and `joint_velocity_world_radps` are
    redundant.
- Booleans: `is_`, `has_`, `should_` prefixes.
- Avoid single-letter variables outside of math-heavy internals where
  they match the literature (`q`, `v`, `a`, `tau`, `T`, `R`).

## 4 · Quaternions, Lie storage, and frames

Storage layouts are **fixed library-wide** (
[17 §1.3](17_CONTRACTS.md), CLAUDE.md):

| Object | Layout |
|--------|--------|
| SE3 pose | `(..., 7) [tx, ty, tz, qx, qy, qz, qw]` |
| SO3 quat | `(..., 4) [qx, qy, qz, qw]` |
| se3 tangent | `(..., 6) [vx, vy, vz, wx, wy, wz]` (linear first) |
| so3 tangent | `(..., 3) [wx, wy, wz]` |
| Spatial Jacobian | `(..., 6, nv) [linear rows; angular rows]` |

Quaternions are **scalar-last** (Hamilton; `qw` last). It is **not**
the `[w, x, y, z]` order from any "standard" you may have seen
elsewhere. This matches PyPose's native layout and the existing code.
SI units everywhere on the public surface — radians, metres, seconds,
kilograms, newtons. Convert at the boundary if user-facing tooling
needs degrees; never internally.

## 5 · Imports

- **Absolute imports** inside the package
  (`from better_robot.lie import se3`, not `from ..lie import se3`).
  Exception: short relative imports (`from .se3 import compose`) are
  fine inside a single sub-package.
- **No wildcard imports** anywhere. `ruff F405` enforces.
- **Lazy imports** for heavy optional deps — `mujoco`, `viser`,
  `warp`, `yourdfpy`, `chumpy`. Place the import inside the function
  body or behind a `TYPE_CHECKING` guard. Document why.
- Group order (`ruff`/isort): stdlib → third-party →
  `better_robot.*` → local relative.

## 6 · Type annotations

Public surface: full type annotations including `jaxtyping`-style
shape annotations (
[13 §2.9](13_NAMING.md)). Internal helpers: annotations encouraged,
not enforced.

- Use modern syntax: `list[int]`, `dict[str, float]`, `X | None`.
  Min Python is 3.10.
- Use `typing.Protocol` for structural interfaces (residuals, joints,
  optimisers, sensors). Reserve `abc.ABC` for cases needing shared
  implementation.
- Use `typing.Self` for fluent methods returning the same type.
- Use `TypeAlias` (or `type` statement on 3.12+) for non-trivial type
  names that appear in many places — see `_typing.py`.
- `mypy --strict` (or `pyright` in strict mode) must pass on
  `src/better_robot/`. New `# type: ignore` comments require a comment
  explaining why and ideally a linked issue.

```python
from typing import Protocol
from torch import Tensor
from jaxtyping import Float

class Optimizer(Protocol):
    def minimize(self, problem: "LeastSquaresProblem") -> "SolverState": ...

def forward_kinematics(
    model: "Model",
    q_or_data: Float[Tensor, "*B nq"] | "Data",
    *,
    compute_frames: bool = False,
) -> "Data": ...
```

## 7 · Docstrings — NumPy style

We pick **NumPy** for these reasons:

- Sphinx + Napoleon supports both Google and NumPy; the
  scientific-Python ecosystem (numpy, scipy, scikit-learn, jax) is
  NumPy-style. New users coming from those projects recognise the pattern.
- NumPy-style is friendlier to multi-paragraph parameter docs.
- The `Shapes` block fits naturally as a sub-section.

```python
def forward_kinematics(
    model: Model,
    q_or_data: ConfigTensor | Data,
    *,
    compute_frames: bool = False,
) -> Data:
    """Compute the placements of every joint, batched.

    Parameters
    ----------
    model : Model
        Frozen kinematic tree.
    q_or_data : Tensor (B..., nq) or Data
        Configuration. Passing a ``Data`` reuses its allocated buffers.
    compute_frames : bool, default False
        If True, also fill ``data.frame_pose_world``.

    Returns
    -------
    Data
        With ``joint_pose_world``, ``joint_pose_local``, and (if
        ``compute_frames``) ``frame_pose_world`` populated.
        Advances ``data._kinematics_level`` to ``KinematicsLevel.PLACEMENTS``.

    Raises
    ------
    QuaternionNormError
        If a free-flyer base quaternion has norm outside ``[0.9, 1.1]``.

    Notes
    -----
    The topology iteration is over ``model.topo_order`` (a Python
    tuple). It compiles cleanly under ``torch.compile``.

    See Also
    --------
    update_frame_placements : compute frame placements after an FK call
    compute_joint_jacobians : the next step toward Jacobian-based costs

    Examples
    --------
    >>> model = better_robot.load("panda.urdf")
    >>> q = model.q_neutral.unsqueeze(0)
    >>> data = better_robot.forward_kinematics(model, q, compute_frames=True)
    >>> data.frame_pose("panda_hand")
    SE3(tensor=tensor([...]))
    """
```

Conventions inside docstrings:

- **Always state units** (SI).
- **Always state shape** for tensor parameters: `(B..., nq)`, `(B..., 6, nv)`.
- **State frame** for any pose, twist, wrench, or velocity:
  "expressed in the world frame".
- Use double backticks for code (`` ``q`` ``).
- One-line summary first, blank line, then details.
- The `>>>` lines run under doctest and under MyST-NB in tutorial
  notebooks (per [10_user_docs_diataxis](../legacy/claude_plan/accepted/10_user_docs_diataxis.md)).

## 8 · Configs and dataclasses

- `@dataclass(frozen=True)` for all configs (`IKCostConfig`,
  `OptimizerConfig`, `OptimizerStage`, `MultiStageOptimizerConfig`).
- Validation in `__post_init__`. Raise `BetterRobotError` subclasses,
  not bare `ValueError`.
- **No Pydantic; no custom metaclass.** Per
  [00 §Non-goals](../design/00_VISION.md). Plain `@dataclass` +
  `__post_init__` is the discipline.

## 9 · Errors

- Subclass `BetterRobotError` (in `better_robot/exceptions.py`).
- Raise the most specific exception possible — see the taxonomy in
  [17 §2](17_CONTRACTS.md).
- Error messages name the offending input *and* the expected shape /
  type / value. Example: `"q has trailing size 6, expected 7 (Panda)"`.
- Bare `except:` and `except Exception:` are forbidden in `src/`
  (lint rule).

## 10 · Logging

- `logger = logging.getLogger(__name__)` per module.
- `print()` is forbidden in `src/`. Allowed in `examples/` and CLI tools.
- The library configures `logging.NullHandler` on `better_robot`
  (already done in `utils/logging.py`).
- Format strings use `%` placeholders for deferred formatting:

  ```python
  logger.debug("iter %d residual %.3e", i, r)
  ```

## 11 · File organisation

- **One concept per file.** Not `utils.py`. Not `helpers.py`.
- Tests live in `tests/<package>/test_<concept>.py`.
- A package's `__init__.py` re-exports its public symbols and lists
  them in `__all__`.

## 12 · Mutability

- `Model`: **immutable** (`@dataclass(frozen=True)`).
- `Data`: **mutable**, but per-thread; `__setattr__` invalidates caches
  (see [02_DATA_MODEL.md §3.1](../design/02_DATA_MODEL.md)).
- `IKResult`, `Trajectory`: frozen dataclasses; modifications return new
  instances (`slice`, `resample`, `with_batch_dims`).
- `CostStack`: controlled mutation via `.add`, `.set_weight`,
  `.set_active`. No direct field assignment.

## 13 · Numerics

- **Vectorise** with `torch` ops. Read the published numerical formula,
  write the readable batched version, profile, optimise.
- **Numerical tolerances are explicit**: never compare floats with `==`.
  Use `torch.allclose(..., atol=..., rtol=...)` or
  `assert_close_manifold(...)` for SE(3) values.
- **Allocating vs. in-place**: by default, functions allocate and return
  new tensors. The hot-path `Data.reset()` and `SolverState` reset are
  the in-place exceptions; they live in pre-allocated buffers.
- **Avoid mutable default arguments** — use `None` and construct inside.
- **Pure functions in math** (`lie/`, `spatial/`, `kinematics/`,
  `dynamics/`); side effects belong at the I/O and viewer layers.

## 14 · Hot-path discipline

The lint rules in `tests/contract/test_hot_path_lint.py` (see
[14 §3](14_PERFORMANCE.md)) forbid:

- Branching on `tensor.dim()` — rely on the leading-batch convention.
- `.item()` / `.cpu()` / `.numpy()` in compiled regions.
- `torch.zeros` / `torch.empty` / `torch.ones` inside a `for` loop body.
- Python `if` on tensor values (use `torch.where` or assert as contract).

Suppress with `# bench-ok: <reason>`. A PR adding more than three new
`bench-ok` comments fails CI.

## 15 · Language features

**Use freely:**
- f-strings (except in `logging` calls).
- Walrus operator `:=` where it improves clarity.
- `match` statements for genuine pattern matching.
- Dataclasses, `Protocol`, `Self`, `|` union syntax.
- Context managers for any resource that needs cleanup.

**Use sparingly, justify in review:**
- Decorators that change function signatures.
- `__getattr__` / `__getattribute__` magic.
- Metaclasses (almost never needed).
- Monkey-patching anything outside the package.

**Avoid:**
- Mutable default arguments.
- Bare `except:` clauses.
- Module-level side effects (besides defining names).
- Global mutable state. Registries are acceptable; configuration
  singletons are not — except for `default_backend()`, which is documented
  as one-time configuration in [10_BATCHING_AND_BACKENDS.md §7](../design/10_BATCHING_AND_BACKENDS.md).
- `eval`, `exec` — never.

## 16 · Pull requests and reviews

- **One logical change per PR.** Many small PRs over one large.
- PR description states **what** changed and **why**, links related
  issues, and notes any breaking changes.
- All checks green before review: format, lint, type-check, tests,
  docs build.
- New public API requires: tests, NumPy-style docstring with at least
  one example, changelog entry, and (if non-trivial) a doc page or
  notebook in `docs/site/`.
- Be kind in review. Critique the code, not the author. Suggest
  concrete alternatives.

## 17 · The skill files

The workspace has skill files (`python-standards`, `cpp-standards`,
`file-naming`, `design-principles`, `write-tests`, `diataxis-docs`,
`sphinx-docs`). Those are **operational** — short pragmatic notes for
Claude Code agents and external contributors. **This document is the
canonical reference**; the skills cite it. When a skill drifts from this
doc, the doc wins; update the skill in the same PR.

## 18 · Quick checklist for a new public function

Before merging:

- [ ] Type-annotated, including return type and shape annotations
      via `_typing.py`.
- [ ] NumPy-style docstring with `Parameters`, `Returns`, `Raises`,
      and at least one `Examples` block.
- [ ] Shapes, units, and frames documented for every tensor.
- [ ] Validates inputs at the boundary; raises a specific subclass of
      `BetterRobotError`.
- [ ] Has tests, including at least one property-based test if a
      mathematical invariant exists.
- [ ] Listed in the relevant `__all__`; if top-level, the
      `EXPECTED` set in `tests/contract/test_public_api.py` was
      updated in the same PR.
- [ ] Mentioned in `CHANGELOG.md` under "Added".

---

*Cross-references:*
[13_NAMING.md](13_NAMING.md) — naming policy and rename table.
[14_PERFORMANCE.md](14_PERFORMANCE.md) — perf budgets and lint rules.
[16_TESTING.md](16_TESTING.md) — test strategy and coverage budgets.
[17_CONTRACTS.md](17_CONTRACTS.md) — input contracts and exception taxonomy.
[20_PACKAGING.md](20_PACKAGING.md) — extras, releases, deprecation.
