# Coding Style

> **Status:** normative. Single source of truth for in-tree code style.

Style fights are the most expensive distraction a library can afford.
Every minute spent debating tab width is a minute not spent writing
analytic Jacobians. We resolve the debate up-front by ceding the small
decisions to tooling — `ruff` for formatting and linting, `mypy` and
`pyright` for typing — and committing the larger decisions
(NumPy-style docstrings, scalar-last quaternions, `Protocol` over
ABC) to this document. When tooling and this document disagree, the
tooling wins; submit a PR to fix the doc.

The principle behind the choices is that boring, predictable code
beats clever code. Library users will read our source when something
goes wrong; they will not have time to decode metaclass tricks or
overloaded operators. So we forbid `__torch_function__` subclassing,
keep `lie/` purely functional, allow operator overloading only where
the semantics are unambiguous (`SE3 @ SE3` is composition; `Motion *
something` is forbidden because it is ambiguous between scale and
cross-product), and prefer dependency injection through `Protocol`s
over deep class hierarchies. The result is code that reads like math
when it expresses math, and like Python when it expresses control flow
— and never both at once.

## 1 · Guiding principles

1. **Boring, predictable code beats clever code.** Library users will
   read our source when something goes wrong. Optimise for that
   reader.
2. **Fail loud at boundaries.** Validate inputs at public API entry
   points; trust internal callers.
3. **Composition over inheritance.** Prefer `Protocol`s and dependency
   injection over deep class hierarchies.
4. **Explicit over implicit**, especially around units, frames, and
   tensor shapes.
5. **Consistency beats personal preference.** If a convention is
   established, follow it even if you would choose differently in a
   green field.

## 2 · Formatting and linting

`ruff` is the source of truth. Configure once; do not argue.

- Line length: **120 characters**.
- Indentation: **4 spaces**, never tabs.
- Quotes: **double quotes** for strings, single only inside
  double-quoted strings.
- Trailing commas in multi-line collections and call sites.
- Imports sorted by `ruff` (isort-compatible): stdlib → third-party →
  `better_robot.*` → relative-equivalent. Blank lines between groups.

Required `ruff` rule sets: `E`, `F`, `PLC`, `PLE`, `PLR`, `PLW`.

```bash
uv run ruff format .
uv run ruff check --fix .
uv run mypy src/better_robot/
uv run pytest
```

A commit-stage hook runs ``ruff format`` and ``ruff check``. The fast contract
subset is configured separately for ``pre-push``; neither hook runs until it is
installed:

```bash
uv run pre-commit install
uv run pre-commit install --hook-type pre-push
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
| Private (module / class-internal) | `_leading_underscore` | `_validate_joint_limits` |
| Really private (name-mangled) | `__double_underscore` | rare; usually a smell |

Naming guidance:

- Storage attributes follow **`<entity>_<quantity>_<frame>`** —
  `joint_pose_world`, `frame_velocity_local`. See {doc}`naming`.
- Functions named after published algorithms keep their canonical
  acronym (`rnea`, `aba`, `crba`). Functions named by what they
  compute use full English (`forward_kinematics`).
- **No unit suffixes** in identifiers (`_m`, `_rad`). Reasoning: the
  library is SI-only on the public surface
  ({doc}`contracts`); the
  `<entity>_<quantity>_<frame>` convention already encodes the
  semantic disambiguation.
- Booleans: `is_`, `has_`, `should_` prefixes.
- Avoid single-letter variables outside math-heavy internals where
  they match the literature (`q`, `v`, `a`, `tau`, `T`, `R`).

## 4 · Quaternions, Lie storage, and frames

Storage layouts are fixed library-wide
({doc}`contracts`):

| Object | Layout |
|--------|--------|
| SE3 pose | `(..., 7) [tx, ty, tz, qx, qy, qz, qw]` |
| SO3 quat | `(..., 4) [qx, qy, qz, qw]` |
| se3 tangent | `(..., 6) [vx, vy, vz, wx, wy, wz]` (linear first) |
| so3 tangent | `(..., 3) [wx, wy, wz]` |
| Spatial Jacobian | `(..., 6, nv) [linear rows; angular rows]` |

Quaternions are scalar-last (Hamilton; `qw` last). It is **not** the
`[w, x, y, z]` order from any "standard" you may have seen elsewhere.
SI units everywhere on the public surface — radians, metres, seconds,
kilograms, newtons. Convert at the boundary if user-facing tooling
needs degrees; never internally.

## 5 · Imports

- **Absolute imports** inside the package
  (`from better_robot.lie import se3`, not `from ..lie import se3`).
  Exception: short relative imports (`from .se3 import compose`) are
  fine inside a single sub-package.
- **No wildcard imports** anywhere.
- **Lazy imports** for heavy optional deps — `mujoco`, `viser`,
  `warp`, `yourdfpy`, `chumpy`. Place the import inside the function
  body or behind a `TYPE_CHECKING` guard. Document why.
- Group order (`ruff` / isort): stdlib → third-party →
  `better_robot.*` → local relative.

## 6 · Type annotations

Public signatures should carry ordinary Python type annotations, with tensor
shapes stated in docstrings (see {doc}`naming` §2.9). Internal-helper
annotations are encouraged. Current tooling is transitional: ``mypy`` runs
with ``strict = false`` in ``pyproject.toml``, and no automatic mypy/pyright
workflow job is configured.

- Use modern syntax: `list[int]`, `dict[str, float]`, `X | None`.
  Min Python is 3.10.
- Use `typing.Protocol` for structural interfaces (residuals, joints,
  optimisers, sensors). Reserve `abc.ABC` for cases needing shared
  implementation.
- Use `typing.Self` for fluent methods returning the same type.
- Use `TypeAlias` (or `type` statement on 3.12+) for non-trivial type
  names that appear in many places — see `_typing.py`.
- Run the configured ``mypy`` and ``pyright`` checks on changed typed surfaces.
  Do not describe them as strict or CI-blocking until configuration and the
  workflow actually enforce that. New ``# type: ignore`` comments require a
  reason.

```python
from typing import Protocol
from torch import Tensor
from jaxtyping import Float

class Optimizer(Protocol):
    def minimize(
        self,
        problem: "LeastSquaresProblem",
        *,
        max_iter: int,
        linear_solver,
        kernel,
        strategy,
        scheduler=None,
    ) -> "SolverState": ...

def forward_kinematics(
    model: "Model",
    q_or_data: Float[Tensor, "*B nq"] | "Data",
    *,
    compute_frames: bool = False,
    check_quaternion_norm: bool = False,
) -> "Data": ...
```

## 7 · Docstrings — NumPy style

We pick **NumPy** for these reasons:

- Sphinx + Napoleon supports both Google and NumPy; the
  scientific-Python ecosystem (numpy, scipy, scikit-learn, jax) is
  NumPy-style. New users coming from those projects recognise the
  pattern.
- NumPy-style is friendlier to multi-paragraph parameter docs.
- The `Shapes` block fits naturally as a sub-section.

```python
def forward_kinematics(
    model: Model,
    q_or_data: ConfigTensor | Data,
    *,
    compute_frames: bool = False,
    check_quaternion_norm: bool = False,
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
    check_quaternion_norm : bool, default False
        Opt-in free-flyer norm diagnostic. It synchronizes accelerator tensors;
        the default hot path assumes pre-normalized quaternions.

    Returns
    -------
    Data
        With ``joint_pose_world``, ``joint_pose_local``, and (if
        ``compute_frames``) ``frame_pose_world`` populated.
        Advances ``data._kinematics_level`` to
        ``KinematicsLevel.PLACEMENTS``.

    Raises
    ------
    QuaternionNormError
        If ``check_quaternion_norm=True`` and a free-flyer base quaternion has
        norm outside ``[0.9, 1.1]``.

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
    >>> from robot_descriptions import panda_description
    >>> model = better_robot.load(panda_description.URDF_PATH)
    >>> q = model.q_neutral.unsqueeze(0)
    >>> data = better_robot.forward_kinematics(model, q, compute_frames=True)
    """
```

Conventions inside docstrings:

- **Always state units** (SI).
- **Always state shape** for tensor parameters: `(B..., nq)`,
  `(B..., 6, nv)`.
- **State frame** for any pose, twist, wrench, or velocity:
  "expressed in the world frame".
- Use double backticks for code (`` ``q`` ``).
- One-line summary first, blank line, then details.

## 8 · Configs and dataclasses

- Match mutability to the current API. ``Model`` and structural specs use
  frozen dataclasses; task configs such as ``IKCostConfig`` and
  ``OptimizerConfig`` are currently plain mutable dataclasses.
- Validate at construction or the owning public boundary. Existing code uses
  typed BetterRobot exceptions where wired and direct ``TypeError`` /
  ``ValueError`` elsewhere; new code should use the most specific established
  exception.
- **No Pydantic; no custom metaclass.** Plain dataclasses and explicit
  validation keep configuration inspectable.

## 9 · Errors

- Subclass `BetterRobotError` (in `better_robot/exceptions.py`).
- Raise the most specific exception possible — see the taxonomy in
  {doc}`contracts`.
- Error messages name the offending input and the expected shape /
  type / value. Example: `"q has trailing size 6, expected 7
  (Panda)"`.
- Bare ``except:`` is forbidden. A narrow ``except Exception`` is permitted at
  an optional-runtime/parser or cleanup boundary only when it immediately
  converts, annotates, or safely cleans up the failure; it must not hide an
  algorithm error.

## 10 · Logging

BetterRobot currently has no package logging surface. Library code should not
print unsolicited diagnostics. If a module adds logging, use
``logging.getLogger(__name__)`` without configuring application handlers, and
use ``%`` placeholders for deferred formatting:

  ```python
  logger.debug("iter %d residual %.3e", i, r)
  ```

## 11 · File organisation

- **One concept per file.** Not `utils.py`. Not `helpers.py`.
- Tests live in `tests/<package>/test_<concept>.py`.
- A package's `__init__.py` re-exports its public symbols and lists
  them in `__all__`.

## 12 · Mutability

- `Model`: shallowly frozen (`@dataclass(frozen=True)`); contained tensors and
  mappings still require read-only discipline.
- `Data`: mutable and evaluation-local; cache levels are advanced/invalidated
  by the owning operations.
- `IKResult`, `Trajectory`: plain mutable dataclasses. ``Trajectory.slice``,
  ``resample``, and ``with_batch_dims`` return new instances, but that does not
  freeze existing fields or tensors.
- `CostStack`: mutable through `.add`, `.remove`, `.set_weight`, and
  `.set_active`; its public `items` mapping also exposes mutable `CostItem`
  values, so keep each stack problem-local.

## 13 · Numerics

- **Vectorise** with `torch` ops. Read the published numerical
  formula, write the readable batched version, profile, optimise.
- **Numerical tolerances are explicit**: never compare floats with
  `==`. Use `torch.testing.assert_close(..., atol=..., rtol=...)` for Euclidean
  values. For SE(3), compare translation normally and rotation through
  `so3.to_matrix`, or compare a relative-log tangent norm; quaternion component
  equality is not sign-invariant.
- **Allocating vs. in-place**: direct math functions normally return new
  tensors. ``Data`` cache mutation and the private ``GraphExecutor.reset``
  lifecycle are explicit exceptions; legacy ``SolverState`` has no reset
  method.
- **Avoid mutable default arguments** — use `None` and construct
  inside.
- **Pure raw math, explicit workspace mutation**: ``lie/`` and raw
  kinematics/dynamics passes are functional; public wrappers may populate a
  caller-provided mutable ``Data`` workspace.

## 14 · Hot-path discipline

The executable rule list and exact watched paths live in
``tests/contract/test_hot_path_lint.py`` and are summarized in
{doc}`performance`. Today they cover ``.item()``/``.cpu()``, proven tensor
scalar conversions, ``.new_tensor()``, selected allocations inside Python
loops, and ``if x.dim()`` rank branches. The test does not implement a general
ban on every tensor conditional, ``.numpy()``, or all allocations.

A legitimate eager/static boundary may use ``# bench-ok: <reason>``. There is
no numeric exemption-count gate or automatic pull-request workflow.

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
  singletons and process-wide compute selectors are not.
- `eval`, `exec` — never.

## 16 · Pull requests and reviews

- **One logical change per PR.** Many small PRs over one large.
- PR description states what changed and why, links related issues,
  and notes any breaking changes.
- Relevant local or manually dispatched checks are green before review:
  format, lint, type-check, tests, and docs build. The repository currently
  has no automatic pull-request workflow.
- New public API requires: tests, NumPy-style docstring with at least
  one example, changelog entry, and (if non-trivial) a doc page or
  notebook.
- Be kind in review. Critique the code, not the author. Suggest
  concrete alternatives.

## 17 · Quick checklist for a new public function

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
- [ ] Listed in the relevant `__all__`; if top-level, the required-core
      contract was updated when applicable.
- [ ] Mentioned in `CHANGELOG.md` under "Added".

---

*Cross-references:*

- {doc}`naming` — naming policy and rename table.
- {doc}`performance` — perf budgets and lint rules.
- {doc}`testing` — test strategy, local commands, and current automation.
- {doc}`contracts` — input contracts and exception taxonomy.
- {doc}`packaging` — extras, releases, deprecation.
