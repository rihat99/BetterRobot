# STYLE.md (ARCHIVED)

> ⚠️ **ARCHIVED — do not follow this draft.**
> The normative coding-style guide is
> [conventions/19_STYLE.md](../conventions/19_STYLE.md). This file is
> kept for historical reference only and contradicts the current spec
> in several places (notably: `[w, x, y, z]` quaternion ordering,
> NumPy-typed APIs, Google-style docstrings). The library uses
> `[qx, qy, qz, qw]` (PyPose-native), `torch.Tensor` with `jaxtyping`
> annotations, and NumPy-style docstrings. See the reconciliation
> table in
> [13 §13.D Style reconciliation](../claude_plan/accepted/13_style_reconciliation.md)
> (once moved into accepted/) or
> [19_STYLE.md §4 Quaternions, Lie storage, and frames](../conventions/19_STYLE.md).

# STYLE.md

Style and conventions for the **Ultimate Robotics Toolbox**.

This document is the source of truth for how code in this project is written. It exists so contributors don't have to guess, and so reviewers have something concrete to point at. Mechanical rules are enforced by tooling (`ruff`, `mypy`/`pyright`); this document covers the conventions tooling can't enforce and the *reasoning* behind the choices.

When this guide and a linter disagree, the linter wins — open a PR to fix the guide.

---

## 1. Guiding principles

1. **Boring, predictable code beats clever code.** Library users will read our source when something goes wrong. Optimize for that reader.
2. **Fail loud at boundaries.** Validate inputs at public API entry points; trust internal callers.
3. **Composition over inheritance.** Prefer protocols and dependency injection over deep class hierarchies.
4. **Explicit over implicit**, especially around units, frames, and array shapes.
5. **Consistency beats personal preference.** If a convention is established, follow it even if you'd choose differently in a green field.

---

## 2. Formatting and linting

All formatting and basic style enforcement is handled by **`ruff`**. Do not argue with it; configure it.

- Line length: **100 characters**.
- Indentation: **4 spaces**, never tabs.
- Quotes: **double quotes** for strings, single quotes only inside double-quoted strings.
- Trailing commas in multi-line collections and call sites.
- Imports sorted by `ruff` (isort-compatible): standard library, third-party, first-party, local — separated by blank lines.

Required `ruff` rule sets: `E`, `F`, `W`, `I`, `N`, `UP`, `B`, `D`, `RUF`, `SIM`, `PTH`, `NPY`.

Run before every commit:

```bash
ruff format .
ruff check --fix .
mypy src/
pytest
```

A `pre-commit` hook is configured in the repo — install it with `pre-commit install`.

---

## 3. Naming

Follows PEP 8 with no exceptions:

| Kind | Convention | Example |
|---|---|---|
| Modules and packages | `lower_snake_case` | `kinematics`, `transform_tree` |
| Classes, exceptions, type aliases | `CapWords` | `RigidTransform`, `IKSolver` |
| Functions, methods, variables | `lower_snake_case` | `forward_kinematics`, `q_dot` |
| Constants | `UPPER_SNAKE_CASE` | `DEFAULT_DT`, `GRAVITY` |
| Type variables | `CapWords`, short | `T`, `StateT` |
| Private (module/class internal) | `_leading_underscore` | `_validate_joint_limits` |
| "Really private" (name-mangled) | `__double_underscore` | rare; usually a smell |

Naming guidance:

- **Be specific**: `velocity_world` is better than `v` in a public API; `q` and `q_dot` are fine in math-heavy internals where they match the literature.
- **Frame in the name** when ambiguous: `pose_world_ee`, `twist_body`. Reads as "pose of `ee` expressed in `world`."
- **Avoid abbreviations** except well-known ones (`ik`, `fk`, `dof`, `urdf`, `se3`, `so3`).
- Boolean names: `is_`, `has_`, `should_` prefixes.

---

## 4. Imports

- **Import modules, not names**, when call-site readability benefits from showing the origin:
  ```python
  from ultimate_robotics import kinematics
  q = kinematics.inverse(target)        # good — clear where it comes from
  ```
  vs.
  ```python
  from ultimate_robotics.kinematics import inverse
  q = inverse(target)                   # ambiguous in a long file
  ```
  For tightly-scoped functions used many times, importing the name is fine. Use judgment.

- **Absolute imports only** within the package. No `from .. import foo`.
- **No wildcard imports** (`from x import *`) anywhere.
- **Lazy imports are allowed** for heavy optional dependencies (e.g., `mujoco`, `jax`, ROS bindings). Place the import inside the function or behind a `TYPE_CHECKING` guard. Document why.
- Group order: stdlib → third-party → `ultimate_robotics.*` → relative-equivalent.

---

## 5. Type annotations

Type annotations are **mandatory on all public APIs** and strongly encouraged on internals.

- Use modern syntax: `list[int]`, `dict[str, float]`, `X | None`, not `List`, `Dict`, `Optional`.
- Use `numpy.typing.NDArray` for arrays; document shape and dtype in the docstring.
- Use `typing.Protocol` for structural interfaces (controllers, solvers, sensors). Reserve `abc.ABC` for cases needing shared implementation.
- Use `typing.Self` for fluent methods returning the same type.
- Use `TypeAlias` (or `type` statement on 3.12+) for non-trivial type names that appear in many places.
- `mypy --strict` (or `pyright` in strict mode) must pass. New `# type: ignore` comments require a comment explaining why and ideally a linked issue.

Example:

```python
from typing import Protocol
import numpy as np
from numpy.typing import NDArray

class Controller(Protocol):
    def compute(
        self,
        state: NDArray[np.float64],
        target: NDArray[np.float64],
    ) -> NDArray[np.float64]: ...
```

---

## 6. Docstrings

We use **Google-style docstrings** throughout. Every public module, class, function, and method has one. Trivial private helpers may omit a docstring if the name and signature speak for themselves.

Required sections (when applicable): `Args`, `Returns`, `Raises`, `Yields`, `Example`. Always include shapes, units, and frames where relevant.

```python
def forward_kinematics(
    model: KinematicModel,
    q: NDArray[np.float64],
) -> SE3:
    """Compute the end-effector pose for a given joint configuration.

    Args:
        model: Kinematic model defining link lengths and joint axes.
        q: Joint configuration, shape ``(n,)``, units of radians for revolute
            joints and meters for prismatic joints, ordered as ``model.joints``.

    Returns:
        Pose of the end-effector expressed in the base frame.

    Raises:
        JointLimitError: If any element of ``q`` lies outside the model's
            configured joint limits.

    Example:
        >>> model = load_urdf("panda.urdf")
        >>> q = np.zeros(7)
        >>> T = forward_kinematics(model, q)
    """
```

Conventions inside docstrings:

- **Always state units** (SI by default — see §8).
- **Always state shape** for array parameters: `shape (n,)`, `shape (B, n)` for batched.
- **State frame** for any pose, twist, wrench, or velocity: e.g., "expressed in the world frame."
- Use double backticks for code (`` ``q`` ``), not single quotes.
- One-line summary on the first line, blank line, then details.

---

## 7. API design

### Public vs. private surface

- The public API is whatever is re-exported in a package's `__init__.py` and listed in `__all__`. Everything else is private and may change without notice.
- Anything starting with `_` is private. Internal modules used only by the package live under `_internal/`.
- If users import something not in `__all__`, that's on them.

### Protocols and dependency injection

Public extension points (controllers, planners, IK solvers, sensor backends, hardware interfaces) are defined as `Protocol`s. Concrete implementations are registered or passed in by the caller — they should not be hard-coded.

```python
robot = Robot(
    model=model,
    controller=PDController(kp=100, kd=10),
    ik_solver=DampedLeastSquares(),
)
```

This makes testing trivial (inject mocks) and allows users to plug in their own components without subclassing.

### Composition, not inheritance

A `Robot` *has* a kinematic model, a dynamics model, a controller. It does not inherit from any of them. Inheritance hierarchies for behavior are reserved for cases where shared implementation genuinely justifies it; even then, prefer at most one level deep.

### Batched and single-call APIs

Where it makes sense, public functions accept either a single state `(n,)` or a batched state `(B, n)` and return matching shapes. Document both shapes in the docstring.

For real-time hot paths, provide an in-place variant suffixed `_into` or accepting an `out=` argument:

```python
def forward_kinematics(model, q) -> SE3: ...
def forward_kinematics_into(model, q, out: SE3) -> None: ...
```

---

## 8. Units, frames, and conventions

Non-negotiable, because robotics bugs from these are nightmare-tier.

- **SI units everywhere**: meters, seconds, radians, kilograms, newtons. No degrees in any function signature, ever. If a user-facing tool needs degrees (e.g., a CLI flag), convert at the boundary.
- **Quaternions**: Hamilton convention, ordering `[w, x, y, z]`. Store and document explicitly.
- **Right-handed coordinate frames**, active transforms (transforms move points, not frames).
- **Rigid transforms** are passed around as `SE3` objects, not raw 4×4 matrices, in all public APIs. Internal numerical code may use matrices when justified.
- **Joint configurations** are 1D arrays in the order defined by the model's `joints` attribute. Never reorder silently.
- **Time** is passed explicitly as timestamps or `dt`. No global wall-clock dependency in core code.

---

## 9. Errors and validation

### Exception hierarchy

All exceptions raised by this package derive from `RoboticsError`. Subclasses are domain-specific:

```
RoboticsError
├── KinematicsError
│   ├── SingularConfigurationError
│   └── IKConvergenceError
├── DynamicsError
├── PlanningError
├── ControlError
├── JointLimitError
├── FrameError
└── HardwareError
```

Users should be able to `except IKConvergenceError` without catching unrelated failures.

### Where to validate

- **Validate at public API boundaries**: function entry points, config loading, file I/O, hardware reads.
- **Trust internal callers**: don't re-check the same precondition five layers deep.
- **Use `dataclasses` with `__post_init__`** (or `pydantic` for config from YAML/JSON) to validate structured inputs once, at construction.
- **Never use bare `except:` or `except Exception:`** without re-raising or a comment explaining why. Catch the narrowest exception that makes sense.

---

## 10. Numerics

- **Vectorize with NumPy** by default. Write the readable loop first, profile, then optimize.
- **Numerical tolerances are explicit**: never compare floats with `==`. Use `np.isclose` / `np.allclose` with documented `atol` and `rtol`.
- **Allocating vs. in-place**: by default, functions allocate and return new arrays. In-place variants (`_into`, `out=`) exist where real-time performance demands them.
- **Heavy acceleration** (`numba`, `cython`, `jax`) is isolated to specific modules behind a clear interface, so the core stays pure-Python-readable.
- **Avoid mutable default arguments**. Use `None` and construct inside.
- **Prefer pure functions** in the math layer; side effects belong in the I/O and hardware layers.

---

## 11. Testing

- Framework: **`pytest`** with `pytest-cov`.
- **Property-based tests via `hypothesis`** are strongly encouraged for any code with mathematical invariants — `inv(T) @ T ≈ I`, `fk(ik(target)) ≈ target`, energy conservation, etc.
- **Numerical comparisons** use `numpy.testing.assert_allclose(actual, expected, atol=..., rtol=...)`. Tolerances are explicit and justified.
- Layout:
  ```
  tests/
  ├── unit/          # fast, isolated
  ├── integration/   # cross-module, possibly slow
  └── conftest.py    # shared fixtures (sample robot models, etc.)
  ```
- Slow tests are marked `@pytest.mark.slow` and excluded from the default run.
- **Coverage targets**: >90% on `core/`, `kinematics/`, `dynamics/`; >80% elsewhere. Coverage is a floor, not a goal — a 100%-covered nonsense test suite is worse than 80% of meaningful tests.
- New features land with tests in the same PR.

---

## 12. Logging

- Use the standard `logging` module. **Never `print()` in library code.**
- Each module gets `logger = logging.getLogger(__name__)`.
- The library configures **no handlers by default** — that's the application's job. We attach a `NullHandler` to the package root logger.
- Log levels:
  - `DEBUG`: intermediate values, solver iterations.
  - `INFO`: significant events users would want to see (model loaded, solver converged).
  - `WARNING`: recoverable issues (fell back to a slower solver).
  - `ERROR`: failures that don't raise but should be visible.
- **No f-strings in log calls** — use `%`-style so formatting is deferred when the level is disabled:
  ```python
  logger.debug("IK iter %d residual %.3e", i, residual)
  ```

---

## 13. Versioning, changelog, deprecation

- **Semantic versioning** (`MAJOR.MINOR.PATCH`) — strictly.
- A human-written `CHANGELOG.md` in [Keep a Changelog](https://keepachangelog.com/) format. Every PR with user-visible impact updates it.
- **Deprecation policy**: deprecated APIs emit `DeprecationWarning`, remain functional for at least one minor release, and are removed only in a major release. Document the replacement in the warning message.

```python
import warnings
warnings.warn(
    "forward_kinematics_legacy is deprecated; use forward_kinematics instead.",
    DeprecationWarning,
    stacklevel=2,
)
```

---

## 14. Language features: use and avoid

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
- `eval`, `exec` (essentially never).
- Monkey-patching anything outside the package.

**Avoid:**

- Mutable default arguments.
- Bare `except:` clauses.
- Module-level side effects (besides defining names).
- Global mutable state. Registries are acceptable; configuration singletons are not.

---

## 15. Documentation

- User-facing docs live in `docs/`, built with **Sphinx** + `myst-parser` + `sphinx.ext.napoleon`.
- API reference is auto-generated from docstrings — so docstrings *are* the documentation; treat them accordingly.
- Tutorials and examples are Jupyter notebooks in `docs/examples/`, built into the docs via `myst-nb`. They are run in CI; broken examples fail the build.
- Every public module has a module-level docstring explaining what it's for and what users should reach for first.

---

## 16. Pull requests and reviews

- One logical change per PR. Prefer many small PRs over one large one.
- PR description states **what** changed and **why**, links related issues, and notes any breaking changes.
- All checks green before review: format, lint, type-check, tests, docs build.
- New public API requires: tests, docstring with example, changelog entry, and (if non-trivial) a doc page or notebook.
- Be kind in review. Critique the code, not the author. Suggest concrete alternatives.

---

## Appendix A: Quick checklist for new public functions

Before merging a new public function, confirm:

- [ ] Type-annotated, including return type.
- [ ] Google-style docstring with `Args`, `Returns`, `Raises`, and at least one `Example`.
- [ ] Shapes, units, and frames documented for all array and pose parameters.
- [ ] Validates inputs at the boundary; raises a specific subclass of `RoboticsError`.
- [ ] Has tests, including at least one property-based test if a mathematical invariant exists.
- [ ] Listed in the module's `__all__` and re-exported from the package `__init__.py` if appropriate.
- [ ] Mentioned in `CHANGELOG.md` under "Added."

---

*Last reviewed: April 2026. Open a PR to update.*