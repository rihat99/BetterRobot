# 04 · Typing, shape annotations, and string-literal cleanup

★★ **Structural.** Introduces explicit shape annotations on the
public API and replaces a handful of free-form string literals with
enums. No runtime behaviour change.

## Problem

Three small things compound into a real ergonomic gap:

### 4.1 Shape annotations are in docstrings, not types

Every public function declares its shapes in the docstring:

```python
def forward_kinematics(model: Model, q_or_data: torch.Tensor | Data,
                       *, compute_frames: bool = False) -> Data:
    """Compute the placements of every joint ...

    Shapes
    ------
    q                              (B..., nq)
    data.joint_pose_world          (B..., njoints, 7)    filled
    data.joint_pose_local          (B..., njoints, 7)    filled
    data.frame_pose_world          (B..., nframes, 7)    filled iff compute_frames=True
    """
```

Docstrings are good documentation but bad enforcement. A typo in the
docstring is invisible to mypy; an off-by-one shape change is
invisible to the IDE. `jaxtyping` (and the upcoming
`torch._jit_internal.Tensor[...]`) gives us *types* for shapes:

```python
from jaxtyping import Float
from torch import Tensor

NQ = "nq"   # type-level dim names; identifiers are documentation
NV = "nv"
NJ = "njoints"
NF = "nframes"

def forward_kinematics(
    model: Model,
    q_or_data: Float[Tensor, "*B nq"] | Data,
    *,
    compute_frames: bool = False,
) -> Data: ...
```

The annotation is now searchable, IDE-visible, and (with `jaxtyping`)
optionally checkable at runtime via a single decorator in tests.

### 4.2 Reference-frame dispatch via stringly-typed argument

```python
# kinematics/jacobian.py
def get_frame_jacobian(model, data, frame_id,
                       reference: str = "local_world_aligned") -> Tensor: ...
```

`reference` is one of three values today (`"world"`, `"local"`,
`"local_world_aligned"`) — but the type system permits arbitrary
strings. A typo (`"local_aligned"`) returns the default with no
warning. Pinocchio uses an enum (`pinocchio.ReferenceFrame.WORLD`,
etc.); we should too.

### 4.3 `JacobianStrategy.AUTO` falls back through tiers

Today: `JacobianStrategy in {ANALYTIC, AUTODIFF, FUNCTIONAL, AUTO}`.
What `AUTO` actually does (per
[05 §3](../../design/05_KINEMATICS.md)):

> AUTO — call residual.jacobian(state); if it returns None, fall
> back to AUTODIFF. This lets individual residuals advertise analytic
> support without the solver caring.

[Proposal 03] retires the central-FD fallback. After that, the
ladder is `ANALYTIC → AUTODIFF`. We should add a fourth value
`FINITE_DIFF` for the rare case a user explicitly wants FD (e.g.,
to validate a hand-coded analytic Jacobian during development), and
make the ordering explicit:

```python
class JacobianStrategy(str, Enum):
    ANALYTIC     = "analytic"
    AUTODIFF     = "autodiff"
    FUNCTIONAL   = "functional"     # forward-mode autodiff
    FINITE_DIFF  = "finite_diff"    # opt-in
    AUTO         = "auto"           # ANALYTIC -> AUTODIFF
```

## The proposals

### 4.A Adopt `jaxtyping`-style annotations for the public surface

For every symbol in `better_robot.__all__` (and for protocols in
`residuals/`, `optim/`, `data_model/joint_models/`):

- Use `Float[Tensor, "*B nq"]`, `Float[Tensor, "*B njoints 7"]`, etc.
  in the type signature.
- Keep the docstring's "Shapes" block — it remains the human-readable
  reference and cross-validates the annotation.
- Internal modules (private functions, hot-loop helpers) may opt out
  per-file for readability; we don't enforce annotations there.

Add the `jaxtyping` dependency under `extras = ["dev"]` (it's
optional at runtime; only the type checker and `tests/contract/`
need it).

**Migration is staged, not big-bang.** Forcing every public function
to land annotated at once produces a large, mostly-mechanical PR
that crowds out review of the ideas. The staged plan:

| Stage | Scope | CI gate |
|-------|-------|---------|
| 1 | `_typing.py` aliases land; new public APIs annotated. | Advisory: PR description mentions any new unannotated public surface. |
| 2 | Touched public functions migrate during their next PR. | Advisory: contract test reports unannotated symbols, does not fail. |
| 3 | All public functions annotated. | Blocking: `tests/contract/test_shape_annotations.py` fails on regressions. |

Stage 3 lands once stage 2 has converged — typically a release
cycle later. The contract test exists from stage 1 onward; it just
runs in advisory mode (printing a coverage report) until the flip
to blocking.

### 4.B Replace `reference` strings with `ReferenceFrame` enum

```python
# kinematics/jacobian.py
from enum import Enum

class ReferenceFrame(str, Enum):
    WORLD              = "world"
    LOCAL              = "local"
    LOCAL_WORLD_ALIGNED = "local_world_aligned"

def get_frame_jacobian(
    model: Model,
    data: Data,
    frame_id: int,
    *,
    reference: ReferenceFrame = ReferenceFrame.LOCAL_WORLD_ALIGNED,
) -> Float[Tensor, "*B 6 nv"]: ...
```

Subclassing `str` keeps the value comparable to the legacy strings
(`reference == "world"` continues to work), so the migration is
painless. The enum is exported in `better_robot.__all__` and
documented in
[13_NAMING.md](../../conventions/13_NAMING.md).

Same pattern for the kinematics-level invariant on `Data`:

```python
class KinematicsLevel(int, Enum):
    NONE          = 0
    PLACEMENTS    = 1
    VELOCITIES    = 2
    ACCELERATIONS = 3
```

(Used by [Proposal 07](07_data_cache_invariants.md).)

### 4.C Type aliases live in `_typing.py`

`_typing.py` exists today. Populate it as the canonical home of
shape aliases:

```python
# src/better_robot/_typing.py
"""Shape-annotated type aliases used across the public API.

Identifier choice follows docs/conventions/13_NAMING.md: names are
the documentation, not the spec.
"""
from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from jaxtyping import Float, Int
    from torch import Tensor

    # Single SE3 / SO3 storage tensors
    SE3Tensor = Float[Tensor, "*B 7"]
    SO3Tensor = Float[Tensor, "*B 4"]
    # Per-joint and per-frame stacks
    JointPoseStack  = Float[Tensor, "*B njoints 7"]
    FramePoseStack  = Float[Tensor, "*B nframes 7"]
    # Configurations and tangents
    ConfigTensor    = Float[Tensor, "*B nq"]
    VelocityTensor  = Float[Tensor, "*B nv"]
    # Jacobians
    JointJacobian   = Float[Tensor, "*B 6 nv"]
    JointJacobianStack = Float[Tensor, "*B njoints 6 nv"]
```

Public functions import these aliases to keep signatures readable.

### 4.D mypy-strict on the public surface; relaxed inside

A two-tier policy:

| Scope | Strictness |
|-------|-----------|
| `better_robot.__all__` symbols, all `Protocol`s | `mypy --strict` clean |
| Public `kinematics/`, `dynamics/`, `tasks/`, `optim/`, `io/` modules | full annotations, `--strict` clean |
| Internal helpers (`_*.py`, `viewer/internal_*.py`, hot-loop kernels) | annotations encouraged, not enforced |
| `lie/_*_backend.py`, `tests/` | annotations optional |

Configured in `pyproject.toml`:

```toml
[tool.mypy]
strict = true
warn_unused_ignores = true

[[tool.mypy.overrides]]
module = ["better_robot.viewer.internal_*", "better_robot.tests.*"]
ignore_errors = true
```

A pre-commit hook runs `mypy src/better_robot/` on every commit
touching `src/`.

### 4.E Pyright (recommended) for IDE feedback

`pyright` is cheaper to run incrementally than `mypy --strict`. Add a
`pyrightconfig.json` with the same scope rules; CI runs `mypy` (it
catches more), IDEs use `pyright`.

## What about Pydantic / `dataclass` validators?

For dataclasses on the public surface (`IKCostConfig`,
`OptimizerConfig`, etc., see
[Proposal 06](06_public_api_audit.md)), keep plain `@dataclass` +
`__post_init__` validation.
[00_VISION.md §Non-goals](../../design/00_VISION.md) is explicit about
not introducing a custom config metaclass; that holds. Pydantic v2
is fast but pulls in a heavy dependency for marginal benefit; we
don't need it for v1.

## Files that change

```
_typing.py                              extended — full alias table
pyproject.toml                          dev extras add jaxtyping; mypy/pyright config
kinematics/jacobian.py                  reference: str -> ReferenceFrame
kinematics/jacobian_strategy.py         add FINITE_DIFF
data_model/data.py                      _kinematics_level: int -> KinematicsLevel
                                        (cross-references Proposal 07)
*.py (public modules)                   imports from _typing
tests/contract/test_shape_annotations.py  new — AST walk
tests/contract/test_no_legacy_strings.py  new — grep for reference="..."
__init__.py                             export ReferenceFrame, KinematicsLevel
                                        (cross-references Proposal 06)
```

## Tradeoffs

| For | Against |
|-----|---------|
| `jaxtyping` annotations are the same shape conventions in machine-readable form. | Adds a dev dependency and a contract test. Mitigation: optional at runtime; tests skip if not installed. |
| `ReferenceFrame` enum eliminates a class of typo-bugs. | Migration touches every call site that used strings. Mitigation: `str`-subclassed enum keeps backwards compat for one release. |
| `KinematicsLevel` enum makes Proposal 07 readable. | None significant. |
| mypy-strict catches signature regressions before review. | Type errors will appear and need fixing. Mitigation: relax for internal modules; one-time cleanup. |

## Acceptance criteria

- *Stage 1:* `_typing.py` exports the alias table; new public APIs
  use `jaxtyping`-style annotations on every tensor parameter and
  return type; `tests/contract/test_shape_annotations.py` runs in
  advisory mode (prints coverage; does not fail PRs).
- *Stage 3 (later release):* every symbol in
  `better_robot.__all__` has a `jaxtyping`-style annotation on every
  tensor parameter and return type; the contract test flips to
  blocking with no `# noqa: shape` exceptions in the public surface.
- `mypy --strict src/better_robot/` is clean (modulo the documented
  internal opt-outs).
- `ReferenceFrame.WORLD == "world"` is `True` (str-subclass keeps
  backwards compatibility).
- `tests/contract/test_no_legacy_strings.py` greps the source tree
  for `reference="(world|local|local_world_aligned)"` and finds
  nothing in `src/`.
- Pre-commit runs `ruff format`, `ruff check`, and `mypy` on every
  commit touching `src/`.

## Cross-references

- [Proposal 06](06_public_api_audit.md) — adds `ReferenceFrame`,
  `KinematicsLevel`, and `JacobianStrategy.FINITE_DIFF` to
  `__all__`.
- [Proposal 07](07_data_cache_invariants.md) — uses
  `KinematicsLevel`.
- [Proposal 11](11_quality_gates_ci.md) — wires mypy and the new
  contract tests into CI.
