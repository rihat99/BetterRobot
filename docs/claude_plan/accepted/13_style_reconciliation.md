# 13 · Style reconciliation — one normative guide, not two drafts

★ **Hygiene.** Resolves a contradiction between the two `style/`
drafts and the actual codebase. Pick one normative style, fold the
others into it.

## Problem

`docs/style/` contains two drafts authored at different times:

- `style_by_claude.md` — proposes Google-style docstrings, `[w, x, y, z]`
  Hamilton quaternions, `numpy.typing.NDArray`-first APIs, `_into`
  in-place variants, Sphinx + napoleon.
- `style_by_gpt.md` — proposes NumPy-style docstrings, dataclass-first
  configs, "core/edge" split, `numpy`-first too, function suffixes
  for units (`distance_m`, `angle_rad`).

Neither draft matches the actual library:

| Topic | style_by_claude | style_by_gpt | Actual library |
|-------|-----------------|--------------|----------------|
| Tensor library | numpy | numpy | **PyTorch** |
| Quaternion ordering | `[w, x, y, z]` | not specified | **`[qx, qy, qz, qw]`** (per [13_NAMING.md](../../conventions/13_NAMING.md), [17_CONTRACTS.md](../../conventions/17_CONTRACTS.md)) |
| Docstring style | Google | NumPy | mixed; some files Google, some NumPy |
| Configs | dataclass + `__post_init__` | dataclass + `__post_init__` | **dataclass + `__post_init__`** (consistent) |
| Units in names | `_m`, `_rad` suffixes | `_m`, `_rad` suffixes | **not used** — frame embedded in name (per [13 §1.1](../../conventions/13_NAMING.md)) |
| Public-API ceiling | small | small | **25 today, ~31 proposed** |

Drafts that contradict the code are worse than no drafts at all —
a contributor reading them will use the wrong quaternion ordering or
the wrong docstring format.

## Goal

A single **normative** in-tree style guide that matches the codebase
and the existing conventions docs. The two drafts collapse into it
or are explicitly archived.

## The proposal

### 13.A One normative file

```
docs/conventions/19_STYLE.md           # normative; cross-cutting
docs/style/                            # archived drafts; kept for reference
```

`19_STYLE.md` slots into the conventions index (currently 13–17).
Once it lands, the two drafts in `docs/style/` get a banner:

> *Archived draft.* The normative coding-style guide is
> [conventions/19_STYLE.md](../../conventions/19_STYLE.md). This file
> is kept for historical reference and may contradict the current
> spec.

### 13.B Topics 19_STYLE.md owns

**Naming**

- Modules / packages: `lower_snake_case`.
- Classes: `CapWords`. Tensor type aliases: `CapWords` too
  (`Float`, `Tensor`, `SE3Tensor`).
- Functions / methods: `lower_snake_case`.
- Constants: `UPPER_SNAKE_CASE`.
- Type variables: `T`, `StateT` (short, `CapWords`).
- Booleans: `is_`, `has_`, `should_`.
- **No unit suffixes in identifiers** (`_m`, `_rad`). Reasoning:
  - The library is SI-only on the public surface (per
    [17 §1](../../conventions/17_CONTRACTS.md)).
  - The `<entity>_<quantity>_<frame>` convention from
    [13 §1.1](../../conventions/13_NAMING.md) already encodes the
    semantic disambiguation.
  - `joint_velocity_world` and `joint_velocity_world_radps` are
    redundant.

**Quaternions and Lie storage**

The format is fixed:

| Object | Layout |
|--------|--------|
| SE3 pose | `(..., 7) [tx, ty, tz, qx, qy, qz, qw]` |
| SO3 quat | `(..., 4) [qx, qy, qz, qw]` |
| se3 tangent | `(..., 6) [vx, vy, vz, wx, wy, wz]` (linear first) |
| so3 tangent | `(..., 3) [wx, wy, wz]` |
| Spatial Jacobian | `(..., 6, nv) [linear rows; angular rows]` |

This matches PyPose's native layout and the existing code.
Quaternions are scalar-last (Hamilton convention; `qw` last). It is
**not** the `[w, x, y, z]` order in `style_by_claude.md`. New code
must follow this.

**Docstrings — NumPy style**

Pick one of Google/NumPy. We pick **NumPy** for these reasons:

- Sphinx + Napoleon supports both, but the scientific-Python
  ecosystem (numpy, scipy, scikit-learn, jax) is NumPy-style. New
  users coming from those projects will recognise the pattern.
- NumPy-style is friendlier to multi-paragraph parameter docs.
- The `Shapes` block from
  [16 §4](../../conventions/16_TESTING.md)-style tests fits naturally as
  a sub-section.

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
    SE3(data=tensor([...]))
    """
```

The `>>>` lines run under doctest (and under MyST-NB in tutorial
notebooks per [Proposal 10](10_user_docs_diataxis.md)).

**Types**

Per [Proposal 04](04_typing_shapes_and_enums.md):

- Public surface: full `jaxtyping`-style annotations.
- `from __future__ import annotations` at the top of every file
  (PEP 563-style, lazy evaluation).
- `dict[str, ...]`, not `Dict[str, ...]`. `X | None`, not
  `Optional[X]`. (Min Python is 3.10.)
- `typing.Protocol` for extension seams; `abc.ABC` only when
  shared implementation is required.

**Imports**

- Absolute imports inside the package (`from better_robot.lie import se3`,
  not `from ..lie import se3`).
  *Exception*: short relative imports (`from .se3 import compose`)
  are fine inside a single sub-package.
- No wildcard imports anywhere.
- Lazy imports are allowed for **heavy optional** deps (mujoco, viser,
  warp). Document why with an inline comment.
- Group order (`ruff`/isort): stdlib → third-party →
  `better_robot.*` → local relative.

**Configs**

- `@dataclass(frozen=True)` for all configs.
- Validation in `__post_init__`. Raise `BetterRobotError` subclasses,
  not bare `ValueError`.
- No Pydantic; no custom metaclass. Per
  [00 §Non-goals](../../design/00_VISION.md).

**Errors**

- Subclass `BetterRobotError` (already done in `exceptions.py`).
- Raise the most specific exception possible.
- Error messages name the offending input and what was expected
  (e.g., "q has trailing size 6, expected 7 for Panda").
- Bare `except:` and `except Exception:` are forbidden in `src/`
  (lint rule).

**Logging**

- `logger = logging.getLogger(__name__)` per module.
- `print()` is forbidden in `src/`. Allowed in `examples/` and the
  CLI.
- The library configures `logging.NullHandler` on `better_robot`
  (already done in `utils/logging.py`).
- Format strings use `%` placeholders for deferred formatting:
  `logger.debug("iter %d residual %.3e", i, r)`.

**File organisation**

- One concept per file. Not `utils.py`. Not `helpers.py`.
- Tests live in `tests/<package>/test_<concept>.py`.
- A package's `__init__.py` re-exports its public symbols and lists
  them in `__all__`.

### 13.C What the `style/` drafts get right that 19_STYLE.md keeps

Both drafts agree on (and we adopt):

- Pure functions for math, side effects at the edges.
- Composition over inheritance for behaviour.
- Validate at boundaries; trust internal callers.
- Standard `dataclass` over Pydantic / custom metaclass.
- SI units everywhere on the public surface.
- A `BetterRobotError` exception hierarchy.
- Semantic versioning + Keep-a-Changelog format.
- Pre-commit hooks; `ruff` for format and lint; `mypy --strict`
  for type checks.
- Pytest + `pytest-cov` + property-based testing via Hypothesis.

### 13.D What the drafts get wrong and 19_STYLE.md drops

| Draft says | We say | Reason |
|------------|--------|--------|
| Hamilton `[w, x, y, z]` | `[qx, qy, qz, qw]` | PyPose-native; matches the entire codebase. |
| `numpy.typing.NDArray` | `torch.Tensor` (with `jaxtyping` shape annotations) | We are PyTorch-first. |
| Google docstrings | NumPy | Ecosystem fit. |
| `_into` / `out=` in-place variants | not adopted in v1 | Not warranted by current performance gap; can be added later if benchmarks demand it. |
| Unit suffixes (`distance_m`) | not adopted | Redundant with `<entity>_<quantity>_<frame>`. |
| Wildcard imports forbidden | adopted | Already enforced by ruff `F405`. |
| `pre-commit install` | adopted | Already in [Proposal 11](11_quality_gates_ci.md). |

## The skill files

The workspace already has skill files: `python-standards`,
`cpp-standards`, `file-naming`, `design-principles`, `write-tests`,
`diataxis-docs`, `sphinx-docs`. These are **operational** — for
Claude Code agents and external contributors. `19_STYLE.md` is the
**canonical** doc; the skills can cite it.

Concretely, `python-standards.md` should have a header:

> *This skill is the operational form of
> `docs/conventions/19_STYLE.md`. When in doubt, the doc is
> normative.*

This avoids drift between the skill (which is short and pragmatic)
and the doc (which is exhaustive).

## Files that change

```
docs/conventions/19_STYLE.md           new — the normative guide
docs/style/style_by_claude.md           prepend "ARCHIVED — see 19_STYLE.md" banner
docs/style/style_by_gpt.md              prepend the same banner
docs/conventions/13_NAMING.md           cross-reference 19 §Naming
docs/README.md                          add 19 to the conventions table
.claude/skills/python-standards.md      add a "see 19_STYLE.md" header
```

## Tradeoffs

| For | Against |
|-----|---------|
| One source of truth; new contributors know what to follow. | Writing 19_STYLE.md takes a few hours. |
| Drafts archived rather than deleted — historical context preserved. | Slight redundancy in `style/`. Mitigation: the banner is unmissable. |
| Skill files cite the doc; drift between the two is impossible to land silently. | Skills must be updated when the doc changes. Mitigation: the citation is short; the skill carries minimal duplication. |

## Acceptance criteria

- `docs/conventions/19_STYLE.md` exists with the topics above.
- Both `style/style_by_*.md` files have an "ARCHIVED" banner.
- `docs/README.md` lists 19 in the conventions table.
- `python-standards` skill cites `19_STYLE.md`.
- One round of cleanup PRs converts mixed-style docstrings (Google
  vs NumPy) to NumPy form. The PR is small, mechanical, and uses
  `ruff` + `numpydoc-validation` to find offenders.

## Cross-references

- [13 §1.1](../../conventions/13_NAMING.md) — entity/quantity/frame
  naming pattern.
- [17 §1.3](../../conventions/17_CONTRACTS.md) — quaternion convention.
- [Proposal 04](04_typing_shapes_and_enums.md) — typing standards.
- [Proposal 10](10_user_docs_diataxis.md) — Sphinx + Napoleon
  setup that consumes the docstring style.
- [Proposal 11](11_quality_gates_ci.md) — pre-commit and ruff
  configuration.
