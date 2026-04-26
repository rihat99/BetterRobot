# 06 · Public API audit — what to add, what to keep, what to clarify

★★ **Structural.** Renegotiates the 25-symbol cap from
[01_ARCHITECTURE.md §Public API contract](../../design/01_ARCHITECTURE.md).

## Problem

`better_robot.__all__` is pinned at exactly 25 symbols, with an
`assert len(__all__) == 25` at the bottom of `__init__.py`. The
discipline is good: every public name is a contract, and a hard cap
forces deliberation. But "25" is arbitrary, and the current cap has
two consequences:

### 6.1 Some user-facing types are not exported

To do idiomatic IK / FK code today, users must reach into
sub-modules:

```python
from better_robot.tasks.ik import IKResult, IKCostConfig, OptimizerConfig
from better_robot.kinematics import ReferenceFrame   # if Proposal 04 lands
from better_robot.optim.state import SolverState
```

The `ik.py` ones are the most visible — every user of `solve_ik`
uses them. They are de-facto public.

### 6.2 Some exports are noise

- `register_residual` is exported — but a user who is not writing a
  custom residual will never call it. It's an extension-point, not
  a routine API.
- `compute_centroidal_map` is exported — but `rnea`, `aba`, `crba`
  (which are also exported and equally stub-only) overshadow it.
- `Joint`, `Body`, `Frame` are exported — but `Body` and `Frame` are
  rarely directly constructed by user code; they're metadata
  attached to `Model`.

The proposal: keep the discipline, drop the magic number, and make
the surface match how users actually program.

## Goal

`__all__` reflects exactly what users reach for in idiomatic code.
Anything else lives one level deeper but is **fully reachable** via
qualified imports. The discipline is the audit, not the cap; the
default bias is **lean top-level + rich submodule surface**.

## The proposal

### 6.A The new `__all__` — promote conservatively

A *small* set of additions. The principle: top-level is for symbols
a user *literally writes the name of in their first program*. Other
genuinely public symbols stay one submodule down.

```python
__all__ = [
    # ───────── Robot model & data (7) ─────────
    "Model",
    "Data",
    "Frame",
    "Body",
    "Joint",
    "load",
    "ModelBuilder",                 # new — programmatic builder; widely used

    # ───────── Geometric value types — top-level (1) ─────────
    "SE3",                          # new — Proposal 01; the headline pose type
    # SO3, Pose, Motion, Force, Inertia, Symmetric3 stay submodule-level:
    #   from better_robot.lie import SO3, Pose
    #   from better_robot.spatial import Motion, Force, Inertia

    # ───────── Kinematics (5) ─────────
    "forward_kinematics",
    "update_frame_placements",
    "compute_joint_jacobians",
    "get_joint_jacobian",
    "get_frame_jacobian",
    # ReferenceFrame stays under better_robot.kinematics:
    #   from better_robot.kinematics import ReferenceFrame

    # ───────── Dynamics (5) ─────────
    "rnea",
    "aba",
    "crba",
    "center_of_mass",
    "compute_centroidal_map",

    # ───────── Optimisation primitives (3) ─────────
    "JacobianStrategy",
    "CostStack",
    "LeastSquaresProblem",
    # SolverState stays under better_robot.optim.state

    # ───────── Tasks (4) ─────────
    "solve_ik",
    "solve_trajopt",
    "retarget",
    "Trajectory",
    # IKResult, IKCostConfig, OptimizerConfig stay under better_robot.tasks.ik:
    #   from better_robot.tasks.ik import IKResult, IKCostConfig, OptimizerConfig

    # ───────── Extension points (1) ─────────
    "register_residual",
]
```

That's **26 symbols** — one above today's 25. The promotion is
deliberately conservative:

- **Promoted:** `ModelBuilder`, `SE3`. Both are ones users name on
  their first program (build a custom robot; type-annotate a pose
  variable).
- **Not promoted:** `IKResult`, `IKCostConfig`, `OptimizerConfig`.
  These are *configuration* and *result* types — every user of
  `solve_ik` touches them, but the import statement
  `from better_robot.tasks.ik import IKResult` is a one-liner that
  anchors them in the right submodule. They are revisited once
  examples and tutorials demonstrate that the qualified path is the
  friction users actually hit.
- **Not promoted (yet):** `SO3`, `Pose`, `Motion`, `Force`, `Inertia`.
  Submodule access (`better_robot.lie.SO3`,
  `better_robot.spatial.Motion`) is the right home until we have
  evidence users repeatedly need them at top-level. The 60-second
  test for the README is "load a robot, run FK, solve IK" — `SE3`
  is on that path; `Motion` is not.
- **Not promoted ever:** `Symmetric3`, every Protocol, every IR
  type, every `JointModel` subclass, every solver internal.
- **Removed (renegotiated later):** `JointKind` is held back —
  the joint taxonomy proposal is still in flux and a top-level
  enum is premature.

This is the gpt-plan-driven correction to an earlier draft of this
proposal that promoted ~31 symbols. The principle:

> **Promote on evidence, not aesthetics.** A symbol earns top-level
> status when example code, tutorials, or user issues demonstrate
> the qualified path is the friction.

Subsequent revisions of this list happen in named PRs that update
both `__all__` and the `EXPECTED` set in 6.C, so each promotion is
visible in the diff.

### 6.B What is **not** exported (intentionally)

| Symbol | Why kept private |
|--------|------------------|
| `JointModel` (Protocol) | Extension API; users implement it but don't construct one for routine use. |
| `Residual` (Protocol) | Same. |
| `Optimizer`, `LinearSolver`, `RobustKernel`, `DampingStrategy`, `StopScheduler` | Extension protocols. |
| `IRModel`, `IRJoint`, `IRBody`, `IRFrame`, `IRGeom` | Parser internals. Importable from `better_robot.io.ir`. |
| `Symmetric3` | Internal helper. (See [Proposal 05](05_value_types_audit.md).) |
| `LeastSquaresProblem.state_factory`, `JacobianSpec` | Solver internals. |
| `RobotCollision` | Optional layer; importable from `better_robot.collision`. |
| `WorldCollision`, `SelfCollision` residuals | Importable from `better_robot.residuals.collision`. |
| `JointFreeFlyer`, `JointRX`, `JointSpherical`, ... | Importable from `better_robot.data_model.joint_models`. Loaders pick the right ones. |

These remain reachable via the qualified path; they are simply not
in the top-level `import better_robot`.

### 6.C Drop the magic-number assert

Replace:

```python
assert len(__all__) == 25, f"better_robot.__all__ must have 25 symbols, not {len(__all__)}"
```

with a contract test:

```python
# tests/contract/test_public_api.py

EXPECTED = frozenset({
    "Model", "Data", "Frame", "Body", "Joint",
    "load", "ModelBuilder",
    "SE3",
    "forward_kinematics", "update_frame_placements",
    "compute_joint_jacobians", "get_joint_jacobian",
    "get_frame_jacobian",
    "rnea", "aba", "crba", "center_of_mass", "compute_centroidal_map",
    "JacobianStrategy", "CostStack", "LeastSquaresProblem",
    "solve_ik", "solve_trajopt", "retarget", "Trajectory",
    "register_residual",
})

def test_public_api_exact():
    """Public API change requires updating EXPECTED in this test."""
    assert set(better_robot.__all__) == EXPECTED, (
        f"missing: {EXPECTED - set(better_robot.__all__)}; "
        f"extra: {set(better_robot.__all__) - EXPECTED}"
    )
```

The discipline: any addition or deletion of a public symbol is a
PR that updates *both* the export list *and* this test, side by
side. The reviewer reads the diff and decides.

A separate contract test enforces **submodule reachability** for
the symbols intentionally not at top-level:

```python
def test_submodule_public_imports():
    """Symbols not at top-level must still be reachable from a
    documented submodule path."""
    from better_robot.lie import SO3, Pose
    from better_robot.spatial import Motion, Force, Inertia, Symmetric3
    from better_robot.kinematics import ReferenceFrame
    from better_robot.optim.state import SolverState
    from better_robot.tasks.ik import IKResult, IKCostConfig, OptimizerConfig
```

If a refactor breaks one of these submodule paths, the test fails
and the PR is either fixing the path or making an explicit
breakage decision — not silently moving symbols.

### 6.D Stability guarantees per symbol

Add a small table to
[17_CONTRACTS.md §7.3](../../conventions/17_CONTRACTS.md) per-symbol:

| Tier | Meaning | Examples |
|------|---------|----------|
| Stable | SemVer-bound; major bump to remove/rename | `Model`, `Data`, `forward_kinematics`, `solve_ik`, `SE3` |
| Stable (Protocol) | Extending the protocol (adding methods) is a major bump; using existing methods is stable | `JointModel`, `Residual`, `Optimizer` |
| Experimental | May change in minor releases with a deprecation warning | `solve_trajopt`, `retarget`, `Trajectory`, `compute_centroidal_map` |

This already exists in [17 §7.3](../../conventions/17_CONTRACTS.md) at
the *module* level; per-symbol gives finer control.

### 6.E Documentation surface

Each public symbol ships with:

- A one-line summary docstring that names the symbol's purpose.
- A `Shapes` block (for tensor-valued returns) or a parameter list.
- An example: at minimum a 3-line snippet a reader can paste.
- A reference back to the design doc that owns it.

Test in `tests/contract/test_docstrings.py` already runs the
docstring-presence check; extending it to verify each symbol has at
least one of `Examples:`, `Example:`, `>>>`, or `Doctest` is a
two-line addition.

### 6.F Suggested extras (`from better_robot import *`)

We do not ship `__all__` to control `from x import *` semantics —
[Proposal 13](13_style_reconciliation.md) records that wildcard
imports are forbidden in user code style. `__all__`'s only purpose is
**discoverability**: it is what shows up in the IDE pop-up when a
user types `import better_robot as br; br.<TAB>`.

## Why audit, not bump

The current cap (25) was a forcing function during the skeleton
phase. It worked: it caught additions that should have lived in
sub-modules. But the audit happens *during* the addition, not at the
cap. With the contract test in 6.C, the audit is required for every
change — and the cap becomes redundant.

Choosing 26 over 31 (or any other number) is the **discipline
correctly applied**. The right top-level surface is the one users
actually reach for in the first program, not the maximum of what
*could* fit. Future promotions are evidence-driven and reviewed in
PR diffs.

## Tradeoffs

| For | Against |
|-----|---------|
| Users don't have to remember which sub-module hides `IKResult`. | Larger `__all__`; the doc tables in [01_ARCHITECTURE.md](../../design/01_ARCHITECTURE.md) and the README grow. |
| Per-symbol stability tier makes deprecation policy clearer. | Slightly more bookkeeping. Mitigation: the tier is a single line in the docstring. |
| Removing the magic-number cap aligns the assert with the actual review process. | Some contributors liked the hard cap as a sign-off discipline. Mitigation: the contract test enforces the same discipline more precisely. |

## Acceptance criteria

- `better_robot.__all__` matches the EXPECTED set in
  `tests/contract/test_public_api.py`.
- `from better_robot import SE3, ModelBuilder` works.
- `from better_robot.lie import SO3, Pose` works.
- `from better_robot.spatial import Motion, Force, Inertia, Symmetric3`
  works.
- `from better_robot.tasks.ik import IKResult, IKCostConfig, OptimizerConfig`
  works.
- `from better_robot.kinematics import ReferenceFrame` works.
- `from better_robot import Symmetric3` raises `ImportError`
  (Symmetric3 is reachable via `better_robot.spatial`, not
  top-level).
- Every symbol in `__all__` has a docstring with at least one
  example.
- [01_ARCHITECTURE.md §Public API contract](../../design/01_ARCHITECTURE.md)
  is updated with the new list and the per-symbol stability tier.
- The `assert len(__all__) == 25` line is removed; the test takes
  over.

## Cross-references

- [Proposal 01](01_lie_typed_value_classes.md) — adds
  `SE3`/`SO3`/`Pose`.
- [Proposal 04](04_typing_shapes_and_enums.md) — adds
  `ReferenceFrame`.
- [Proposal 05](05_value_types_audit.md) — adds `Motion`, `Force`,
  `Inertia` (already in `spatial/__all__`, now top-level).
- [Proposal 08](08_trajectory_lock_in.md) — pins `Trajectory`
  before exposing it.
