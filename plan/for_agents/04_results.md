# 04 Results — Polish the core

**Status:** complete on `dev` (2026-07-18).

## Delivered

- Moved `ModelValues` structural validation to model attachment/rebinding.
  Public kinematics and dynamics now validate call inputs once and perform zero
  hot-path `ModelValues.validate` calls; the boundary-count contract records
  that rule.
- Added one small tensor-boundary helper and compressed the named validation
  walls: contact forces 69 → 12 public lines, IK optimizer config 35 → 11,
  initial configuration 30 → 10, and `Model.with_values.checked` 27 → 5.
- Centralized revolute, prismatic, and helical transform formulas while keeping
  dispatch compile-friendly. Consolidated the four prismatic joint families
  through a shared implementation.
- Centralized optimizer weight helpers, removed caller-dominated residual and
  execution-batch checks, and retained all semantic/range/unit validation.
  Padded-NaN and detached-index residual math is unchanged.
- Replaced `ccrba`'s anonymous tuple with exported, tuple-compatible
  `CCRBAResult(centroidal_map, momentum)` and regenerated the affected API
  reference.
- Removed the downstream impossible optimizer and enum fallbacks, normalized
  boundary errors to `"<what> must <rule>, got <actual>"`, and updated only
  tests whose implementation-detail expectations changed.

## Verification

| Check | Result |
|---|---|
| Full non-benchmark/non-CUDA gate | `1443 passed, 2 skipped, 16 deselected, 38 warnings` |
| Pinocchio parity + hot-path lint + FK compile | `184 passed, 15 warnings` |
| Joint/protocol/compile focus | `121 passed, 15 warnings` |
| Optimizer suite | `230 passed` |
| Final residual focus | `71 passed` |
| Sphinx HTML | succeeded |
| Sphinx doctest | `1 passed`; four expected offline-intersphinx warnings |
| Task-scoped Ruff, format, mypy, pyright, and diff checks | clean |
| Hot-path value-validation grep | no kinematics/dynamics call sites |
| Duplicate-helper grep | one definition of each helper |

CUDA-marked tests were not run, per the standing owner-run rule.

## Line and check accounting

- Task-owned Python under `src/`: 500 additions, 782 deletions, net **−282**.
  This meets the user-revised 250–310-line removal target.
- `ModelValues.validate` moved from repeated public passes to the two model
  attachment paths. Raw tensor seams remain caller-trusted as planned.
- Private tensor checks were removed only when a retained public check or a
  constructed invariant dominates them. Value-content finiteness scans were
  removed under `plan/02_architecture.md` §3; shape, dtype, device, semantic,
  range, and unit checks remain at their boundaries.
- The zero-coordinate optimizer guard was retained: a focused test proved it
  load-bearing for locked/mimic mappings.
- Only the authorized boundary-count contract changed. Ordinary behavior tests
  gained direct named-CCRBA coverage and a Chamfer frame-axis regression; no
  Pinocchio/parity assertion, tolerance, seed, or expected value changed.

## Deviations and findings

1. Three audits showed the original 600-line deletion target could not coexist
   with the no-behavior-change constraint. The user revised it to 250–310;
   actual removal is 282 lines.
2. Valid-input numerical behavior is unchanged. As explicitly required by T5,
   invalid inputs can now expose more specific typed validation subclasses and
   normalized messages. Under the architecture's boundary policy, nonfinite
   tensor contents propagate instead of being redundantly rejected below the
   boundary.
3. Cold `with_values` tensor arguments receive helper checks and then the
   attachment-level structural validation. This small duplicate is deliberate
   and is outside all execution hot paths.
4. Independent residual review found that broadcasting could hide a missing
   Chamfer frame axis. Raw configured event suffixes are now checked before
   cloud broadcasting, with a two-case regression test.
5. Whole-repository Ruff still reports unrelated pre-existing findings; every
   Task 04-owned file is clean.
6. Existing user edits in `docs/concepts/index.md`, untracked
   `docs/concepts/design_decisions.md`, and the matched manifolds source/test
   wording pair were preserved and excluded from this task's commit.

No requested Task 04 work remains incomplete.
