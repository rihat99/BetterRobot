# BetterRobot — Strategic Proposals (claude_plan)

> **Status:** all 17 proposals **accepted** (2026-04-25). The proposal
> bodies have moved to [`accepted/`](accepted/); their content has been
> folded into the canonical specs under
> [`../design/`](../design/) and [`../conventions/`](../conventions/).
> Implementation phases live in [`../UPDATE_PHASES.md`](../UPDATE_PHASES.md).

This folder is the audit trail of a strategic review of BetterRobot at
the end of phase-1 (skeleton + IK working, dynamics stubs, viewer
partial, Warp backend stubbed). The brief was to **stop and think**
before filling in stubs — to identify high-leverage changes while
breaking them is still cheap.

The proposals span three altitudes:

- **High-altitude** — module shape, public-API contract, backend
  abstraction, value-type story.
- **Mid-altitude** — config-object discipline, cache-invariant
  enforcement, regression and benchmark infrastructure.
- **Low-altitude** — naming, docstring style, packaging extras.

## What's in this folder now

| File | Purpose |
|------|---------|
| [00_principles.md](00_principles.md) | The lens these proposals were drafted through (kept for posterity) |
| [RECONCILIATION.md](RECONCILIATION.md) | Audit trail of how this folder was reconciled with the gpt_plan review |
| [accepted/](accepted/) | The 17 accepted proposals — historical record |

## Where the proposals landed

Each proposal folded into one or more canonical docs. The fold-in is
described inline in the relevant spec; this table is the index.

| # | Proposal | Folded into |
|---|----------|-------------|
| 01 | [lie_typed_value_classes](accepted/01_lie_typed_value_classes.md) | [03_LIE_AND_SPATIAL.md §7](../design/03_LIE_AND_SPATIAL.md), [01_ARCHITECTURE.md](../design/01_ARCHITECTURE.md) |
| 02 | [backend_abstraction](accepted/02_backend_abstraction.md) | [10_BATCHING_AND_BACKENDS.md §7](../design/10_BATCHING_AND_BACKENDS.md), [01_ARCHITECTURE.md](../design/01_ARCHITECTURE.md) |
| 03 | [replace_pypose](accepted/03_replace_pypose.md) | [03_LIE_AND_SPATIAL.md §10](../design/03_LIE_AND_SPATIAL.md), [PYPOSE_ISSUES.md](../status/PYPOSE_ISSUES.md), [UPDATE_PHASES P10](../UPDATE_PHASES.md) |
| 04 | [typing_shapes_and_enums](accepted/04_typing_shapes_and_enums.md) | [13_NAMING.md §2.8 / §2.9](../conventions/13_NAMING.md), [05_KINEMATICS.md](../design/05_KINEMATICS.md), [02_DATA_MODEL.md](../design/02_DATA_MODEL.md) |
| 05 | [value_types_audit](accepted/05_value_types_audit.md) | [03_LIE_AND_SPATIAL.md §7, §7.X](../design/03_LIE_AND_SPATIAL.md) |
| 06 | [public_api_audit](accepted/06_public_api_audit.md) | [01_ARCHITECTURE.md §Public API contract](../design/01_ARCHITECTURE.md), [16_TESTING.md §5.2](../conventions/16_TESTING.md), [17_CONTRACTS.md §7.3](../conventions/17_CONTRACTS.md) |
| 07 | [data_cache_invariants](accepted/07_data_cache_invariants.md) | [02_DATA_MODEL.md §3.1](../design/02_DATA_MODEL.md) |
| 08 | [trajectory_lock_in](accepted/08_trajectory_lock_in.md) | [08_TASKS.md §2](../design/08_TASKS.md) |
| 09 | [human_body_extension_lane](accepted/09_human_body_extension_lane.md) | [15_EXTENSION.md §15](../conventions/15_EXTENSION.md), [20_PACKAGING.md §1.1](../conventions/20_PACKAGING.md) |
| 10 | [user_docs_diataxis](accepted/10_user_docs_diataxis.md) | [UPDATE_PHASES Phase P12](../UPDATE_PHASES.md) |
| 11 | [quality_gates_ci](accepted/11_quality_gates_ci.md) | [16_TESTING.md §4.6](../conventions/16_TESTING.md), [UPDATE_PHASES Phase P8](../UPDATE_PHASES.md) |
| 12 | [regression_and_benchmarks](accepted/12_regression_and_benchmarks.md) | [16_TESTING.md §4.5](../conventions/16_TESTING.md), [14_PERFORMANCE.md §4.2](../conventions/14_PERFORMANCE.md), [UPDATE_PHASES Phase P9](../UPDATE_PHASES.md) |
| 13 | [style_reconciliation](accepted/13_style_reconciliation.md) | [conventions/19_STYLE.md](../conventions/19_STYLE.md) (new normative file) |
| 14 | [dynamics_milestone_plan](accepted/14_dynamics_milestone_plan.md) | [06_DYNAMICS.md §5, §5.1](../design/06_DYNAMICS.md), [UPDATE_PHASES Phase P11](../UPDATE_PHASES.md) |
| 15 | [packaging_extras_releases](accepted/15_packaging_extras_releases.md) | [conventions/20_PACKAGING.md](../conventions/20_PACKAGING.md) (new normative file) |
| 16 | [optim_wiring_and_matrix_free](accepted/16_optim_wiring_and_matrix_free.md) | [07_RESIDUALS_COSTS_SOLVERS.md §4, §7, §8, §9, §10](../design/07_RESIDUALS_COSTS_SOLVERS.md), [08_TASKS.md §3.1](../design/08_TASKS.md) |
| 17 | [io_ergonomics_and_assets](accepted/17_io_ergonomics_and_assets.md) | [04_PARSERS.md §2.1, §6, §11](../design/04_PARSERS.md), [09_COLLISION_GEOMETRY.md §9.1](../design/09_COLLISION_GEOMETRY.md), [12_VIEWER.md §4.3](../design/12_VIEWER.md) |

## Reading order (if you're reviewing the strategic decisions)

1. Start with **00_principles** — the lens.
2. **RECONCILIATION** — what was kept, changed, or rejected from the
   gpt_plan review.
3. The foundational triplet (01, 02, 03) — they decide the bottom-layer shape.
4. **06** then **04** — the public-API surface.
5. **07–08** — mutable-state and trajectory contracts.
6. **16** — optimisation-stack wiring (cross-reads with 08).
7. **10–12** — documentation, CI, and regression infrastructure.
8. **09**, **14**, **17** — domain / IO plans.
9. **13** and **15** — housekeeping.

## Implementation guidance

Once you understand the *why*, the *how* lives in
[../UPDATE_PHASES.md](../UPDATE_PHASES.md). That document sequences the
proposals into independently-mergeable code phases (P0 → P13) with
acceptance criteria.

## Why this folder still exists

The `accepted/` proposals are kept as historical reference. The
canonical specs cross-link back to them when the spec contains "the
gpt-plan-driven correction" or "an earlier draft proposed X" lines —
the audit trail is part of the explanation.
