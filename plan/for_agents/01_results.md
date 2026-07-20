# Order 01 results — simplify

Status: complete on `dev` (2026-07-20).

## Delivered

- Added one-shot `AutodiffFallbackWarning` diagnostics for automatic residual
  AD and missing temporal numeric blocks; explicit strategies stay silent.
- Removed the zero-caller LM step limits, LU solver, variable mask/scale
  subsystem, and standalone optimizer manifolds module. Moved `Bounds` into
  `variables.py` and consolidated optimizer helpers in `utils.py`.
- Skipped the projected-gradient candidate for unbounded LM solves. The bounds
  flag is refreshed once at state initialization, including restarted solves.
- Restructured implicit differentiation around one terminal-state record and
  four named sections without changing its public API or numerical guards.
- Inlined the Lie implementation facade into `so3.py` and `se3.py`; `so3.py`
  is 306 lines and public operations passed bitwise value/gradient parity.
- Removed the unused dynamics derivative/integrator modules and task
  parameterization surface, simplified IK configuration validation, and
  consolidated task helpers.
- Removed Node epoch state, retained nested evaluation-scope behavior, and
  consolidated residual protocols/helpers in `residuals/utils.py`.
- Updated migration entries, concepts, extension guidance, package contracts,
  generated API pages, and tests with every public removal.

## Verification

- Full CPU gate: **1,537 passed, 2 skipped, 16 deselected**.
- CUDA gate on GPU 2: **15 passed, 1,540 deselected**.
- Pinocchio parity: **135 passed**.
- Architecture/API contracts: **344 passed**.
- Documentation tests: **27 passed**; Sphinx doctest: **30 passed**.
- Changed Python files: Ruff check and format check passed (**63 files**).
- `git diff --check` and source compilation passed.
- Sphinx emitted four offline intersphinx-inventory warnings; no documentation
  test failed.

The new fallback warning accounts for the expected warning increase in tests
whose custom residuals intentionally rely on automatic differentiation. The
built-in analytic IK graph remains silent in its regression test.

## Deletion evidence

Scoped source and non-generated-doc greps returned zero live references for
`block_step_limits`, `class LU`, variable `mask=`/`scale=` arguments, removed
tangent helpers, `optim.manifolds`, `lie._impl`, dynamics derivative/integrator
modules, `tasks.parameterization`, `residuals._variables`, Node epoch state,
and the removed L-BFGS spellings. `MIGRATION.md` intentionally retains old
public spellings. Generic conceptual words such as “derivatives” and
“manifolds” remain where they describe mathematics rather than removed APIs.

## Physical source-line accounting

| Scope | Before | After | Delta |
|---|---:|---:|---:|
| `optim` | 3,777 | 3,582 | −195 |
| `lie` | 1,265 | 1,064 | −201 |
| `dynamics` | 1,346 | 1,200 | −146 |
| `tasks` | 1,571 | 1,377 | −194 |
| `residuals` | 2,628 | 2,598 | −30 |
| `_warp_kernels.py` | 310 | 296 | −14 |
| **All `src/**/*.py`** | **21,464** | **20,684** | **−780** |

The hard limit of 20,700 lines is met with 16 lines of headroom.

## Deviations and findings

- `implicit.py` ended at 439 lines rather than the approximate 360-line target.
  Further compression would have obscured explicit KKT and active-set guards;
  the hard repository budget and all implicit-differentiation tests pass.
- `lie/` ended at 1,064 lines rather than approximately 1,000. The requested
  facade was still fully removed, and `so3.py` meets the 250–310-line target.
- Ruff formatting of the already-touched Warp kernel file removed 14 physical
  lines, and one unused local in `lie/tangents.py` was deleted. Both are
  behavior-neutral ancillary cleanup beyond the written tasks.
- Adversarial review found that the first implementation cached `_has_bounds`
  in the optimizer constructor. It was moved to `_init_state` and guarded by a
  late-bounds regression test before the final gates.
