# 05 Results — Rewrite the documentation

**Status:** complete on `dev` (2026-07-18).

## Delivered

- Rebuilt the hand-written documentation around tutorials, task-focused
  guides, explanation, conventions, and reference material. The front page
  now gives readers a short route into each quadrant.
- Added the missing beginner path from model anatomy through FK, IK,
  floating-base models, and batched GPU work. Added practical guides for
  loading, visualization, differentiation, custom residuals, and owning an
  optimization loop.
- Added plain-language explanations of FK, IK, Jacobians and their three
  construction methods, quaternions, SE(3), tangents, least squares, LM, and
  batching. The glossary now covers the vocabulary used by the tutorials.
- Kept design tradeoffs explicit and linked chapters to the relevant
  decisions. Claims were checked against the post-Task-04 source rather than
  copied from older documentation.
- Classified every published Python example as executable. Nineteen Sphinx
  doctests and a 12-snippet pytest harness cover the tutorial and guide code;
  the exact front-page example remains covered by its existing marker test.
- Regenerated the API reference policy so private root modules are omitted.
  The three accidentally generated private pages were removed.
- Updated documentation paths in source docstrings, errors, and test comments.
  These are mechanical documentation-link edits; runtime and numerical
  behavior did not change.

## Verification

| Check | Result |
|---|---|
| Full non-benchmark/non-CUDA gate | `1468 passed, 2 skipped, 16 deselected, 38 warnings` |
| Documentation + vertical-slice focus | `30 passed, 18 warnings` |
| Fresh Sphinx doctest build | `19 tests, 0 failures` |
| Fresh Sphinx HTML build | complete tree; only four expected offline-intersphinx warnings |
| Banned-process-language audit | zero hits |
| Renamed/deleted-page reference audit | zero hits |
| Ordinary Python fences | one: the executable front-page marker example |
| Task-scoped Ruff, format, and diff checks | clean |
| Roadmap explicit-raise inventory | unchanged byte-for-byte |

CUDA-marked tests were not run, per the standing owner-run rule.

## Line accounting

- Hand-written Markdown under `docs/`, excluding generated API pages and
  `_build`: 41 files / 8,164 lines before, 43 files / 4,550 lines after; net
  **−3,614 lines**.
- Generated API reference: 0 additions, 113 deletions, net **−113 lines**
  from removing three private-module pages.
- Entire Task-05 commit, including tests, link-only source edits, plan log,
  and this report: **4,140 additions, 7,641 deletions, net −3,501**.

## Per-page disposition

| Area | Disposition |
|---|---|
| Admin | `index` rewritten; `README` and `CHANGELOG` kept and rewritten where needed. |
| Getting started | `index` and `installation` rewritten; old FK, IK, and floating-base pages rewritten as numbered `02`, `03`, and `04`; new `01_robot_model` and `05_batched_gpu`. |
| Guides | plural `custom_residuals` replaced by singular `custom_residual`; new `load_a_robot`, `visualize`, `differentiate_through_kinematics`, and `own_your_optimization_loop`; index rewritten. |
| Concept entry | `vision` rewritten as `why_betterrobot`; `design_decisions` retained and extended; index rewritten. |
| Concept core | `architecture`, `model_and_data`, `joints_bodies_frames`, `lie_and_spatial`, `dynamics`, `parsers_and_ir`, and `viewer` retained and rewritten. |
| Concept merges | `kinematics` rewritten as `kinematics_and_jacobians`; `residuals_and_costs` + `solver_stack` merged into `residuals_costs_and_solvers`; `batching_and_backends` + `warp_bridge` merged into `the_compute_seam`. |
| Concept removals | `collision_and_geometry` moved to reference; `tasks` deleted after its useful material was distributed to tutorials and the residual/solver chapter. |
| Conventions | Index and all eight named convention pages retained and rewritten; `engineering` merged into `contracts` and deleted. |
| Reference | Index, glossary, roadmap, and changelog wrapper rewritten; collision note added; generated API refreshed; `named_block_solvers` merged into the solver concept and deleted. |

## Deviations and findings

1. The requested `sphinx-docs` and `diataxis-docs` skills were not available
   in this session. The accepted structure and validation gates were followed
   directly.
2. The plan's approximate 24-page target conflicts with its explicit required
   file list. The checked-in baseline had 41 hand-written Markdown files; the
   explicit disposition produces 43 including section indexes and admin
   pages. The explicit file-by-file requirements were treated as authoritative
   while the corpus still shrank by 3,614 lines.
3. Sphinx reports exactly the four allowed warnings because the sandbox cannot
   fetch the Python, PyTorch, NumPy, and trimesh intersphinx inventories.
4. Optional MJCF and interactive-viewer commands require extras unavailable in
   the test environment. Their public setup/API paths are executable-tested;
   environment-dependent launch commands are shown as non-executable text.
5. The plan called analytic Jacobians universally exact. Shipped residual
   blocks include documented approximations, so the final wording says exact
   where derived exactly and names approximations where they exist.
6. The private-module autodoc skip expression did not match private modules at
   the package root. It was corrected, and only the three private generated
   pages disappeared on regeneration.
7. Existing user edits in `src/better_robot/optim/manifolds.py` and
   `tests/optim/test_manifolds.py` were preserved and excluded from this task.

No requested Task 05 work remains incomplete.
