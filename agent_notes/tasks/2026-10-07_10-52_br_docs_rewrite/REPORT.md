# REPORT — BetterRobot docs rewrite and legacy removal (2026-10-07_10-52_br_docs_rewrite)

**Outcome.** BetterRobot `dev` now has one API (the state API), one optimiser package (`optim`), and a new
MkDocs Material documentation site of 35 pages whose every Python snippet runs in CI. The mutable `Data`
API, the legacy `optim`/`residuals`/`tasks` packages, the old examples and the Sphinx tree are gone (all in
`trash/`). Net: **8 300 lines added, 51 800 removed** across 494 files (`git diff --shortstat cb58627^..HEAD`);
inside `src/` alone 1 510 added, 11 900 removed.

Branch: `dev` of `/home/rikhat.akizhanov/better/dev/BetterRobot`, commits `cb58627` → `651a156` (7 commits).
BetterHuman `dev` (`/home/rikhat.akizhanov/better/dev/BetterHuman`): `3df98a0`, `109ef11` (rename + test port).
Nothing was pushed or merged.

## Final gates (all run after the last commit)

| Gate | Result |
|------|--------|
| `CUDA_VISIBLE_DEVICES="" uv run pytest tests -q -m "not bench and not cuda"` (BR) | 1131 passed, 3 skipped |
| `uv run pytest tests/bench -m "not cuda"` (BR) | 1 passed |
| `uv run mkdocs build --strict` | 0 warnings (Material prints an informational MkDocs-2.0 banner; not a warning) |
| `uv run ruff check` on touched files | no new errors vs HEAD (≈90 pre-existing errors in untouched files remain) |
| grep for `Data`, `create_data`, `StaleCache`, `KinematicsLevel`, `JointMimic`, `forward_kinematics_raw`, `_fk_matrix`, `bias_forces` in `src tests examples benchmarks README.md CLAUDE.md docs` | empty, except `docs/concepts/design-decisions.md` (history) and `docs/reference/changelog.md` |
| BetterHuman suite (`BETTERHUMAN_MODELS_DIR=$PWD/models uv run pytest tests`) | 297 passed, 1 skipped, 1 failed — the pre-existing 1-ulp `tests/core/test_reduction.py::test_corrective_slice_is_exact_when_dropped_joint_is_at_rest` |

## What the docs are now

`uv run mkdocs serve` and open `http://127.0.0.1:8000`. Diátaxis layout, Material theme, mkdocstrings API
reference (numpy style, `strict`), MathJax, light/dark.

| Section | Pages | Lines | Content |
|---------|------:|------:|---------|
| `docs/index.md` | 1 | 73 | landing with a runnable batched-IK example and the Panda screenshot |
| `getting-started/` | 8 | 630 | installation, first model, forward kinematics, inverse kinematics, trajectories, floating base, visualize, GPU & batching |
| `concepts/` | 12 | 1525 | architecture, model-and-state, joints-bodies-frames, kinematics-and-jacobians, dynamics, lie-and-spatial, io, collision, optimization, time-and-structure, viewer, design-decisions |
| `guides/` | 8 | 826 | load-a-robot, write-a-term, own-the-loop, hard-constraints, fit-a-clip, differentiate, benchmark-and-profile, use-with-betterhuman |
| `contributing/` | 3 | 211 | conventions, testing, extending |
| `reference/` | 3 + 11 API | 414 | glossary, roadmap, changelog; one API page per package |

Figures (`docs/figures/*.svg`, 8): architecture, model_values_state, joints_bodies_frames, lie_spatial, jacobians,
optim_problem, banded_normal, viewer_layers. Screenshots (`docs/images/*.png`, 5): panda_rest, panda_ik,
g1_floating, trajectory_strip (offline rasterizer, generators in `subagents/make_screenshots.py` and
`make_panda_ik.py`), banded_sparsity (matplotlib, `subagents/make_banded_sparsity.py`).

`tests/docs/test_snippets.py` executes every ```python block of every page (and of `README.md`) in order
under a fake viser server, and checks that every referenced image exists: 75 tests.

Public docstrings were rewritten in `optim`, `terms`, `lie`, `spatial`, `collision`, `viewer` and the state API
(`tests/contract/test_docstrings.py` guards coverage).

## Code changes beyond docs

- **Removed** (`trash/`): `src/better_robot/{optim (old), residuals, tasks}`, `data_model/{data,_kinematics_level}.py`,
  `joint_models/mimic.py`, `kinematics/{forward,_fk_matrix}.py`, `dynamics/{rnea,crba,aba,centroidal,_execution}.py`,
  `StaleCacheError`, examples 01–07 (old), Sphinx `docs/`, `benchmarks/{baseline,m2a_dense_assembly}.py`.
- **Renamed** `optim2` → `optim` (both repos); examples renumbered `01_ik.py` … `08_dynamics.py`.
- **State API additions**: `joint_jacobian`, `frame_jacobian_time_variation`, `joint_jacobian_time_variation`,
  `centroidal_momentum`, `forward_kinematics_state(..., check_quaternion_norm=False)`; `dynamics` exports
  `center_of_mass`; root `__all__` is 21 names.
- **Viewer fixes** (found by the docstring agents): `Viewer.record` fits the camera over the whole clip (was frame 0)
  and honours group `offset` offline (`_Shifted` view); `GroundLayer(plane=None)` follows `Theme.up`;
  `Layer.fits_camera` replaces a class-name check. Tests in `tests/viewer/test_core.py`.
- **`Inertia.from_sphere/box/capsule/ellipsoid`** take `dtype`/`device` (were float32-only).
- **Bench tests** `test_mem_watermark`, `bench_solve_ik` were broken (`br.solve_ik` gone); now use `terms.solve_ik`.
- Package guides `src/better_robot/{data_model,kinematics,dynamics}/CLAUDE.md`, root `CLAUDE.md`, `README.md` rewritten.

## Tests dropped (subject no longer exists) — everything else ported with identical tolerances

`tests/contract/test_cache_invariants.py` (whole file: Data cache); `test_compute_fills_joint_jacobians_byproduct`,
`test_velocity_reassignment_invalidates_dot`, `test_missing_velocity_raises` (kinematics cache);
`test_model_create_data`, `test_model_create_data_batch`, `test_data_batch_shape`, `test_data_reset`,
`test_data_clone`; `test_dynamics_wrappers_store_the_execution_batch_in_data`;
`test_fk_and_frame_raw_blocks_reach_q_and_model_values`, `test_joint_jacobians_raw_reaches_q_and_model_values`,
`test_raw_dynamics_reaches_q_and_body_inertias`, `test_joint_jacobians_raw_matches_workspace_pass`,
`test_frame_jacobian_raw_matches_workspace_pass`, `test_frame_placements_raw_matches_public_fk` (raw vs Data);
`test_raw_passes_are_in_package_all` (replaced by `test_raw_kernels_are_not_public`);
`test_data_has_core_fields`, `test_data_legacy_aliases_are_removed`;
`test_direct_joint_mimic_requires_a_concrete_target_kind` (JointMimic placeholder). The fullgraph-compile test
now compiles `_fk_lean.lean_core` (the public wrapper's placement cache probes functorch, which dynamo cannot trace).
Pinocchio parity: all 11 files ported, same tolerances, 0 dropped.

## Review of the docs against the code

Three read-only reviewers covered all 31 narrative pages (~130 prose claims). 15 wrong or stale claims were
found and fixed: the error table in `load-a-robot` (`IRError`, `ModelInconsistencyError` scope, base classes),
float64 parity tolerances, `record` sampling wording, naming/extending tables, `fit-a-clip` solver name,
`differentiate` test claim, `use-with-betterhuman` path and shape-fit snippet, `lie.tangents` names,
`Inertia` dtype claim (fixed in code), `first-model` frames claim, `random_start` scope, roadmap `center_of_mass`
sentence and inventory. Verified correct: LM cost linear in knots (201/226/278/376 ms at 10/20/40/80 knots),
the Warp history numbers, the GPU benchmark table (note: its source JSONs are marked `dirty: True`), the
constrained-solver loop, every joint `nq`/`nv`, the Lie identities numerically.

A writer's claim that `optim.linearize` rejects a windowed term linear in a time-free variable was **not
reproducible** (7 linear forms linearize fine); the check at `problem.py:576` fires only for ops without a
second derivative, as intended.

## Not done / for the owner

- ≈90 pre-existing ruff errors in files this task did not touch (`io/parsers/urdf.py`, `benchmarks/run.py`,
  `tests/spatial`, `spatial/force.py` F821).
- `tests/kinematics/test_fk_contracts.py` (ex `test_fk_matrix.py`) — the compile parity test is marked `slow`.
- The GitHub Pages URL in `README.md` assumes the site is published from `dev` later.
- `trash/` holds everything removed; empty it when satisfied.
