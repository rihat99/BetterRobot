# Design contracts

A *contract* is a rule the code must follow so the library grows without refactoring. Every rule has an ID
(`C-<PAGE>-<n>`), a strength (**MUST** = never violated; **SHOULD** = violated only with a one-line reason in the
code review), a short rationale, a good/bad example from this codebase, and an *enforcer*: a `ruff` code, a test
under `tests/contract/`, or `[review]` (a human or agent reviewer checks it by hand).

Read the pages in order once; afterwards use the matrix below to find the rule behind a failing check.

| Page | Covers |
|------|--------|
| [Scope](scope.md) | what the library does, its non-goals, the bar for inclusion, commonly rejected ideas |
| [Public surface](public-surface.md) | what is public, what users may do, horizons, compatible changes |
| [Coupling standard](coupling-standard.md) | the shapes, layouts, identifiers and file formats other repositories read; changes only by addition |
| [01 Naming](01-naming.md) | monogram notation in math code, descriptive names elsewhere, layout words, SI units |
| [02 Tensors](02-tensors.md) | batch axes, broadcasting, dtype/device, shapes in docstrings, `optim` batching |
| [03 State and mutability](03-state-mutability.md) | frozen model and state, no input mutation, `Problem.with_` views, `Model.to` and caches |
| [04 Differentiability](04-differentiability.md) | autograd and `torch.func` safety |
| [05 API shape](05-api-shape.md) | functions vs classes, keyword-only args, ids |
| [06 Validation and errors](06-validation-errors.md) | where to validate, opt-in value checks, the exception tree, string-wired reads |
| [07 Dependencies and layering](07-dependencies-layering.md) | package DAG, lazy optional deps |
| [08 Extension points](08-extension-points.md) | terms, joint kinds, loaders, adding solvers/layers/packages, no deprecated alias, pre-1.0 versioning |
| [09 Performance](09-performance.md) | no host sync, batch not loop, no compiler or kernel in the library, benchmarks |
| [10 Docs and tests](10-docs-tests.md) | docstring format, no Sphinx roles, executable snippets |
| [Decisions](decisions/index.md) | architecture decision records (ADRs) |

## Enforcement matrix

| Rule | Enforcer |
|------|----------|
| C-NAME-1 | `tests/contract/test_ruff_ratchets.py`, `[review]` |
| C-NAME-2 | `tests/contract/test_ruff_ratchets.py`, `tests/contract/test_naming.py`, `tests/contract/test_vocabulary.py`, `tests/contract/test_vocabulary.py::test_glossary_words` |
| C-NAME-3 | `tests/contract/test_name_words.py::test_layout_words_name_converters_only` |
| C-NAME-4 | `tests/contract/test_name_words.py::test_public_quantities_are_in_si_units`, `tests/contract/test_name_words.py::test_frame_means_a_robot_frame_in_the_viewer` |
| C-NAME-5 | `tests/contract/test_se3_vocabulary.py` |
| C-TENS-1 | `tests/contract/test_dtype_device.py` |
| C-TENS-2 | `tests/contract/test_dtype_device.py`, `[review]` |
| C-TENS-3 | `tests/contract/test_docstrings.py`, `[review]` |
| C-TENS-4 | `tests/contract/test_hot_path_lint.py` |
| C-TENS-5 | `tests/optim/test_problem.py`, `[review]` |
| C-STATE-1 | `tests/contract/test_frozen.py` |
| C-STATE-2 | `tests/contract/test_frozen.py`, `[review]` |
| C-STATE-3 | `tests/contract/test_no_inplace.py` |
| C-STATE-4 | `tests/contract/test_caches.py`, `[review]` |
| C-STATE-5 | `tests/optim/test_problem.py::test_problem_is_immutable`, `tests/optim/test_problem.py::test_callbacks_cannot_register_on_the_views_run_hands_them` |
| C-STATE-6 | `tests/contract/test_caches.py`, `tests/contract/test_dtype_device.py`, `tests/contract/test_state_attachment.py`, `[review]` |
| C-DIFF-1 | `tests/contract/test_public_differentiability.py`, `tests/lie/test_torch_backend_gradcheck.py`, `[review]` |
| C-DIFF-2 | `tests/contract/test_hot_path_lint.py`, `[review]` |
| C-DIFF-3 | `tests/lie/test_torch_backend_gradcheck.py`, `[review]` |
| C-API-1 | `tests/contract/test_frozen.py`, `[review]` |
| C-API-2 | `tests/contract/test_keyword_only.py` |
| C-API-3 | `[ruff:FBT001,FBT002]` |
| C-API-4 | `tests/contract/test_identifiers.py` |
| C-API-5 | `tests/contract/test_public_surface.py`, `tests/contract/test_submodule_public_imports.py` |
| C-ERR-1 | `tests/contract/test_boundary_validation_count.py`, `[review]` |
| C-ERR-2 | `tests/contract/test_error_tree.py`, `[review]` |
| C-ERR-3 | `tests/contract/test_error_tree.py::test_unknown_name_lists_the_valid_names`, `[review]` |
| C-ERR-4 | `[ruff:BLE001,E722]`, `[review]` |
| C-ERR-5 | `tests/optim/test_problem.py::test_unknown_read_fails_at_registration` |
| C-DEP-1 | `tests/contract/test_layer_dependencies.py`, `tests/contract/test_optional_imports.py` |
| C-DEP-2 | `tests/contract/test_optional_imports.py`, `[ruff:PLC0415]` |
| C-DEP-3 | `[ruff:TID251]` |
| C-DEP-4 | `tests/contract/test_optional_imports.py`, `[review]` |
| C-EXT-1 | `tests/contract/test_extension_points.py`, `[review]` |
| C-EXT-2 | `tests/contract/test_abstract_or_final.py`, `tests/model/test_joints.py`, `[review]` |
| C-EXT-3 | `tests/contract/test_public_surface.py`, `tests/contract/test_submodule_public_imports.py`, `tests/contract/test_ruff_ratchets.py` |
| C-EXT-4 | `tests/contract/test_versioning.py::test_src_emits_no_deprecation_warning` |
| C-EXT-5 | `tests/contract/test_layer_dependencies.py`, `[review]` |
| C-EXT-6 | `tests/contract/test_versioning.py::test_version_is_pre_1_0`, `tests/contract/test_api_snapshot.py` |
| C-EXT-7 | `tests/contract/test_extension_points.py::test_normal_equations_come_from_a_sink` |
| C-EXT-8 | `tests/collision/test_collision.py::test_a_new_obstacle_shape_is_only_a_function`, `[review]` |
| C-PERF-2 | `tests/contract/test_performance_rules.py::test_a_batch_dispatches_the_same_operators_at_every_size` |
| C-PERF-3 | `tests/contract/test_compile_counters.py` |
| C-PERF-5 | `tests/contract/test_performance_rules.py::test_src_compiles_nothing_and_builds_no_kernel` |
| C-DOC-1 | `tests/contract/test_docstrings.py`, `tests/contract/test_ruff_ratchets.py` |
| C-DOC-2 | `tests/docs/test_snippets.py` |
| C-DOC-3 | `tests/contract/test_contract_matrix.py` |
| C-DOC-4 | `tests/contract/test_docstrings.py` |
| C-TEST-1 | `tests/contract/test_no_skips.py`, `[review]` |
| C-TEST-2 | `[ruff:ANN201]`, `tests/contract/test_ruff_ratchets.py`, `tests/contract/test_pyright_ratchet.py` |

This table repeats the pages' `Enforcement:` paragraphs, and `tests/contract/test_contract_matrix.py` fails when the
two differ: every enforcer exists today (no planned one), and a rule that only `[review]` checks waits in
`tests/contract/ratchets/review_only.json` for a test or its deletion (the file is empty).

## When to write an ADR

Write an architecture decision record (ADR) under `decisions/` when a change does any of the following:

- adds, removes or re-orders a package in the import DAG (`07`);
- changes a public data layout (pose order, twist order, batch convention, `nq`/`nv` semantics);
- adds a new extension mechanism (registry, protocol, plugin hook) or a new optional dependency;
- replaces a dependency the public API exposes (for example the viewer backend);
- rejects a contract rule for a whole package.

A bug fix, a new term, a new loader for an existing registry, or a rename inside one package does not need one.
Only the owner accepts an ADR: one the integrator writes is `Proposed (integrator, awaiting owner)` until then. An
accepted ADR changes only in its Status line; a new ADR supersedes it or amends part of it.
