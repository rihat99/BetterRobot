# Design contracts

A *contract* is a rule the code must follow so the library grows without refactoring. Every rule has an ID
(`C-<PAGE>-<n>`), a strength (**MUST** = never violated; **SHOULD** = violated only with a one-line reason in the
code review), a short rationale, a good/bad example from this codebase, and an *enforcer*: a `ruff` code, a test
under `tests/contract/`, or `[review]` (a human or agent reviewer checks it by hand).

Read the pages in order once; afterwards use the matrix below to find the rule behind a failing check.

| Page | Covers |
|------|--------|
| [01 Naming](01-naming.md) | monogram notation in math code, descriptive names elsewhere, units |
| [02 Tensors](02-tensors.md) | batch axes, broadcasting, dtype/device, shapes in docstrings, `optim` batching |
| [03 State and mutability](03-state-mutability.md) | frozen model and state, no input mutation, `Problem.with_` views, `Model.to` and caches |
| [04 Differentiability](04-differentiability.md) | autograd and `torch.func` safety |
| [05 API shape](05-api-shape.md) | functions vs classes, keyword-only args, ids |
| [06 Validation and errors](06-validation-errors.md) | where to validate, opt-in value checks, the exception tree, string-wired reads |
| [07 Dependencies and layering](07-dependencies-layering.md) | package DAG, lazy optional deps |
| [08 Extension points](08-extension-points.md) | terms, joint types, loaders, adding solvers/layers/packages, deprecation, pre-1.0 versioning |
| [09 Performance](09-performance.md) | no host sync, batch not loop, benchmarks |
| [10 Docs and tests](10-docs-tests.md) | docstring format, no Sphinx roles, executable snippets |
| [Decisions](decisions/index.md) | architecture decision records (ADRs) |

## Enforcement matrix

| Rule | Enforcer |
|------|----------|
| C-NAME-1 | `[ruff:N803,N806]` [planned: L-7], `[review]` |
| C-NAME-2 | `tests/contract/test_naming.py`, `[ruff:N801,N802,N803,N806,N815,N816]` [planned: L-7] |
| C-NAME-3 | `[review]` |
| C-NAME-4 | `[review]` |
| C-NAME-5 | `tests/contract/test_se3_vocabulary.py` |
| C-TENS-1 | `tests/contract/test_dtype_device.py` |
| C-TENS-2 | `tests/contract/test_dtype_device.py`, `[review]` |
| C-TENS-3 | `tests/contract/test_docstrings.py` |
| C-TENS-4 | `tests/contract/test_hot_path_lint.py` |
| C-TENS-5 | `tests/optim/test_problem.py`, `[review]` |
| C-STATE-1 | `tests/contract/test_frozen.py` |
| C-STATE-2 | `tests/contract/test_frozen.py`, `[review]` |
| C-STATE-3 | `tests/contract/test_no_inplace.py` |
| C-STATE-4 | `[review]` |
| C-STATE-5 | `tests/optim/test_problem.py`, `[review]` |
| C-STATE-6 | `tests/contract/test_dtype_device.py`, `[review]` |
| C-DIFF-1 | `tests/lie/test_torch_backend_gradcheck.py`, `[review]` |
| C-DIFF-2 | `tests/autograd/test_vmap.py` [planned: L-8], `[review]` |
| C-DIFF-3 | `tests/lie/test_torch_backend_gradcheck.py`, `[review]` |
| C-API-1 | `[review]` |
| C-API-2 | `tests/contract/test_keyword_only.py` |
| C-API-3 | `[ruff:FBT001,FBT002]` |
| C-API-4 | `tests/contract/test_public_api.py`, `[review]` |
| C-API-5 | `tests/contract/test_public_api.py`, `tests/contract/test_submodule_public_imports.py` |
| C-ERR-1 | `tests/contract/test_boundary_validation_count.py`, `[review]` |
| C-ERR-2 | `tests/contract/test_public_api.py`, `[review]` |
| C-ERR-3 | `[review]` |
| C-ERR-4 | `[ruff:BLE001,E722]`, `[review]` |
| C-ERR-5 | `tests/optim/test_problem.py::test_unknown_read_fails_at_registration` |
| C-DEP-1 | `tests/contract/test_layer_dependencies.py` |
| C-DEP-2 | `tests/contract/test_optional_imports.py`, `[ruff:PLC0415]` |
| C-DEP-3 | `[ruff:TID251]` |
| C-DEP-4 | `[review]` |
| C-EXT-1 | `tests/contract/test_protocols.py`, `[review]` |
| C-EXT-2 | `tests/contract/test_protocols.py`, `[review]` |
| C-EXT-3 | `tests/contract/test_public_api.py`, `tests/contract/test_submodule_public_imports.py`, `[ruff:SLF001]` [planned: L-7] |
| C-EXT-4 | `[review]` |
| C-EXT-5 | `tests/contract/test_protocols.py`, `tests/contract/test_layer_dependencies.py`, `[review]` |
| C-EXT-6 | `[review]` |
| C-PERF-2 | `[review]` |
| C-PERF-3 | `[review]` (benchmarks under `tests/bench/` are advisory) |
| C-PERF-4 | `[review]` |
| C-DOC-1 | `tests/contract/test_docstrings.py`, `[ruff:D]` [planned: L-7] |
| C-DOC-2 | `tests/docs/test_snippets.py` |
| C-DOC-3 | `tests/contract/test_contract_matrix.py` |
| C-DOC-4 | `tests/contract/test_docstrings.py` [planned: L-9], `[review]` |
| C-TEST-1 | `[review]` |
| C-TEST-2 | `[ruff:ANN001,ANN201]` [planned: L-7 for ANN001] |

This table is generated from the pages' `Enforcement:` paragraphs and checked by
`tests/contract/test_contract_matrix.py`. An enforcer tagged `[planned: L-<n>]` on its page is a ledger entry in
the task notes, not a soft rule; a rule without an enforcer is a bug in this document.

## When to write an ADR

Write an architecture decision record (ADR) under `decisions/` when a change does any of the following:

- adds, removes or re-orders a package in the import DAG (`07`);
- changes a public data layout (pose order, twist order, batch convention, `nq`/`nv` semantics);
- adds a new extension mechanism (registry, protocol, plugin hook) or a new optional dependency;
- replaces a dependency the public API exposes (for example the viewer backend);
- rejects a contract rule for a whole package.

A bug fix, a new term, a new loader for an existing registry, or a rename inside one package does not need one.
Accepted ADRs are never edited except their Status line; write a new ADR that supersedes the old one.
