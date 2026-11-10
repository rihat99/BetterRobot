# Test suite

What each group of tests guards. How to run them: [Run the tests](../guides/run-the-tests.md).

## Markers

| Marker | Meaning |
|--------|---------|
| `slow` | takes seconds; deselected in the fast tier, run in the full tier |
| `cuda` | needs a CUDA device; deselected in the fast and full tiers, run in the release tier |

## Pinocchio parity

`tests/parity/pinocchio/` compares kinematics, Jacobians and dynamics against Pinocchio on the same
URDFs; it is part of the full tier and runs alone with `uv run pytest tests/parity/pinocchio -q`. The tolerances
are part of the contract:

| Inputs | Typical `atol` |
|--------|----------------|
| float64 | `1e-12` to `1e-9` for kinematics and Jacobians; up to `2e-6` for a few dynamics cases (`test_rnea_missing_joint_kinds.py`, `test_centroidal.py`) |
| float32 | `2e-6` |

A parity failure means the library changed its answer. Fix the code, not the tolerance.

## Contract tests

`tests/contract/` guards properties that no unit test sees:

| Test | Guards |
|------|--------|
| `test_layer_dependencies.py` | every import between packages is in the `ALLOWED` edge table (`optim` imports only `exceptions`) |
| `test_public_surface.py` | the root `__all__` is the decided list; every name of the `__all__` tree resolves |
| `test_submodule_public_imports.py` | only the decided packages exist; no raw `*_raw` kernel is public |
| `test_docstrings.py` | every public object has a docstring that gives each tensor parameter's shape |
| `test_boundary_validation_count.py` | public kinematics, dynamics, lie and spatial functions check each tensor input once; raw passes below do not re-check |
| `test_hot_path_lint.py` | `tools/hot_path_lint.py` finds no `torch.compile`-breaking pattern (host syncs, `.item()`, Python branching on tensors) in the hot paths of contract 09 (kinematics, dynamics, lie, spatial, terms, and the solver step with the linearization and linear algebra it runs), which it keeps equal to the lint's watched list |
| `test_naming.py` | a `Model` or `KinematicState` field is plural exactly when its shape has an axis over joints, bodies or frames; integer indices end in `_ids` |
| `test_vocabulary.py` | one name per thing (no alias, no class name a library below uses), acronyms in capitals, no `utils`-style module; every word of every public name is in the glossary |
| `test_ruff_ratchets.py` | no new ruff naming (`N`), docstring (`D1`), annotation (`ANN001`) or other-object private access (`SLF001`) site in `src/` |
| `test_pyright_ratchet.py` | no new pyright error in `src/` (full tier: pyright takes seconds) |
| `test_optional_imports.py` | each package imports only the outside packages `pyproject.toml` declares for it; `lie`, `spatial` and `optim` import torch only |
| `test_no_skips.py` | the `no_skips` plugin fails every skip and xfail, whatever spells it |
| `test_pytest_config.py` | pytest runs strictly: unknown markers and config keys, passing xfails, empty parameter sets and new warnings fail |
| `test_python_version.py` | `.python-version`, `requires-python` and ruff's target agree on Python 3.12 |
| `test_fixtures.py` | every test robot has its licence and provenance |
| `test_repository_layout.py` | the tracked top-level entries are the standard set; each `src` package has its `tests/` folder and every other `tests/` folder is a named suite; no tracked symlink and no tracked file over 1 MiB (record [0039](../contracts/decisions/0039-test-robot-fixtures.md)); no directory under `src/`, `tests/` or `tools/` without a tracked file |
| `test_citations.py` | every repository path and `path::symbol` cited in tracked text resolves; no line number is cited; a docs page outside the changelog, the migration pages and the records tells no history and names no plan code |
| `test_agent_files.py` | the root `CLAUDE.md` is at most 200 lines and a nested one 60; no capitals for emphasis in an agent file or skill; `AGENTS.md` is one line naming `CLAUDE.md`; the shared `.claude/` files are tracked and the rest ignored |
| `test_extension_points.py` | every extension point is a base class or a plain function: no `register_*` function, no module-level mutable registry (see [Extend BetterRobot](../guides/extend.md)) |
| `test_abstract_or_final.py` | every class in `src` is abstract or `@final` |
| `test_api_snapshot.py` | the public API equals its snapshot, `tests/contract/public_api.json` |
| `test_contract_matrix.py` | the enforcement matrix of the contracts index matches what every contract page says |
| `test_coupling_standard.py` | every line of the [coupling standard](../contracts/coupling-standard.md) against the library; the page's lines only grow (frozen copy `coupling_standard_lines.txt`) |
| `test_decision_records.py` | every decision record follows the template, has an allowed status and an index row, and an accepted record keeps its accepted text |
| `test_dtype_device.py`, `test_no_inplace.py`, `test_state_attachment.py` | public ops keep the input dtype and device, give each batch element its own result, refuse another dtype, never write their inputs, and refuse a state of another model; their results are the [effects table](effects.md) |
| `test_error_tree.py` | every error the library raises is in one tree under `BetterRobotError` |
| `test_frozen.py`, `test_model_fields.py`, `test_hashing.py`, `test_caches.py` | public records are frozen, `Model` is one flat record, equality and `structure_key` follow record 0010, caches and plans follow records 0011 and 0040 |
| `test_identifiers.py`, `test_keyword_only.py`, `test_name_words.py`, `test_se3_vocabulary.py` | a frame, joint or body parameter takes `int \| str`; at most two positional arguments; layouts, units and SE(3) words in public names |
| `test_internal_checks.py`, `test_no_probing.py` | one switch guards every check that reads tensor values; no `getattr` with a literal and no `hasattr` in `src` |
| `test_performance_rules.py` | a batch of two and a batch of seven dispatch the same torch operators (slow); `src` compiles nothing and builds no kernel |
| `test_compile_counters.py` | the graph breaks, recompiles, operators and host syncs of every counter row of `benchmarks/run.py` stay at or below `benchmarks/thresholds/` on CPU (slow); each hot path contract 09 names has counter rows |
| `test_public_differentiability.py` | gradients reach `q` and the model tensors through every public kinematics and dynamics function (see [Differentiate](../guides/differentiate.md)) |
| `test_repr.py` | every public type has a one-line repr under 200 characters that names the type |
| `test_versioning.py` | the version is `0.MINOR.PATCH` before 1.0, and this major keeps no deprecated alias |
| `test_roadmap_stub_inventory.py` | every `NotImplementedError` stub is listed in the [roadmap](roadmap.md) |

A rule that still has known violations keeps them in a ratchet, `tests/contract/ratchets/<name>.json`: each key
is one violation with its reason or witness, or `{"count": n, "reason": ...}` when one key holds several sites (a
symbol with three pyright errors). A ratchet only shrinks: a key with more sites than its count fails, and the fix
that removes a violation lowers or removes its entry, which a stale-entry test enforces. `tools/ratchet_counts.py`
walks each ratchet's mainline history and the working tree and fails when a ratchet's or a key's count rose or a key
appeared that an earlier step did not have; `test_ratchet_history.py` runs it in the full tier, beside planted
histories that it must fail.

## Docs tests

`tests/docs/test_snippets.py` runs every `python` fence of each page (and `README.md`) once, in order, in one
namespace, on CPU, seeded (`torch`, `random` and `numpy` at 0), with a fake viewer server, 120 s per page, through
`better_robot.testing.snippets.check_page`. Every `output` fence is compared in full with what the fence above it
prints, a fence that prints needs one, and a page fails when the number of compared outputs differs from its number
of `output` fences or when a Python fence is tagged ```` ```py ````, ```` ```pycon ```` or `~~~`, which the runner would not read. It also
checks that every image, poster and clip a page references exists; `tests/docs/test_readme_first_screen.py` checks
that the README's first Python fence and the output fence under it close by line 40, and
`tests/docs/test_effects_table.py` reruns the contract tests behind the [effects table](effects.md) and
compares the page with a fresh render, and `tests/docs/test_codemap.py` checks that the layer section of
[Architecture](../concepts/architecture.md) lists the `ALLOWED` edge table, and `tests/docs/test_contracts_map.py`
checks that the contracts map figure shows each rule family's kinds of check as the contracts index gives them. The rules for a page:

| Fence | Meaning |
|---|---|
| ```` ```python ```` | runs in the full tier, in page order |
| ```` ```python { .no-run data-reason="..." } ```` | shown but not run; the reason is required (a GPU, a live server, licensed data). The fences of `docs/contracts/` illustrate rules and records and opt out together |
| ```` ```output ```` | docked under a fence that runs: its real output. A line `...` stands for any number of lines; a number with fraction digits matches within half a unit of its last printed digit, a float printed without them (`0.`, `1e-06`) exactly; all other text matches exactly |
