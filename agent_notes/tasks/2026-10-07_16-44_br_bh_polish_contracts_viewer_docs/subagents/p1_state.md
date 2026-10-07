# P1 state — second worker stopped on context budget (2026-10-07)

## Done (BetterRobot, /home/rikhat.akizhanov/better/dev/BetterRobot)
- `docs/contracts/00-index.md` … `10-docs-tests.md` (11 pages, 36 rules) + `decisions/` (index + 5 ADRs). All snippets
  fenced ```python no-run (the token `tests/docs/test_snippets.py` recognises); `uv run pytest tests/docs` 90 passed.
- `mkdocs.yml`: Contracts nav; `uv run mkdocs build --strict` passes after the fold-in.
- Fold-in: `docs/contributing/conventions.md` and `docs/concepts/design-decisions.md` trashed to `trash/` (trash.py),
  nav lines removed; only `mkdocs.yml` had linked them. `docs/contributing/extending.md` now links
  `contracts/00-index.md` and `contracts/08-extension-points.md`.
- C-EXT-2 in `08-extension-points.md` rewritten: there is no `JOINT_TYPES` registry; joints satisfy the structural
  `JointModel` protocol (`data_model/joint_models/base.py`) and only parsers register (`io.register_parser`).
- Enforcer citations re-pointed to the existing tests instead of duplicates: import DAG -> `test_layer_dependencies.py`,
  optional deps -> `test_optional_imports.py`, docstrings -> `test_docstrings.py`.
- New `tests/contract/test_keyword_only.py` (C-API-2; `KEYWORD_ONLY_ALLOWLIST`, 30 entries incl. re-exports, with
  file:line) and `tests/contract/test_frozen.py` (C-STATE-1; `MUTABLE_ALLOWLIST`: 6 lifecycle objects accepted,
  3 LEDGER items `optim.Phase`, `optim.Result`, `viewer.MeshSpec`). Both pass; ruff clean on both.
- `p1_violations.md` complete: ruff 25, keyword-only table, non-frozen dataclasses, missing enforcers, p0 §3 items.
- `00-index.md` marks `test_dtype_device.py` and `test_no_inplace.py` as "(pending, ledger)".
- ruff: pyproject rules unchanged from first worker. `ruff check src` 58 (= 33 baseline + 25 contract findings);
  `ruff check src tests` 107 (tests 49 vs 47 baseline; the two new contract tests are clean, the delta is in
  files other agents are trimming).

## Remaining (in brief order)
1. (done) Full BR suite: 1421 passed, 10 skipped, 97 deselected (`-m "not bench and not cuda"`, docs run separately: 90 passed).

2. `tests/contract/test_dtype_device.py` (C-TENS-1/2) and `test_no_inplace.py` (C-STATE-3, C-DIFF-2/3): need a small
   entry-point registry (model fixture + per-function argument factory). Then drop "(pending, ledger)" in 00-index.
3. Ruff: after polish clears the 25 findings, consider moving N, D, ANN001, SLF001 out of the pyproject comment block.
4. BetterHuman addendum (`/home/rikhat.akizhanov/better/dev/BetterHuman`): `docs/contracts/index.md`, `bh-addendum.md`
   (families/assets layout, model loading, `BETTERHUMAN_MODELS_DIR`, studio = thin shell over BR viewer, data layout
   `<data>/{segmentations,frames,poses}/<key>/`); fold `docs/contracts.md` + `docs/design_principles.md`; adapt
   `test_layer_dependencies.py` and `test_keyword_only.py` into BH `tests/contract/`; run BH suite. Nothing done.

## Open questions
- `02-tensors.md` cites `test_hot_path_lint.py` for C-TENS-4; its package list was not checked against
  (lie, spatial, kinematics, dynamics, optim, terms).
- Orchestrator reported 8 `tests/docs/test_snippets.py` failures on contract pages from another agent's run; they do
  not reproduce in this worktree (90 passed). Possibly a different checkout or a concurrent edit of the test.

## P1c (2026-10-07) — review items applied; remaining
Done: items 1-13 (see P1c final message). `tests/contract/test_contract_matrix.py` passes; 00-index matrix is now
generated from the pages' `Enforcement:` paragraphs (one row per rule, 42 rules). Ledger entries L-1..L-8 in
p1_violations.md.
Remaining (budget ran out): 14a,b,c,d,f,g,h (one short rule each; new rules must be added to the matrix — regenerate
with the snippet in the P1c message or add the row by hand, enforcer `[review]`), 14e (ADR 0006 stub, Status:
Proposed, "serialization/format versioning of Model and IR"), 15 (06:5 "never about values" -> opt-in value checks
behind a keyword flag defaulting off, cite `kinematics/state.py:151 check_quaternion_norm`), 16 (explain "Array API
broadcasting" at 02:5; add "o" origin and Jacobian prefix to the monogram legend at 01:10-11; C-DIFF-4 and C-PERF-1
say "restates C-STATE-3 / C-TENS-4" explicitly).
