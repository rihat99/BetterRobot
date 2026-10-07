# P1 brief — write the design contracts (BetterRobot, then a BetterHuman addendum)

Read `context.md`, then `p0_synthesis.md`, `p0_architecture_br.md`, `p0_docs_refs.md` (Part 2), and skim
`p0_viewer.md` §1 and `p0_bh_studio.md` "Opinion". Then read the existing docs you will replace:
`docs/contributing/conventions.md`, `docs/contributing/extending.md`, `docs/concepts/design-decisions.md`,
`docs/concepts/architecture.md`, root `CLAUDE.md`, `src/better_robot/*/CLAUDE.md`, `tests/contract/*.py`,
`pyproject.toml` (ruff section). Also `/home/rikhat.akizhanov/better/BetterRobot/references/design/BEST_PRACTICES.md`.

## Goal
A contract set that a new contributor (human or agent) can follow mechanically, that the current code mostly
already satisfies, and whose remaining violations are listed so the polish phase can fix them. The owner's words:
"contracts we set and systematically follow... naming, code style, classes, mutability, differentiability, etc.,
plus architectural decisions... allow systematic extension... avoid refactoring when the project grows."

## Owner decisions already taken (do not re-open)
- Monogram notation (`X_AB`, `p_AoBo_W`, `V_WB`) is adopted for local variables inside math code
  (lie/spatial/kinematics/dynamics) only; public API keeps descriptive names.
- The viewer's time axis is renamed `frame`→`step` (`num_steps`, `set_step`); "frame" = robot frame only.
  Another agent does the rename now; write the rule as if done.
- `int | str` ids accepted everywhere a frame/joint/body is named. Args after the second are keyword-only.
  No boolean positionals. `IRError` joins the `BetterRobotError` tree; never-raised exception types are removed.
- Terms are plain functions; `optim` imports only torch; optional deps lazy at the boundary.

## Deliverables (BetterRobot, `/home/rikhat.akizhanov/better/dev/BetterRobot`)
1. `docs/contracts/` — pages `00-index.md` … `10-docs-tests.md` as in p0_docs_refs Part 2 (i), plus
   `decisions/index.md` and ADRs `0001`–`0005` seeded from `docs/concepts/design-decisions.md` (torch-only,
   immutable FK state, terms as plain functions, viser viewer, nq≠nv) with the template in p0_docs_refs (iv).
   Each rule: ID (`C-NAME-1`…), MUST/SHOULD, one-paragraph rationale, a 2–6 line good/bad example from THIS
   codebase, enforcement tag `[ruff:<codes>]`, `[test:tests/contract/<file>::<test>]` or `[review]`.
   Keep each page ≤ 120 lines. Plain language; a term is defined the first time it appears.
   `00-index.md` has the enforcement matrix (rule → enforcer) and the "when to write an ADR" rule.
2. Fold `docs/contributing/conventions.md` and `docs/concepts/design-decisions.md` into the contracts (trash
   them via `python3 ~/.claude/skills/autoresearch/scripts/trash.py`; fix every link: `grep -rn` in docs/,
   README.md, CLAUDE.md, src/**/CLAUDE.md, mkdocs.yml nav). `docs/contributing/extending.md` stays but links to
   `08-extension-points`. Add the contracts section to `mkdocs.yml` nav. `uv run mkdocs build --strict` must pass.
3. Enforcement in `tests/contract/`: add tests from p0_docs_refs (iii) that can pass TODAY or that fail only on
   the violations you list in (4): `test_import_dag.py` (ast-based; replace or extend the existing
   `test_layer_dependencies.py`, do not duplicate), `test_optional_deps.py`, `test_frozen.py`,
   `test_no_inplace.py`, `test_dtype_device.py`, `test_public_docstrings.py` (every `__all__` symbol has a
   docstring; shape line where a tensor is returned), `test_keyword_only.py` (public callables: args after the
   second are keyword-only; list current violations as an explicit allowlist that the polish phase will empty).
   `test_vmap.py`/`test_gradcheck.py` only if existing autograd tests don't already cover them (check
   `tests/autograd`, `tests/lie/test_torch_backend_gradcheck.py`). Run the full suite; all green or every red
   test is an allowlisted violation with a clear message.
4. `ruff` config: propose the added rule codes in `pyproject.toml` but ONLY enable those that are clean now or
   whose violations are ≤ 20 and listed in the violations ledger; the rest go in a commented block with the
   count of current violations each would raise (`uv run ruff check --select <code> src | tail -1`).
5. `subagents/p1_violations.md`: the ledger — every rule the current code breaks, with file:line, grouped by
   package, so the polish agent can work through it. Include the items from p0_architecture_br §3 that the
   contracts now forbid.
6. Update root `CLAUDE.md` "Start here" and each `src/better_robot/*/CLAUDE.md` to point at the contracts
   (one line each; do not rewrite them).

## BetterHuman addendum (`/home/rikhat.akizhanov/better/dev/BetterHuman`)
7. `docs/contracts/index.md` + `bh-addendum.md`: BH follows BR's contracts (link by path), plus the BH-specific
   rules: families/assets layout, model loading, `BETTERHUMAN_MODELS_DIR`, studio = thin shell over BR viewer,
   the data layout `<data>/{segmentations,frames,poses}/<key>/`. Fold `docs/contracts.md` and
   `docs/design_principles.md` in (trash the originals, fix links). BH has no mkdocs yet — plain markdown.
   Its `tests/contract/` gets the import-DAG and keyword-only tests adapted (allowlist as in BR).

## Gates (report exact tails)
BR suite, BH suite, `uv run ruff check src tests`, `uv run mkdocs build --strict`.
Final message ≤ 30 lines: files created/trashed, test counts, ruff codes enabled vs deferred (with violation
counts), ledger size per package, open questions. Budget is tight: write the contracts first, tests second,
ledger third; if you run low, write `p1_state.md` with exactly what remains and stop.
