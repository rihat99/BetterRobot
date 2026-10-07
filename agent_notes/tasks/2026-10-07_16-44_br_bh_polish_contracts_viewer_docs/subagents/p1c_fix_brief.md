# P1c brief — apply the adversarial review to docs/contracts

Read `context.md`, then `p1_review.md` (the fix list; items marked **[O]** carry the orchestrator's decision —
follow them), then `p1_state.md` and `p1_violations.md` (what already exists), then every page in
`docs/contracts/` and `docs/contracts/decisions/`. Repo `/home/rikhat.akizhanov/better/dev/BetterRobot`.
Scope: `docs/contracts/**`, `docs/contracts/decisions/**`, `tests/contract/test_contract_matrix.py` (new),
`pyproject.toml` ruff `banned-api` table only, `p1_violations.md` (append). Nothing else. Another agent is
concurrently adding `tests/contract/test_dtype_device.py` and `test_no_inplace.py` — cite them as existing
enforcers; do not create them.

Work through p1_review.md items 1–16 in order. Rules of the game:
- A rule must describe the code as it IS, or be tagged `[planned: L-<n>]` pointing at a ledger entry in
  p1_violations.md (append the entry if missing). No invented modules, classes or helpers.
- Every rule ID appears in the 00-index matrix with exactly the enforcer its page names (write
  `tests/contract/test_contract_matrix.py`: parse the pages, compare rule IDs and enforcer tags with the matrix,
  and check that every `tests/contract/<file>::<test>` and ruff code named actually exists / is selected —
  `[planned]` tags are exempt). Make it pass.
- Counts are `n<thing>` (`njoints`, `nframes`, `nsteps`); viewer time axis is `step`.
- C-DOC-1: MUST for new or rewritten code; existing public symbols need only a docstring with the shape line.
- Gaps 14a,b,c,d,f,g,h get a rule each (short); 14e becomes an ADR stub with Status: Proposed.
- Keep each page ≤ 120 lines, plain language, define terms on first use (fix item 16).
- Examples cite real `file:line` (verify with grep) or are labelled "illustrative".

Gates: `uv run pytest tests/contract tests/docs -q -p no:cacheprovider`, `uv run mkdocs build --strict`,
`uv run ruff check src` (count must not change except via the banned-api additions you make — report both).
Final message ≤ 25 lines: items fixed (by number), rules added/renumbered, ledger entries added, gate tails,
items you could not resolve and why. Do not spawn agents. Do not commit. If budget runs low, append the exact
remaining item numbers to p1_state.md and stop.
