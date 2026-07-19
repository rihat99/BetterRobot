# For agents — how to work these orders

Read `plan/README.md` (ground rules) and `plan/01_target_api.md` (the design
order 01 implements) before touching code. The orders run 01 → 02 → 03; each
ends with the full gate green and a results file.

## Commands

```bash
UV_CACHE_DIR=/tmp/betterrobot-uv-cache uv run pytest tests/ -q -m "not bench and not cuda"
uv run pytest tests/test_pinocchio/ -q          # parity anchor
uv run pytest tests/contract/ -q
uv run ruff check src/ tests/ && uv run ruff format --check src/ tests/
.venv/bin/sphinx-build -b html docs docs/_build/html
.venv/bin/sphinx-build -b doctest docs docs/_build/doctest
```

CUDA-marked tests are owner-run. Anything device-conditional you add must be
correct on both CPU and GPU paths (ground rule 5) — this host has GPUs; use
them for spot checks where cheap, but do not run the CUDA suite wholesale.

## Contract tests

Each order lists the contract files it may update. Updating any other
contract file, or weakening a parity tolerance, is a stop-and-report, not a
judgment call. If a required change forces an unlisted contract edit, do the
minimal edit and flag it under "Deviations" in the results file — the polish
round showed this happens (about once per order) and honesty is the fix.

## Results files

`for_agents/NN_results.md`, same shape as the polish round: Delivered /
Verification table (exact pytest counts) / Line accounting / Deleted-test
disposition (every removed test named, with why) / Deviations and findings.
Write it for a skeptical reviewer who will spot-check every claim — the
previous round's files survived adversarial review because they were honest
about deviations; keep that bar.

## Style and skills

Python work triggers the `python-standards` skill; new test files trigger
`write-tests`; new module names trigger `file-naming`; docs pages trigger
`sphinx-docs`/`diataxis-docs`. If a skill is unavailable, follow the nearest
existing file's conventions. Keep hot paths free of `.item()` host syncs and
Python-over-batch loops (contract-linted). Error voice everywhere:
`"<what> must <rule>, got <actual>"`.

## MIGRATION.md

Every removed or renamed public symbol gets one row (symbol → replacement) in
the root `MIGRATION.md`. Append rows; never rewrite history there. Downstream
repositories are the owner's problem, not yours.
