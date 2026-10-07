# P2a brief — apply the test trim (BetterRobot + BetterHuman)

Read `context.md`, then `p0_tests.md` in full. Repo `/home/rikhat.akizhanov/better/dev/BetterRobot` (and BH).
Owner decision: the −22 % proposal in `p0_tests.md` is approved as written. Scope: `tests/` only. Do NOT touch
`tests/contract/`, `tests/viewer/`, `tests/docs/`, `tests/bench/` (other agents work there) nor anything in `src/`.

## Rules
- "drop" = delete the test function (and fixtures/imports your deletion orphaned). Never skip/xfail/weaken.
- "merge" = one parametrized test that keeps EVERY case and EVERY assertion of the originals, same tolerances.
  Keep the ids readable (`pytest.param(..., id="spherical")`).
- Structural items from p0_tests "Structure": fold `tests/test_skeleton_signatures.py` remains into
  `tests/contract/test_public_api.py` ONLY if that means moving unchanged functions (otherwise leave a note);
  merge `test_types_so3.py`+`test_types_se3.py` into `tests/lie/test_types.py` parametrized over the group;
  mark `tests/examples/test_examples_run.py` cases `@pytest.mark.slow` only if a `slow` marker exists in
  `pyproject.toml` (check; do not add markers).
- Fix the stale `bias_forces` / `rnea` / `crba` mentions in `tests/test_pinocchio/*` docstrings (names only;
  keep the file names).
- After applying: re-run the public-symbol coverage grep (every `__all__` symbol must still be referenced by at
  least one test or docs page); report any symbol that lost coverage and restore the cheapest test for it.
- BH: apply the merges listed under "BetterHuman" in p0_tests.md; verify the `test_smoke.py` claim before
  dropping it (if its two tests are genuinely covered elsewhere, drop; else keep).

## Deliverable
`subagents/p2_tests_ledger.md`: table `file | test | action | reason | superseded_by` for every change (this goes
into the REPORT), plus before/after counts (`grep -rc "def test_"`) and suite timings.

## Gates (exact tails in the final message)
```bash
cd /home/rikhat.akizhanov/better/dev/BetterRobot && CUDA_VISIBLE_DEVICES="" UV_CACHE_DIR=/tmp/betterrobot-uv-cache uv run pytest tests -q -m "not bench and not cuda" -p no:cacheprovider
uv run ruff check tests
cd /home/rikhat.akizhanov/better/dev/BetterHuman && CUDA_VISIBLE_DEVICES="" BETTERHUMAN_MODELS_DIR=$PWD/models uv run pytest tests -q -p no:cacheprovider
```
Final message ≤ 25 lines: counts before/after per repo, files merged/created, symbols that lost coverage (should
be none), gate tails, anything undone.
