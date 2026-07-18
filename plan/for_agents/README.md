# for_agents — Work Orders for the Polish Phase

One file per roadmap phase. Each is self-contained: goal, evidence, steps,
tests, acceptance. Execute in numeric order unless the order says otherwise.
Read `plan/README.md`, `plan/01_vision.md`, and `plan/02_architecture.md`
first — the work orders assume them.

## Files, in execution order

| File | Phase | Depends on |
|------|-------|------------|
| `01_delete_legacy.md` | Delete the legacy optimization stack | nothing |
| `02_rebuild_optim.md` | One lean optimizer package | 01 |
| `03_open_the_core.md` | Export, un-hide, wire differentiability | 02 |
| `04_polish_the_core.md` | Validation diet + style uniformity | 02 (03 preferred first) |
| `05_rewrite_docs.md` | Human documentation, new structure | 01–04 (API must be settled) |
| `06_loose_ends.md` | Small fixes, honest bookkeeping | 01; benchmark items after 02 |

## Standing rules — binding for every task

1. **No backward compatibility, no deprecation shims, no legacy keeps.** The
   project is unshipped. When you remove a public symbol, append one row to
   `plan/migration_ledger.md` (surface → replacement). That is the entire
   obligation. Never read or modify any repository outside this one —
   downstream migration is the owner's job.
2. **The safety net stays green.** After every task: full gate
   `UV_CACHE_DIR=/tmp/betterrobot-uv-cache uv run pytest tests/ -q -m "not bench and not cuda"`,
   plus scoped ruff and a Sphinx build when you touched docs. The
   pinocchio-parity suite (`tests/test_pinocchio/`) is the numerical anchor —
   it never changes to make a refactor pass. Contract tests
   (`tests/contract/`) change only where a work order explicitly says so.
3. **Deletion follows evidence.** Work orders cite caller-graph facts
   (file:line, dated 2026-07-18). Re-verify with a grep before deleting; if
   you find a caller the order says doesn't exist, stop and report — do not
   improvise a shim.
4. **Size budgets are hard.** Each order states a LOC budget. If your
   implementation cannot meet it, the design is wrong — stop and report
   rather than shipping bulk.
5. **Simplicity bar for new code.** No abstraction without a second concrete
   caller in-tree. No value-content policing (`isfinite` scans of user
   tensors). Validate structure once at the public boundary; below it, trust.
   Match the surrounding file's style. If a senior engineer would call it
   overcomplicated, rewrite it before review, not after.
6. **Behavior parity where the order demands it.** Refactors of `solve_ik` /
   `solve_trajopt` / `solve_contact_forces` keep existing regression tests
   passing with unchanged tolerances. Numerical conventions never change:
   SE(3) pose `[tx,ty,tz,qx,qy,qz,qw]`, tangent `[linear(3), angular(3)]`,
   LOCAL_WORLD_ALIGNED Jacobians, batched `(B..., feature)`.
7. **Docs and CLAUDE.md follow the code in the same change.** A change that
   makes a documented statement false updates that statement. Never introduce
   milestone/process jargon into `docs/` — the words "M1…M6", "named-block",
   "legacy", "prevalidated", "work order", "ledger" (outside
   `source_and_license.md`) do not belong in shipped documentation.
8. **Report honestly.** Deviations, failures, and surprises go in your report
   verbatim. Write results to `plan/for_agents/<order>_results.md`: what was
   delivered, what was not, exact test counts, net line delta.
9. **Skills.** If available in your session, invoke `python-standards` before
   editing Python, `write-tests` before adding tests, `sphinx-docs` +
   `diataxis-docs` before docs work, `design-principles` before any API
   reshaping.
10. **CUDA.** The host has eight RTX 6000 Ada GPUs; the default sandbox hides
    them (a false-negative `nvidia-smi`). CUDA-marked tests are owner-run;
    do not "fix" them based on sandbox output, and do not enable CI.

## Repo facts

- Root: `/data3/rikhat.akizhanov/better/BetterRobot`; source
  `src/better_robot/`, tests `tests/`, docs `docs/` (Sphinx + MyST).
- Baseline at plan time (2026-07-18): 26,881 src lines, 1,543 tests green,
  branch `dev`, HEAD `461e79a` plus the polish-plan commit.
- The docs build: `uv run sphinx-build -b html docs docs/_build/html`
  (four offline-intersphinx warnings are expected and fine).
