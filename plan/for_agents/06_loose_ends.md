# 06 — Loose ends

**Goal:** close the small, concrete items the post-implementation review
surfaced (2026-07-18) that no other phase owns. Each item is independent;
fix, or move to `plan/for_future.md` with a reason. Items marked (after 02)
depend on the optimizer rebuild landing first.

---

1. **Stale test pinning an old error string** — `tests/optim/
   test_manifolds.py:200` vs `manifolds.py:104`. **Already fixed by the owner
   session on 2026-07-18** (the test now matches the improved, de-milestoned
   message). Verify it stayed fixed; nothing to do if green.

2. **Inactive-residual objective inconsistency** — `_objective_with_context`
   adds `kernel.rho(0)` for inactive items while the residual path skips
   them. **Owned by phase 02 T6**; verify it landed, add a regression test
   with a custom kernel where `rho(0) != 0` if none exists.

3. **Contact-point honesty** — `solve_contact_forces` applies each wrench as
   `[force, torque=0]` at the joint origin (`tasks/contact_forces.py:102`);
   an offset contact point would contribute `r × f`. Not a bug — a
   limitation. Say so plainly in the docstring and the task's doc page;
   `for_future.md` already carries the feature.

4. **No placeholder artifacts ship** (after 02) — `tests/bench/baselines/
   trajopt_sparse_cpu.json` is still `"_status": "PENDING_MEASUREMENT"`.
   Either run the committed CPU scaling sweep once on this box against the
   rebuilt solver and commit real numbers, or delete the placeholder and the
   sweep's promise together (the definition file may stay as a definition).
   A baseline that says "pending" is a claim the repo doesn't back.

5. **Benchmark definitions vs deleted code** (after 02) — the M6 baseline
   harness (`benchmarks/m6_baseline.py`, 144 selectors) and
   `tests/bench/definitions.md` reference solver surfaces phases 1–2
   reshaped. Re-run the harness's own schema/selector tests, fix references
   mechanically, and rename anything "m6"-flavored to neutral names
   (`benchmarks/baseline.py`); no new measurements required beyond item 4.

6. **`CLAUDE.md` re-sync** (after 01–04) — root `CLAUDE.md`,
   `src/better_robot/optim/CLAUDE.md`, `data_model/CLAUDE.md`,
   `io/CLAUDE.md`: rewrite the stale sections (two-stack tables, legacy
   notes, milestone references, "1,543 tests" style counts) to describe the
   polished reality, at half their current length or less. These files brief
   future agents — wrong briefings are worse than none.

7. **Example hygiene** — `examples/05_panda_trajopt.py` gets its phase-1
   input change; also sweep all examples for deleted-API imports and the
   fp32-viewer downcast wart (keep the wart if the viewer still needs it,
   but comment it honestly). Examples must run: `uv run python examples/<f>`
   smoke-checked (viewer ones with `--no-viewer`).

8. **Changelog** — rewrite `docs/CHANGELOG.md`'s Unreleased section as a
   plain user-facing list (what changed for a user of the library), no
   milestone names; phase 5 owns the prose standard, this item owns the
   content being complete w.r.t. phases 1–4 (especially every removed
   symbol pointing at `plan/migration_ledger.md`).

9. **Stray directory** — `src/better_robot/utils/` holds only `__pycache__`;
   remove it (owned by 03 T5; verify).

## Acceptance

Every numbered item: fixed (with the diff/test named in the results file),
verified already-done, or moved to `for_future.md` with one honest sentence.
No `PENDING_MEASUREMENT` or placeholder artifacts anywhere in the tree
(`grep -rn "PENDING_MEASUREMENT\|TODO(milestone" src/ tests/ docs/ benchmarks/`
→ empty). Full gate green.
