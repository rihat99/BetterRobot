# P2 sweep — slice `bh` (BetterHuman, worktree `/home/rikhat.akizhanov/better/dev/BetterHuman`)

Scope note: the agent had a 50k-token budget, so the package (~6400 lines, 40 files) was NOT read end to end.
Coverage was targeted: the reduction/drop path, the known failing test, device/dtype/sync hazards (grep over
`core/`, `terms.py`, `families/`), mass topology, plus all four examples `--check`, `benchmarks/run.py --device cpu`,
full test suite and ruff. A follow-up sweep of the unread files is still needed (list below).

## Files read
- Fully or in the relevant part: `core/spec.py` (145-226: `to`, `top_k_skin`, `drop_joints`), `core/reduction.py`
  (100-163), `core/pipeline.py` (`linear_corrective`), `core/rig.py` (`native_joint_ids`), `core/body.py` (44-58,
  250-263), `core/mass.py` (116-160 `build_part_topology`), `families/smpl_family.py` (59-115), `families/classic.py`
  (85-110), `terms.py` (`ground`, 135-155), `tests/core/test_reduction.py` (55-82), `tests/families/test_smplx.py`.
- Grep-only (device/dtype/`.item()`/`.cpu()`/`.detach()` hazards): all of `core/`, `terms.py`,
  `families/{classic,smpl_family,mhr,mhr_rig,mhr_tree}.py` — no hot-path hazard found.
- NOT read: `core/{classic,contact_frames,__init__}.py`, `families/{mhr*,tables,assets/*}.py` (beyond grep),
  `io/*`, `viz/human_layers.py`, `viz/studio/*` (incl. `app.py`, `gizmo.py`: no findings to report about them).

## Fixed (what, where, test)
1. **`drop_joints` applied to an already-reduced spec lost the native (file) joint numbering.**
   `src/better_human/core/spec.py::drop_joints` set `meta["native_joint_ids"]` to the survivor indices of the
   *current* spec, not of the file. Repro: `bh.load("smplx", face=False)` has native ids `[..20, 21, 25, 26..]`;
   dropping one more joint gave `[..20, 21, 22, 23..]` (last id 50 instead of 53). `families/classic.py::_assemble`
   indexes the classic pose axes with these ids, so a twice-reduced body read the wrong classic pose axes
   (`RegressedRig.native_joint_ids` returns an `arange` and is not affected).
   Fix (2 lines): compose with the existing `meta["native_joint_ids"]` when present.
   Pinning test: `tests/families/test_smplx.py::test_repeated_drop_joints_keeps_native_joint_ids` (passes; the probe
   above shows the pre-fix value it would reject).

## Reported, not changed (ranked; severity; evidence; suggested fix)
1. **Known failing test `tests/core/test_reduction.py::test_corrective_slice_is_exact_when_dropped_joint_is_at_rest`
   — test expectation, not a library bug (low).** The reduction itself is exact: `reduced_posedirs == posedirs[..., :18]`
   passes with atol 0, and the dropped joint's feature (R - I at identity) is exactly 0.0. The failing assert compares
   `feature(27) @ matrix(27, 6)` with `feature(18) @ matrix(18, 6)` at `rtol=0, atol=0`; float32 GEMV with a different
   inner length sums in a different order. Max diff 7.63e-6 at magnitude 112.9 = exactly 1 ulp (rel 1.49e-7); the
   float64 reference is 1.17e-5 from the full result and 1.94e-5 from the reduced one, so neither is "wrong". Same
   diff with 1 and 192 threads (not a threading race). Suggested fix (owner decision, test change): compare the two
   correctives at float32 tolerance (e.g. default `assert_close`), or compute this check in float64, keeping the
   exact atol-0 check on `reduced_posedirs` which is the real invariant.
2. **`drop_joints` is O(J) Python with `int(tensor)` syncs** (`spec.py` ancestor loop) — setup-time only; info.
3. **`terms.ground` keeps `ids` on CPU** and indexes device tensors with it; works in PyTorch (CPU long index on CUDA
   is allowed) but costs an H2D copy per residual call. Low; could move `ids` to the positions' device once.
   Not measured on GPU.

## Performance notes (measured)
- `core/mass.py::build_part_topology` (Python loop parts x faces): 0.19 s for SMPL-X (20908 faces, 55 parts), out of
  2.74 s for `bh.load("smplx")` (1.97 s with `mass=False`). Setup-time; not worth changing.
- Benchmark: see Gate results.

## Gate results
- Full BH suite (started before the fix was applied, so it ran pre-fix `spec.py` and without the new test):
  `1 failed, 332 passed, 1 skipped` in 691 s; the one failure is the known `test_reduction` case above.
- After the fix: `tests/core` + `tests/families/test_smplx.py` -> `1 failed, 66 passed` (same known failure; new
  test passes). The full suite was NOT re-run after the fix (budget); the orchestrator should re-run it once.
- Examples `01`-`04 --check`: all exit 0.
- `benchmarks/run.py --device cpu` exited 2: `--out` is a required argument. Not re-run (budget). Stale docs:
  BH `CLAUDE.md` gives `uv run python benchmarks/run.py --device cpu --threads 8` without `--out`; the brief's
  command has the same gap. Suggest adding `--out <path>` to the documented command (or giving `--out` a default).
- `uv run ruff check src tests`: All checks passed.
