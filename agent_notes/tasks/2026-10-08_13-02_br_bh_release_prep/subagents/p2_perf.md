# P2 perf slice — benchmarks, regressions, hot-call profiles

Raw outputs: `agent_notes/tasks/2026-10-08_13-02_br_bh_release_prep/artifacts/perf/` (gitignored; `ART/` below).
CPU only: `gpu_free.py` reported "no usable GPUs" (all 8 busy). Machine: AMD EPYC 9654 (same CPU as every stored
json), load ~95 at start and ~97% busy on every core during the run, so absolute numbers carry ~10-20% noise
(pinocchio rows, which did not change, came out 1.0-1.2x the stored values).

## Files read
- BR `benchmarks/run.py`, `benchmarks/m1_eager_cpu.py` (CLI, algo lambdas, metadata), `benchmarks/results/*.json`
  (metadata of all four), `tests/bench/README.md`, bench file list, `pyproject.toml` markers.
- BH `benchmarks/run.py` (CLI, spec/inputs/bake/vertices/rnea cases).
- Docs claims: `docs/getting-started/gpu-and-batching.md`, `docs/reference/changelog.md`,
  `docs/contracts/decisions/0001-torch-only.md`, `docs/guides/benchmark-and-profile.md`, BH `docs/contracts/bh-addendum.md`.

## Fixed (what, where, test)
1. **All three benchmark scripts crashed** on the dynamics/Jacobian rows after commit 31a37ed made
   `frame_jacobian(..., *, frame=)`, `inverse_dynamics(..., *, v=, a=)`, `forward_dynamics(..., *, v=, tau=)`
   keyword-only. BR `run.py` silently wrote error rows (48 of 160 rows: every jacobian/rnea/aba row);
   `m1_eager_cpu.py` and BH `run.py` died with `TypeError`. Fixed the call sites:
   BR `benchmarks/run.py` (jacobian, rnea, aba lambdas + the `JACOBIAN_FN` metadata string),
   BR `benchmarks/m1_eager_cpu.py` (rnea), BH `benchmarks/run.py:294` (rnea). Test: re-ran all three; BR `run.py`
   now has 0 error rows (`ART/br_run_t1.json`), m1 completes (`ART/br_m1_eager.json`). No pytest added (scripts).

## Reported, not changed (ranked)
1. **(medium) Nothing guards the benchmark scripts against API drift.** The break above went unnoticed because no
   test imports or runs them. Suggest a tiny smoke test (`run.py --models panda --batches 1 --repeats 1`, assert no
   `error` key) under `-m bench` or the normal suite.
2. **(medium) `pytest -m bench` collects almost nothing by default.** Bench files are `tests/bench/bench_*.py`, which
   the default `python_files` does not collect: `pytest tests/bench -m bench` runs 1 test. The README command
   `pytest tests/bench/bench_*.py -m bench --benchmark-only` works (13 passed, `ART/br_pytest_bench.log`). Suggest
   adding `bench_*.py` to `python_files` or saying so in CLAUDE.md.
3. **(low) `tests/bench/README.md` names `tests/bench/bench_trajopt_sparse.py`, which does not exist.** Stale doc.
4. **(low, environment) Intra-op threads collapse under core contention.** FK Panda with 8 threads on 8 cores that
   another 8-thread process also used: B=16 52 ms, B=256 61 ms, vs 0.31 / 1.06 ms with 1 thread on the same cores;
   on less-contended cores 8 threads give 0.32 / 0.70 ms (`ART/fk_threads.log`). Not a library bug (OpenMP
   oversubscription), but the `--threads 8` run (`ART/br_run_t8.*`, B=64/256 FK 60-80 ms) and the FK profile below are
   contaminated by it. Small batches gain nothing from threads anyway (B=256: 0.80 -> 0.70 ms). Worth one sentence in
   `docs/guides/benchmark-and-profile.md`: benchmark with `--threads 1` or on dedicated cores.
5. **(info) ADR 0001 numbers (6.6 ms Panda, 27.8 ms G1 at B=4096) refer to the `fk_state` algo** (joint transforms
   only), which `run.py` does not run by default (`fk` = `joint_pose`, 20.1 ms Panda now). Consistent with
   `results/p2_lean_fk_cpu.json` (6.64 / 26.86); not re-measured.

## Performance notes (measured)
Single thread, `run.py --models panda,g1 --repeats 10` vs stored `results/0953f28_cpu.json` (same CPU, 1 thread,
2026-10-06). Full table: `ART/compare_t1_vs_0953f28.txt`. **No row is slower than stored by more than 15%; every
better_robot row is faster** (ratio new/old 0.12-0.94). Selected medians (ms):

| algo | model | B | old fwd | new fwd | old fwd+bwd | new fwd+bwd |
|---|---|---|---|---|---|---|
| fk | panda | 4096 | 32.9 | 20.1 | 70.5 | 40.3 |
| jacobian | g1 | 4096 | 270.9 | 50.0 | 669.3 | 169.7 |
| rnea | panda | 4096 | 53.4 | 44.3 | 165.5 | 111.3 |
| crba | g1 | 1 | 4.50 | 0.77 | 19.07 | 2.27 |
| aba | panda | 1 | 6.09 | 0.79 | 16.65 | 3.08 |
| aba | g1 | 4096 | 359.4 | 333.4 | 955.5 | 755.5 |

Docs claim (`gpu-and-batching.md`, Panda, 1 thread): RNEA 1.07 / 52.0, CRBA 0.90 / 56.8, ABA 1.30 / 81.0 ms at
B=1 / 4096. Now: RNEA 0.66 / 44.3, CRBA 0.59 / 48.8, ABA 0.79 / 72.8. Claims hold (current code is faster);
the CUDA column was not re-checked (no free GPU).

pytest-benchmark (8 threads, `ART/br_pytest_bench.log`, µs median): joint_jacobian_panda 16, so3_exp[1024] 67,
se3_compose[1024] 156, fk_panda_unbatched 272, fk_panda_batched[256] 750, fk_panda_batched[16] 11993 (contention
outlier, see item 4), smpl integrate/difference grouped 1048/1147 vs per-joint loop 3759/4140 (3.6x), solve_ik_panda
132.8 ms.

Profiles (`ART/profile_hot.py`, `ART/profile_hot.log`, torch.profiler, top-5 by self CPU):
- `forward_kinematics_state` Panda B=256: mm 30%, bmm 24% (15 calls, one per joint), cos 12%, copy_ 12% (31), sin 12%.
  Timing (57 ms) is contaminated by item 4; the uncontended 1-thread median is 0.80 ms.
- `frame_jacobian` LWA Panda B=256 (state precomputed): one bmm = 97% (5.1 ms); everything else < 1%. The
  column-table bmm is the whole cost; possibly worth checking whether it multiplies more than the frame's support.
- `solve_ik` LM Panda B=64, 10 iters: 52 ms; bmm 14% (594 calls), mul 6% (1698), mm 6% (22), cat 5% (577),
  index 5% (50). Launch-bound: ~60 bmm and ~58 cat per LM iteration. Final loss after 100 iters: median 5.8e-4.
- BH `identity.forward(q, stream).vertices()` SMPL-X B=64: see `ART/profile_hot.log` (written after this file;
  not summarised here).
- BH `benchmarks/run.py --threads 8`: was still running (SMPL done; SMPL-X, MHR pending) when this report was
  written -> `ART/bh_run_t8.{log,json}`; not compared against BH `benchmarks/results/p4_bodies_cpu.json`.

## Gate results
- BR full suite: see final message (was running at write time, `ART/br_full_tests.log`).
- `ruff check` of the three edited benchmark scripts: see final message. `ruff check src tests` untouched by this
  slice (no src/tests edits). mkdocs not needed (no docstrings touched).

Gate update (after handback): BR full suite 3 failed, 1586 passed, 11 skipped. Failures are tests/docs/test_snippets.py for docs/concepts/viewer.md and docs/getting-started/visualize.md. They are viewer doc pages, outside this slice, and other agents were editing them concurrently. Not diagnosed.
