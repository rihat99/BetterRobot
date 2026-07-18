# Benchmarks

Micro-benches for SE3/SO3 ops, forward kinematics, Jacobians, manifold
integration/difference, and `solve_ik`. They are advisory — the
bench-cpu-advisory CI job records numbers but does not gate merges.

## Running

```bash
uv run pytest tests/bench/bench_*.py -m bench --benchmark-only
```

The M5 trajectory scaling benchmark is a separate fresh-process harness, not
a pytest-benchmark micro-benchmark. Its canonical CPU run writes the committed
schema at `baselines/trajopt_sparse_cpu.json`:

```bash
uv run python tests/bench/bench_trajopt_sparse.py
```

Use `--quick --allow-unpinned` for a single structured T=50 harness check; a
quick or partial run never overwrites the canonical baseline unless an
explicit `--output` path is supplied.

## Bumping the baseline

Use the following repository-local review procedure:

1. Confirm the change is intentional (algorithm bump, dependency
   upgrade, hardware flip).
2. Run the bench on the standard CI machine
   (`benchmark-json=tests/bench/baseline_cpu.json`).
3. Open a PR titled `bench-baseline: bump <reason>`, with the new
   `baseline_cpu.json` and any hardware-named CUDA artifact if relevant.
4. Two reviewers must approve a baseline bump.

CUDA artifacts must name the GPU actually measured and carry the environment
header from `definitions.md`; do not infer a self-hosted runner from a filename.

## File layout

| File | Purpose |
|------|---------|
| `bench_lie.py` | SE3/SO3 compose/exp/log throughput |
| `bench_forward_kinematics.py` | FK on Panda, batched and unbatched |
| `bench_jacobian.py` | `compute_joint_jacobians` on Panda |
| `bench_integrate_difference.py` | Grouped vs loop manifold operations on a length-200 SMPL-like trajectory |
| `bench_solve_ik.py` | One-shot Panda IK |
| `bench_trajopt_sparse.py` | M5 dense-vs-banded CPU scaling harness with isolated subprocess RSS |
| `test_trajopt_sparse_smoke.py` | Normal-suite T=50, one-update structured smoke |
| `test_mem_watermark.py` | Nightly only: peak memory tracking |
| `definitions.md` | M6 hardware header, canonical matrix, timing rules, and evidence status |
| `baseline_cpu.json` | CI-runner baseline (currently a `_status: PLACEHOLDER`; populate from one CI run before relying on the comparison gate) |
| `baselines/warp_fk_cuda_rtx6000_ada_b*.json` | Measured, fresh-process M6 SMPL FK Warp-vs-compiled-Torch CUDA cases |
| `baselines/m6_torch_filtered_smpl_b1_rtx6000_ada.json` | Filtered B=1 SMPL FK/RNEA/IK CPU/CUDA eager/compiled evidence; not the complete canonical matrix |
| `baselines/trajopt_sparse_cpu.json` | M5 Phase-C schema/results; pending the canonical full CPU run |

> ### Status: placeholder baselines
>
> `baseline_cpu.json` ships with ``_status: "PLACEHOLDER"`` and an empty
> ``benchmarks`` list. The
> CI advisory job will run the suite but cannot detect regressions
> until it is replaced with real numbers from one known-good CI run. Until
> then, `bench-cpu-advisory`
> records numbers but the comparison is meaningless. The first PR that
> stabilises the bench fixtures should bump it following the procedure above.
