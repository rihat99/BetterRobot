# Benchmarks

Micro-benches for SE3/SO3 ops, forward kinematics, Jacobians, manifold
integration/difference, and `solve_ik`. They are advisory — the
manual `benchmarks` workflow job records numbers but does not compare a
blocking baseline or gate merges.

## Running

```bash
uv run pytest tests/bench/bench_*.py -m bench --benchmark-only
```

The trajectory scaling benchmark is a separate fresh-process harness, not
a pytest-benchmark micro-benchmark:

```bash
uv run python tests/bench/bench_trajopt_sparse.py
```

Use `--quick --allow-unpinned` for a single structured T=50 harness check and
`--output <path>` to retain any run. The repository does not ship a trajectory
baseline: the full canonical sweep could not complete enough dense points
within its committed timeout and memory bounds to support its scaling checks.

## Bumping the baseline

Use the following repository-local review procedure:

1. Confirm the change is intentional (algorithm bump, dependency
   upgrade, hardware flip).
2. Record the exact hardware, dependency versions, source commit, affinity,
   warmup, and timing policy. There is no designated standard CI machine yet.
3. Run the unchanged definition and write a fresh JSON artifact with
   `--benchmark-json=<path>` (or the standalone harness's `--output`).
4. Review the raw samples and functional checks, then replace a committed
   baseline only in an explicit baseline-change commit. No automated reviewer
   count or comparison gate is configured by this repository.

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
| `bench_trajopt_sparse.py` | Dense-vs-banded CPU scaling harness with isolated subprocess RSS |
| `test_trajopt_sparse_smoke.py` | Normal-suite T=50, one-update structured smoke |
| `test_mem_watermark.py` | Explicit local/manual peak-memory tracking; not scheduled |
| `definitions.md` | Hardware header, canonical matrix, timing rules, and evidence status |
| `baselines/warp_fk_cuda_rtx6000_ada_b*.json` | Measured, fresh-process SMPL FK Warp-vs-compiled-Torch CUDA cases |
| `baselines/torch_filtered_smpl_b1_rtx6000_ada.json` | Filtered B=1 SMPL FK/RNEA/IK CPU/CUDA eager/compiled evidence; not the complete canonical matrix |
