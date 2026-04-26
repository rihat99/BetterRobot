# Benchmarks

Micro-benches for SE3/SO3 ops, forward kinematics, Jacobians, and
`solve_ik`. They are advisory — the bench-cpu-advisory CI job records
numbers but does not gate merges.

## Running

```bash
uv run pytest tests/bench/ -m bench --benchmark-only
```

## Bumping the baseline

Per `docs/claude_plan/accepted/12_regression_and_benchmarks.md §12.D`:

1. Confirm the change is intentional (algorithm bump, dependency
   upgrade, hardware flip).
2. Run the bench on the standard CI machine
   (`benchmark-json=tests/bench/baseline_cpu.json`).
3. Open a PR titled `bench-baseline: bump <reason>`, with the new
   `baseline_cpu.json` (and `baseline_cuda_l40.json` if relevant).
4. Two reviewers must approve a baseline bump.

The same procedure applies to `baseline_cuda_l40.json` on the
self-hosted GPU runner.

## File layout

| File | Purpose |
|------|---------|
| `bench_lie.py` | SE3/SO3 compose/exp/log throughput |
| `bench_forward_kinematics.py` | FK on Panda, batched and unbatched |
| `bench_jacobian.py` | `compute_joint_jacobians` on Panda |
| `bench_solve_ik.py` | One-shot Panda IK |
| `test_mem_watermark.py` | Nightly only: peak memory tracking |
| `baseline_cpu.json` | CI-runner baseline |
| `baseline_cuda_l40.json` | Self-hosted L40 baseline |
