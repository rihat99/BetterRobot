# Order 02 — Results: batched `joint_jacobians_raw`

Batched rewrite of `joint_jacobians_raw`: the per-joint Python loop (parent
copy + per-joint `joint_motion_subspace(q)` + small matmuls + in-place index
writes) is replaced by three batched stages — one shared full-width world
column table built in a single sweep, then a static supports-derived mask, then
mimic reduce. Static artifacts (owner map, gather index, support mask) are
memoised on the `ModelStructure` instance (`_joint_jacobian_plan`), rebuilt when
`ModelStructure.to()` mints a fresh instance.

## Protocol

FK round protocol. GPU = one idle card pinned via `CUDA_VISIBLE_DEVICES=2`
(NVIDIA, batch 256, fp32). Timing = median of 30 iters after 5 warmups,
`torch.cuda.synchronize` around the timed region. Launches = `cudaLaunchKernel`
events from `torch.profiler` for one `joint_jacobians_raw` pass (joint poses
precomputed under `no_grad`), divided by `njoints`. `fwd` = the Jacobian pass
alone; `fwd+bwd` = `q → forward_kinematics_raw → joint_jacobians_raw → sum →
backward` (grads to `q` through the whole chain, so FK sits in the graph both
before and after — identical on both sides, so the delta is the Jacobian pass).
Models = programmatic free-flyer-root + spherical-chain zoo approximating the
SMPL-25 / SMPLX-56 / MHR-203 DOF counts, plus Panda. Bench script lives in the
session scratchpad (`bench_jac.py`), not the repo.

## GPU (batch 256, fp32)

Launches are the **total for the whole Jacobian pass** (constant, independent of
`njoints`) with the per-joint figure alongside.

| Model   | njoints | nv  | launches before | /joint | launches after | /joint | fwd before | fwd after | fwd+bwd before | fwd+bwd after |
|---------|---------|-----|-----------------|--------|----------------|--------|------------|-----------|----------------|---------------|
| smpl25  | 26      | 78  | 1324            | 50.9   | 52             | 2.00   | 14.204 ms  | 0.614 ms  | 64.531 ms      | 16.929 ms     |
| smplx56 | 57      | 171 | 2967            | 52.1   | 52             | 0.91   | 39.338 ms  | 0.778 ms  | 164.444 ms     | 28.586 ms     |
| mhr203  | 68      | 204 | 3550            | 52.2   | 52             | 0.76   | 47.349 ms  | 0.842 ms  | 192.867 ms     | 31.280 ms     |
| panda   | 14      | 8   | 486             | 34.7   | 52             | 3.71   | 7.392 ms   | 0.480 ms  | 44.787 ms      | 13.230 ms     |

Speedups (fwd / fwd+bwd): smpl25 23× / 3.8×, smplx56 51× / 5.7×,
mhr203 56× / 6.2×, panda 15× / 3.4×.

## CPU small-model check (Panda, batch 1, fp32)

| Case      | fwd before | fwd after | fwd+bwd before | fwd+bwd after |
|-----------|------------|-----------|----------------|---------------|
| panda-cpu | 2.291 ms   | 0.131 ms  | 12.501 ms      | 2.656 ms      |

**No CPU regression.** The batched pass is faster even for a tiny model on CPU
(fwd 17×, fwd+bwd 4.7×): a handful of batched ops beat ~14 joints of Python-loop
glue and per-joint `.to()` allocations even without launch pressure.

## Notes

- **Launch count is now constant (52) across all four models** — exactly the
  "independent of `njoints`, O(10) per pass" target. Before, launches scaled at
  ~50/joint (Panda's lower 34.7 reflects its many zero-DOF fixed joints, which
  skip the loop body). 52 total is the fixed op sequence (`so3.to_matrix`,
  `hat_so3`, subspace cast, three batched matmuls, `cat`, transpose/flatten/
  `index_select`, mask multiply, mimic reduce), each expanding to a few kernels.
- `fwd+bwd` after is now FK-dominated (the Jacobian pass and its backward are
  cheap), so its speedup is smaller than `fwd`'s — expected; FK is out of scope
  this round.
- Behavior unchanged: full CPU gate 1697 passed / 61 deselected, focused
  `tests/kinematics tests/test_pinocchio` 205 passed, zero tolerance changes.
- Numbers vary a few percent run-to-run (shared host); the launch counts and
  order-of-magnitude wins are stable.
