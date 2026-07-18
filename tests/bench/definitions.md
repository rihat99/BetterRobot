# M6 benchmark definition

This file is the committed measurement contract for M6. Results are evidence,
not automatic default-lane decisions; every Warp default-on change remains an
owner review.

## CUDA gate and measurement host

The local gate passed on 2026-07-18. The default agent sandbox hides the
NVIDIA device nodes, so CUDA commands must run in the approved host context.
That access distinction caused the earlier false negative; it was not a Torch
or driver failure.

| Field | Value |
|---|---|
| Host GPUs | 8 × NVIDIA RTX 6000 Ada Generation |
| Measurement device | physical GPU 4 / UUID `GPU-ba51bb8b-da02-ea99-e99e-350952268322`, exposed as logical `cuda:0` |
| Compute capability / memory | sm_89 / 50,896,961,536 bytes |
| NVIDIA driver | 560.35.03 (CUDA 12.6 reported by `nvidia-smi`) |
| Torch | 2.13.0+cu126; build CUDA 12.6 |
| Warp | 1.15.0; toolkit 12.9; driver API 12.6 |
| Kernel cache | `/tmp/betterrobot-warp-m6-bench/1.15.0` |
| Host | `robotics2-ESC8000-E11`; Linux 6.8.0-110-generic; Python 3.11.15 |
| CPU | Intel Xeon Platinum 8570, one intra/inter-op thread for canonical CPU runs |

The gate included `nvidia-smi`, a finite Torch CUDA matrix multiply, Warp
initialization and FK kernel launch, a trivial `torch.cuda.CUDAGraph`
capture/replay, and the mixed Torch/Warp current-stream graph test. CI remains
manual-only by explicit owner request, so no GPU runner or scheduled job is
claimed.

## Canonical matrix

Set seed `20260718`, `CUDA_VISIBLE_DEVICES=4`, and one Torch intra/inter-op
thread. Record CPU affinity; the first measured FK artifact inherited host
affinity `0-223`, while future canonical CPU runs should pin an explicit core.
Use fresh, per-case TorchInductor and Warp cache directories whenever a field
is described as a cold compile.

The complete T6.1 matrix is three models (fixed-base Panda, free-flyer Panda,
and the 25-joint SMPL-like model) × three operations (FK, RNEA, and public IK)
× four batches (`B ∈ {1, 16, 256, 4096}`) × CPU/CUDA × eager/compiled:
144 selectors. Public IK uses one unbatched target at B=1 and independently
batched targets at B>1. FP32 is the canonical performance dtype; FP64 is a
correctness and numerical-reference lane where required by the kernel matrix.

Use 20 untimed warmups and 100 synchronized samples. Report median, inclusive
Q1/Q3, min, max, first-call latency, and peak allocated CUDA memory. CUDA wall
timings must synchronize before and after each sample. Solver cases use a
fixed iteration budget so early convergence cannot change the work. Record
Torch compilation, Warp module compilation/first launch, and CUDA graph
recording separately. A true per-shape Torch cold start requires a fresh
process and cache directory; later shapes in one dynamically compiled process
must not be labelled cold.

## Committed evidence

- `baselines/warp_fk_cuda_rtx6000_ada_b*.json` are the known-good SMPL FK
  cases from the host above. Each batch ran in its own fresh process with
  distinct empty TorchInductor and Warp caches, comparing opt-in fused Warp
  with full-graph compiled Torch. All four were measured from clean commit
  `c0560e3c16ee974a2bf6a8d09c618b45a5311163`.
- `baselines/m6_torch_filtered_smpl_b1_rtx6000_ada.json` is a filtered B=1
  cross-device run of the new Torch baseline harness. It records 10 successful
  SMPL FK/RNEA/public-IK rows and two honest `UNSUPPORTED` compiled-public-IK
  rows; every evaluated eager/compiled parity and input-identity check passes.
  It was measured from the same clean commit.
- `baselines/trajopt_sparse_cpu.json` is the separate M5 trajectory study and
  is not an M6 GPU baseline.
- `baseline_cpu.json` is retained as legacy pytest-benchmark scaffolding. Its
  placeholder status means it is not a regression gate.

The current artifacts do not complete T6.1: the filtered Torch run covers only
SMPL B=1, compiled public IK is not an expressible full-graph workload, and
the remaining batches/models plus graph-record timing remain open. The files
do include raw samples, allocator peaks, isolated cold compilation, exact input
fingerprints, and numerical lane validation.
