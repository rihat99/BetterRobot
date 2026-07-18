# M6 results — Warp fast path, CUDA graphs, and implicit solve

**Status:** incomplete; selected T6.0, T6.1, T6.3, T6.9, T6.11, and T6.12
slices are delivered on branch `dev` (2026-07-18). No Warp default changed,
and no complete-M6 claim is made.

## Corrected CUDA gate

The earlier “CUDA is broken” conclusion was wrong. The default agent sandbox
hid the NVIDIA device nodes; the host GPU stack was healthy.

- Host: 8 × NVIDIA RTX 6000 Ada Generation, driver 560.35.03, 49,140 MiB
  reported per device, compute capability 8.9.
- Measurement device: physical GPU 4,
  `GPU-ba51bb8b-da02-ea99-e99e-350952268322`, exposed as logical `cuda:0`.
- Locked Torch: `2.13.0+cu126`, build CUDA 12.6. Warp: 1.15.0, toolkit
  12.9, driver API 12.6.
- Torch matrix multiply, a trivial `torch.cuda.CUDAGraph`, Warp
  initialization/kernel launch, and mixed Torch/Warp graph replay all pass in
  the approved host context.

The correction is recorded in the M6 plan, standing agent rules, and
`tests/bench/definitions.md`. The fictional L40 placeholder was deleted.

## Delivered

### T6.1 benchmark contract and partial evidence

`benchmarks/m6_baseline.py` defines 144 isolated selectors across fixed
Panda, free-flyer Panda, and SMPL; FK, RNEA, and public IK; CPU/CUDA;
eager/compiled; and B=1/16/256/4096. Every case uses a fresh child process and
TorchInductor cache, fixed input seed/fingerprints, raw synchronized samples,
cold-call timing, and CUDA allocation peaks. Unsupported public compiled IK is
reported as `UNSUPPORTED`; it is never replaced by a different workload.

The committed filtered SMPL B=1 result was measured from clean commit
`c0560e3c16ee974a2bf6a8d09c618b45a5311163`. It has 10 successes, two
compiled-IK `UNSUPPORTED` rows, zero errors/OOM/timeouts, and all evaluated
input and eager/compiled numerical checks passing. Representative medians and
cold first calls are:

| Operation | CPU eager | CPU compiled | CUDA eager | CUDA compiled |
|---|---:|---:|---:|---:|
| FK | 5.620 ms | 0.631 ms (49.44 s cold) | 16.136 ms | 0.653 ms (23.88 s cold) |
| RNEA | 8.732 ms | 2.453 ms (82.46 s cold) | 26.707 ms | 4.913 ms (71.81 s cold) |
| Public IK, 5 iterations | 132.514 ms | unsupported | 336.789 ms | unsupported |

This is filtered evidence, not the canonical 144-case baseline.

### T6.3 FK graduation evidence

The opt-in fused Warp FK lane now has real CUDA coverage for fp32/fp64,
fixed/free bases, branched and >16-joint models, multi-axis/value batches,
shared values, q and placement VJPs, a numerical zero-angle gradcheck,
current-stream ordering, and graph replay.

Four fresh-process, fresh-cache SMPL CUDA artifacts compare forward-only Warp
with full-graph compiled Torch:

| Batch | Compiled Torch median | Warp median | Warp speedup |
|---:|---:|---:|---:|
| 1 | 0.712 ms | 0.541 ms | 1.31× |
| 16 | 1.424 ms | 0.582 ms | 2.45× |
| 256 | 1.253 ms | 0.630 ms | 1.99× |
| 4096 | 1.246 ms | 0.625 ms | 1.99× |

Each artifact includes 100 raw samples, inclusive quartiles/IQR, measured
cold calls, allocator peaks, exact GPU UUID/driver mapping, cache scope, and
the same clean source commit.

The backward path still recomputes through Torch and was not timed, so the
lane remains opt-in pending owner review.

### T6.9 internal graph-capture slice

An experimental internal
`better_robot.optim._graph_executor.GraphExecutor` now provides tensor-pytree
CPU/disabled eager fallback, side-stream warmup, lazy record/replay, stable
input copies, cloned outputs, synchronized reset, record-time counters, and
controlled signature/caller-stream re-recording. It is deliberately not
exported from `better_robot.optim`.

CUDA tests cover nonlinear LM update groups with changing jacrev-derived work,
resize, two-window 100-replay memory stability, public Warp selection, mixed
Torch/Warp capture, and capture-time hard errors. Unsafe offsets,
stride-zero/internal-overlap views, overlapping leaf storage, grad-bearing
inputs, and grad-bearing outputs are rejected before unsafe reuse. Scalar
Python residual weights now use capture-safe device fills.

Only explicit arguments are signatured. Closure tensors/configuration must
remain alive and unchanged until `reset()`; changing targets belong in
explicit inputs. This promotion hazard and the lack of a production solver
caller are why the helper stays internal.

### T6.11 conservative implicit solve

The previously delivered `LevenbergMarquardt.solve` (and inherited GN path)
keeps detached execution as the default and offers explicit
`differentiate="implicit"` for declared external parameters. Its first-order
boundary works in reduced tangent charts, handles stable active bounds and
robust objectives, rejects invalid whole batches and nonsmooth/singular
conditions strictly, and caps dense backward size. Initial values, warm state,
bounds/masks, and solver hyperparameters do not receive implicit gradients.

### T6.12 definition first

`tests/bench/external/definition.md` fixes the Panda target set, success
oracle, 32-iteration work budget, cold/warm timing, memory schema, provenance,
and competitor caveats before any cuRobo or JAX-class measurement.

## Verification

- CUDA Graph/Warp suite: **28 passed** on physical GPU 4.
- Full CPU, non-benchmark/non-CUDA suite: **1,543 passed, 2 skipped,
  28 deselected**.
- Benchmark harness/schema tests: **19 passed**.
- Scoped Ruff passes for every changed Python file. Scoped strict mypy passes
  for the new GraphExecutor and changed Warp bridge production modules.
- Offline `uv lock --check` and `git diff --check` pass.
- Sphinx HTML builds successfully; its four warnings are only unreachable
  external intersphinx inventories in the network-isolated sandbox.
- The earlier same-host M1 eager-CPU advisory rerun remained within its
  documented 20% window; it is not a CI or GPU claim.

## Deviations and open work

1. **CI remains stopped/manual-only by owner request.** No GPU runner job was
   enabled, so the full T6.0 acceptance checkbox remains open despite the local
   host gate passing.
2. **T6.1 is partial and the prescribed ordering was not fully met.** The
   144-case definition/harness exists, but only four Warp FK cases and one
   filtered SMPL B=1 Torch result were measured. Panda, remaining batches,
   full IK modes, and graph-record costs remain open. The existing M1 FK
   graduation work proceeded before the complete baseline.
3. **T6.2, T6.4–T6.8, and T6.10 remain open.** No Warp Lie library, Jacobian,
   pose-residual, dynamics, integrate/difference, or formulation kernel study
   was added. T6.6 is additionally blocked because collision remains stubs and
   provides no Torch oracle.
4. **T6.3 is not default-on.** Its CUDA correctness matrix and forward-only
   evidence landed, but Torch-recompute backward performance was not measured
   and the owner has not approved a device/dtype default flip.
5. **T6.9 is partial.** Public `run` remains eager; there is no persistent
   inner/outer fixed-trip driver, B=128 end-to-end IK q/status proof, capture
   speed benchmark, graph-record artifact, or runtime residual-eligibility
   warning/fallback. Warp proof is forward-only. Automatic signature re-record
   differs from the plan's explicit resize event, and closure state requires
   manual reset.
6. **T6.11 remains a conservative subset.** There is no true banded/operator
   implicit backward, returned per-element gradient-quality result,
   implicit-vs-unrolled oracle, or Panda task-facade finite-difference check.
   Small banded forwards may use an explicitly capped dense oracle; long
   trajectories never silently densify. Generic residual code still owns
   principal-log and custom nonsmoothness declarations.
7. **T6.12 has definition only.** No competitor environment or external
   measurement was created.
8. BHF, BVR, and every path outside this repository were neither read nor
   modified. No CI trigger was re-enabled.
