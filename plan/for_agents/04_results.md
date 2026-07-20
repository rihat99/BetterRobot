# Order 04 results — Warp FK and RNEA

Status: complete on `dev` (2026-07-20).

## Delivered

- Panda and chained mimic models now run fused Warp FK through the canonical
  reduced-to-full `q_expansion`/`q_offset` tables. The unused `nqs` custom-op
  argument is gone.
- Explicit FK and RNEA Warp requests now warn once with a concrete decline
  reason. A decline during CUDA graph capture is a hard error, and internal
  bridge import failures are no longer mistaken for a missing optional Warp
  runtime.
- Added a dedicated dynamics bridge and one-launch, forward-only Warp RNEA
  sweep: fused FK, velocity/acceleration propagation, and reverse force
  accumulation. It supports Panda and SMPL-like topologies, mimic q/v/a and
  tau reduction, gravity, optional external forces, model-value batching, and
  CUDA graph replay.
- RNEA backward recomputes canonical Torch RNEA. Its VJP covers q, velocity,
  acceleration, external force, joint placement, body inertia, and gravity;
  first- and second-order derivative parity is tested.
- The tiny one-joint IK example now defaults to CPU and explains its
  launch-bound GPU behavior. Compute-seam, dynamics, installation,
  performance, packaging, contracts, API, and changelog documentation now
  describe both optional fused lanes.

## Verification

- Full CPU gate: **1,516 passed, 2 skipped, 44 deselected**.
- CUDA gate on pinned GPU 2: **43 passed, 1,519 deselected**.
- Complete Warp directory on pinned GPU 2: **60 passed**.
- RNEA CUDA suite: **23 passed**, including the 16-cell Panda/SMPL ×
  fp32/fp64 × B=1/4096 × with/without-`fext` matrix.
- Optional-import contracts plus CPU Warp prototype: **145 passed**.
- Documentation tests: **27 passed**; strict Sphinx HTML build succeeded;
  Sphinx doctest: **30 passed**.
- All Order 04 Python files pass Ruff check and format check;
  `git diff --check` passes. Warp kernel modules contain no Torch import or
  call, and the dead `nqs` ABI spelling is absent from the Warp paths.

CUDA parity includes forward outputs, all seven differentiable RNEA inputs,
float32 finite-difference gradcheck, higher-order q gradients, frame-induced
execution batching, and graph replay. The largest observed float32 RNEA error
in the wider parity audit was `6.10e-05` (SMPL with `fext`); float64 was
`3.41e-13`.

## Timings

Both tables use float32 on an NVIDIA RTX 6000 Ada Generation, Torch
2.13.0+cu126 / CUDA 12.6, fullgraph Inductor, 10 warmups, then the median of
50 host-wall samples individually bracketed by `torch.cuda.synchronize()`.
Times are steady-state forward-only milliseconds.

### Forward kinematics

| Model | Batch | Eager Torch | Compiled Torch | Warp | Warp max error |
|---|---:|---:|---:|---:|---:|
| Panda | 1 | 8.227498 | 0.498078 | 0.591270 | 6.95e-08 |
| Panda | 4,096 | 5.210226 | 0.456173 | 0.366627 | 6.95e-08 |
| SMPL-like | 1 | 15.014071 | 0.795019 | 0.601797 | 0 |
| SMPL-like | 4,096 | 18.697170 | 1.686543 | 0.727610 | 0 |

Compiled Torch is **1.19× faster** than Warp for Panda B=1. Warp is
1.24×–2.32× faster than compiled Torch in the other three cells and
13.9×–25.7× faster than eager Torch in all four.

Raw artifact: `/tmp/order04_fk_timings_gpu3_final.json`; SHA-256
`9af2e9a35cdf9da00bbeb9e80fb92e4ded7f6c851f5ec7c4034b5be19d5818b6`.

### Inverse dynamics

| Model | Batch | Eager Torch | Compiled Torch | Warp | Warp max error |
|---|---:|---:|---:|---:|---:|
| Panda | 1 | 25.861715 | 2.614348 | 1.095416 | 3.05e-05 |
| Panda | 4,096 | 15.846244 | 2.754012 | 0.705283 | 3.05e-05 |
| SMPL-like | 1 | 44.338067 | 5.135494 | 1.088642 | 2.38e-07 |
| SMPL-like | 4,096 | 48.635577 | 7.469487 | 1.355488 | 0 |

Compiled Torch is **2.39×–5.51× slower** than Warp, so it is not within 2×
in any measured RNEA cell. Warp is 22.5×–40.7× faster than eager Torch.

Raw artifact: `/tmp/order04_rnea_timings.json`; SHA-256
`a3acf78d3630df5f1a15ed8c313d059023c6bea8018cd108a222b0d0db43dfef`.

The artifacts identify commit `56b5247` with `dirty: true`: they measure the
final Order 04 working tree atop that committed Order 03 base, before this
order's report and commit. No performance number is attributed to the base
commit alone.

## Physical source-line accounting

| Scope | Before | After | Delta |
|---|---:|---:|---:|
| `kinematics/_warp_bridge.py` | 367 | 400 | +33 |
| `kinematics/_warp_kernels.py` | 296 | 468 | +172 |
| `kinematics/forward.py` | 323 | 350 | +27 |
| `dynamics/rnea.py` | 321 | 371 | +50 |
| `dynamics/_warp_bridge.py` | 0 | 584 | +584 |
| `dynamics/_warp_kernels.py` | 0 | 749 | +749 |
| **All `src/**/*.py`** | **20,679** | **22,294** | **+1,615** |

The new RNEA CUDA test file is 366 lines. The two existing FK test files grew
from 874 to 1,081 lines net, covering mimic, degenerate quaternion, fallback,
import, and custom-op ABI regressions.

## Deviations and findings

- No public numerical behavior intentionally changed. A dedicated dynamics
  bridge was chosen instead of extending the FK bridge: RNEA has seven
  differentiable inputs, six outputs, independent broadcast maps, and distinct
  VJP plumbing; only the validated transform helpers are shared.
- The requested f32/f64 collapse was attempted with Warp generic functions.
  Float64 transform/quaternion constructors resolved through float32 during
  code generation, so the readable typed twins were retained as the plan
  permits.
- The plan's claim that Warp spatial builtins map 1:1 was incomplete. Warp
  spatial vectors are angular-first while BetterRobot is linear-first, so the
  kernel converts explicitly. Cross, cross-dual, and dot remain Warp builtins.
- Warp transform transport also differs from the Torch reference for
  off-unit placement quaternions. RNEA therefore uses explicit
  Torch-polynomial rotation/transport helpers, which made placement gradcheck
  exact instead of relying on Warp's off-manifold convention.
- Warp quaternion normalization maps zero to identity, while Torch divides by
  `max(norm, 1e-8)`. FK now implements the Torch rule exactly; zero and tiny
  spherical/free-flyer regressions cover the discovered edge case.
- The plan described mimic multiplier/offset reads, but the canonical model
  representation after Order 03 is the packed q/v expansion tables. Using
  those tables also handles chained mimic relationships without a parallel
  source of truth.
