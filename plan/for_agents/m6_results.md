# M6 results — Warp fast path, CUDA graphs, and implicit solve

**Status:** incomplete and hardware-blocked on branch `dev` (2026-07-17).
The CUDA prerequisite failed, so no GPU production, performance, capture, or
default-on claim is made. The independent CPU-developable T6.11 subset is
implemented conservatively.

## Hardware gate

- `nvidia-smi` could not communicate with an NVIDIA driver.
- Torch is `2.13.0+cu126` (CUDA build `12.6`), but
  `torch.cuda.is_available()` is false and the device count is zero.
- Warp is `1.15.0`. With `WARP_CACHE_PATH=/tmp/betterrobot_warp_probe`, it
  initializes its CPU device and reports toolkit `12.9`, no CUDA driver, and
  zero CUDA devices. The temporary cache separated the hardware failure from
  the read-only default home-cache path.
- CI remains `workflow_dispatch`/manual-only. No nonexistent GPU runner was
  wired into it.

Updating Torch or regenerating the lock cannot fix this gate: the installed
Torch already has a CUDA build, while the machine exposes no working CUDA
device/driver.

## Delivered: conservative T6.11 subset

- Added `LevenbergMarquardt.solve` / inherited `GaussNewton.solve` with
  `differentiate="detached"` by default and explicit `"implicit"` opt-in;
  existing `run` remains always detached and compatible.
- Added a first-order custom autograd boundary that retains no forward
  iteration graph and returns gradients only to names in
  `Problem.differentiable_external_parameters`. Initial values, warm state,
  bounds/masks, and solver hyperparameters receive no implicit gradient.
- Recomputes the terminal exact robust objective in a reduced tangent chart,
  maps ambient output cotangents through Euclidean/SO(3)/SE(3)/RobotConfig
  retractions, eliminates stable active-bound axes, and solves the undamped
  adjoint system with Cholesky or a full-rank/residual-checked least-squares
  fallback.
- Strictly rejects a whole requested batch on invalid terminal status/KKT,
  unstable active sets, Huber kinks, terminal-manifold quaternion
  representatives at the absolute-pi principal-log cut, nonfinite systems, or
  singular adjoint systems. It also rejects identity collisions between
  optimized inputs and external parameters, and declared differentiable
  parameters disconnected from terminal optimality. The custom backward is
  first-order only.
- Adds `ImplicitDiffConfig` and `ImplicitDifferentiationError`. Dense backward
  is capped at 512 tangent coordinates by default; matrix-free is rejected and
  a small banded forward requires an explicit capped dense-oracle opt-in.

## Verification

- Focused implicit/LM/bounds/robust/temporal/contract verification:
  **158 passed**. Coverage includes fp32/fp64 closed form, 16-way and
  multi-axis batches, shared-parameter reduction, SO(3), `RobotConfig` with
  `nq != nv`, stable active bounds, exact Huber behavior,
  Huber-kink/absolute-pi-terminal/singular/invalid-batch rejection, tensor-role
  separation, disconnected-parameter refusal, and dense-vs-small-banded parity.
- Full CPU gate: **1,522 passed, 2 skipped, 3 deselected** in 62.98 s.
- Scoped Ruff and mypy, offline `uv lock --check`, and `git diff --check` pass.
- Sphinx HTML builds successfully. Its four warnings are only the unavailable
  Python/Torch/NumPy/Trimesh intersphinx inventories in the offline sandbox.

The canonical M1 eager-CPU definition was rerun on the same Xeon 8570 class,
one thread, Panda fp32, 20 warmups, and 100 samples. Medians were 3.367 ms
(FK B=1), 9.401 ms (RNEA B=1), 3.674 ms (FK B=64), and 10.918 ms (RNEA B=64).
They are 1.8–4.6% slower than the owner-reviewed M1 measurements and remain
inside the documented 20% advisory window. This is a same-host advisory
observation, not a CI or GPU claim.

## Deferred tasks and deviations

1. **T6.0–T6.10 are open.** The ordered GPU gate blocks baselines, the Warp Lie
   library, FK/Jacobian/pose/dynamics kernel graduation, integrate/difference
   decisions, CUDA graph certification, and formulation performance studies.
   The existing FK Warp path remains a prototype; no CPU result promotes it.
2. **T6.6 has a second blocker.** M4 left collision residuals as stubs under the
   repository boundary, so there is no Torch capsule-collision oracle to
   kernelize even after hardware becomes available.
3. **T6.12 is open.** No cuRobo/JAX-class environment, definition measurement,
   or external performance ratio was created without the required GPU box.
4. **T6.11 is partial, not full milestone acceptance.** There is no true
   banded/operator implicit backward or returned per-element gradient-quality
   result. A small banded forward may explicitly use a dense oracle; long
   trajectories cannot silently densify.
5. Tensor item weights and kernel attributes have no stable named binding, so
   implicit backward treats them as static and never infers a parameter role
   from Python object identity. Declaring only the same tensor object as an
   external parameter raises the disconnected-parameter error; callers must
   read differentiable scales from named residual/provider context. Direct
   differentiable `ModelValues` reconstruction through `RobotStateProvider` is
   also deferred.
6. Active bounds use the existing projected feasible retraction and then
   eliminate active axes, rather than a separate unprojected fixed-active
   chart. Stable lower/upper and fixed-coordinate behavior is tested; broader
   constraint-normal semantics remain future work.
7. Huber-kink detection and the terminal quaternion-representative check at
   absolute pi are implemented. A principal-log cut of a relative rotation
   hidden inside generic residual/provider code (even when the optimized
   quaternion itself is near identity), or nonsmoothness in a custom robust
   kernel, cannot be certified automatically. Those domains remain an author
   responsibility rather than a claimed guarantee.
8. The plan requested implicit-vs-unrolled comparisons for the small
   fp32/fp64 and Huber cases plus a Panda target-pose finite-difference check.
   The landed tests use closed-form/analytic gradients instead of the unrolled
   oracle, and manifold coverage uses analytic SO(3)/SE(3)/RobotConfig
   problems. The unrolled comparisons remain deferred; Panda task-facade
   implicit differentiation is also deferred because `solve_ik` remains
   intentionally detached.
9. Scoped mypy exposed narrow typing mismatches in the existing M5 solver
   route (Torch VJP stubs, dense/structured unions, and optional operator block
   metadata). They were corrected without changing the numerical algorithm.
10. BHF, BVR, and every path outside this repository were neither read nor
   modified. CI remains stopped/manual-only as requested.
