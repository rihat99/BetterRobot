# 02 — Vision

## What BetterRobot is

A **PyTorch-native differentiable rigid-body toolkit** — kinematics,
dynamics, and manifold-aware nonlinear least squares — that other projects
build on for years. We already know three consumers, and together they
define what the library must be good at:

- **BetterVideoReconstruction / BetterHumanForce** — estimation from video:
  fit articulated models to keypoints, meshes, point clouds, and physics
  consistency terms. The parameters are heterogeneous (poses, shapes,
  cameras, contact forces) and are optimized jointly, in staged pipelines.
- **better_human** — parametric human body models (the SMPL family). Their
  kinematics, dynamics, and Jacobians should come from BR natively, while
  the parametric specifics (blend shapes, LBS, regressors) stay outside BR.
- **Classic robotics** — IK, trajectory optimization, and retargeting for
  fixed-base and floating-base robots.

Notice what these three have in common. It is not "robots". It is
**batched differentiable optimization over articulated structures**. That
is the product.

"PyTorch-native" describes the **surface** of the library: every public API
takes and returns torch tensors, and autograd flows through everything.
Inside, the hot computations run as Warp kernels on the GPU, with a torch
reference implementation everywhere else — the two-lane compute model
described in `03 §2`.

## Scope fence

Inside BR:
- Lie/SE(3) math, spatial algebra, kinematic trees, Featherstone dynamics.
- The residual/cost/solver stack, generic over *parameter blocks* — not
  just over q.
- A residual library that covers robotics **and** the estimation residuals
  the consumers actually optimize: projection, point-cloud terms, priors,
  limits, smoothness, contact consistency.
- Model construction: URDF/MJCF parsing and the programmatic builder.
- Visualization (viser-based), as an optional extra.

Outside BR (a hard fence, the way optax owns no training loop):
- Parametric body models — blend shapes, LBS, joint regressors, model file
  formats. These belong to better_human. BR provides the hooks (parametric
  Model values, marker frames, inertia-from-mesh) but never the body model
  itself.
- Task pipelines, datasets, and camera models beyond a simple pinhole
  projection residual. These belong to the consumer projects.
- Simulation. Contact dynamics and physics integrators are not BR's job —
  BR does analysis and optimization, not a simulator. If that ever changes,
  it changes by deliberate decision, not by scope creep.

## Design principles

1. **Batched by default, for real.** Every public code path accepts
   `(B..., ·)` shapes, and solvers run B independent problems with
   per-element damping and per-element convergence. A claim like this is
   either enforced by tests or deleted from the docs — no third option.
2. **Autograd is the substrate; analytic is the accelerator.** Every
   operation must be cleanly differentiable — including at singular points,
   with gradient checks *at* θ=0, not just near it. Analytic Jacobians are
   opt-in speedups. CI asserts they really are faster; they are never
   silently replaced by finite differences.
3. **Functions over frameworks — two lanes, one contract.** The unit of
   architecture is the whole-pass function: tensors in, tensors out
   (`forward_kinematics_raw` style). Each hot pass has a torch reference
   implementation (the testing oracle, the CPU fast path, the only path for
   second-order gradients) and a Warp kernel (the GPU fast path). Both sit
   behind one signature and one memory-layout contract, and a plain `if`
   at the call site picks between them. No dispatch layers, no registries,
   no Protocols. Kernel boundaries are whole passes — the FK sweep, RNEA,
   a residual with its Jacobian — never single small ops. And warp-lang is
   the only kernel language: no hand-written CUDA (cuRobo's dual-stack
   maintenance cost is the counter-example we are avoiding).
4. **Pure solver state.** Optimizers follow the jaxopt/optax pattern:
   `init_state / update / run`, with state as a plain value. This lets
   consumers own the outer loop (their #1 adoption request), warm-start
   from the previous video frame, and it keeps every step friendly to
   `torch.compile`, vmap, and CUDA-graph capture. Capturing solver inner
   loops is a planned feature — so `update` is sync-free and
   allocation-free by contract, not by luck.
5. **Structure static, values dynamic.** The model's topology, joint kinds,
   and index maps are frozen Python. Every numeric quantity — placements,
   inertias, limits, frames — is a device tensor that may be batched,
   swapped, and differentiated. This single principle is what makes
   better_human possible.
6. **No speculative code.** No stubs shipped as API, no config options that
   crash when selected, no deprecation shims before 1.0, no "flexibility"
   without a second caller. The public roadmap page is generated from the
   code, not maintained by hand.
7. **Docs that execute.** Every code snippet in docs/ runs in CI. A doc
   that can drift is a doc that will drift.
8. **Own the numerics.** LM damping (Madsen–Nielsen), robust-kernel IRLS,
   real bounded steps (projected/reflective methods, not clamp-after-the-
   fact), dtype discipline (no silent float64→float32), and per-dtype
   epsilon tables are the library's responsibility — not something users
   should work around.

## Where this goes (18-month horizon)

- BVR and BHF delete their hand-rolled optimization engines and loss
  libraries; their stage configs simply construct BR problems.
- better_human becomes a thin parametric layer: betas → BR Model values;
  all kinematics, dynamics, Jacobians, and optimization come from BR.
- A two-lane core, Warp-first on GPU: warp-lang whole-pass kernels for the
  tree-scan passes and geometry, CUDA-graph-captured solver loops, and a
  compile-clean torch lane as the oracle and the CPU fast path. The
  performance target: cuRobo-class batched-IK throughput and
  JAX-library-class (mjx/pyroki) FK and dynamics sweeps, tracked by
  benchmarks committed to the repo. Each kernel becomes the default only
  when it beats the compiled torch lane. A measured gap versus cuRobo's
  robot-specialized hand-written CUDA is acceptable and stated openly,
  never hidden — BR buys maintainability with warp-lang, and closes the
  distance with kernel fusion, graph capture, and (if needed) static
  specialization.
- A public release with a small, honest API surface.
