# Autodiff vs analytic Jacobian through BetterRobot FK (measured 2026-10-06)

Env: uv env of dev/BetterRobot; CPU = shared host; GPU = RTX PRO 6000 Blackwell #0, **shared GPU, indicative only**: 82-100% util by another user's job during all runs (gpu_free.py found no free GPU). Residual = se3.log(T_target^-1 T_ee) of one frame, Jacobian d r / d delta at delta=0 with q' = model.integrate(q, delta). float64 checks (B=8), float32 timing, median of 20 after 3 warm-ups, cuda.synchronize. Models: Panda (nq=nv=8, body_panda_hand), Unitree G1 29-dof + free flyer (nq=36, nv=35, body_left_rubber_hand). Scripts: subagents/proto_autodiff_jacobian_{bench,fdcheck,tables}.py; logs: artifacts/proto_autodiff_*.log. GN step = J, JtJ + lambda I, Cholesky solve (lambda=1e-6 Panda, 1e-2 G1: 1e-6 failed Cholesky in float32 for rank-6 JtJ).

## Findings
- **No variant failed.** vmap(jacfwd), vmap(jacrev), jacfwd with shared tangent, jacrev of batch-sum, autograd is_grads_batched, 6-pass autograd, forward-AD tangent batching all run through the public `br.forward_kinematics` (validation included) and the raw FK, on both models, CPU and CUDA. All 7 agree to ~1e-10 and match central FD to 1e-9.
- **The analytic path is wrong at large errors, not autodiff.** Analytic (and `PoseResidual.jacobian()`) minus FD: Panda 5.2e-3 at |err|=3.5, G1 3.7e-2 at |err|=5.1; 5e-7 / 1.3e-7 near target. Cause: `src/better_robot/lie/tangents.py:187-204` truncates the Jr series at N=9 and claims "error < eps for |xi|<6"; the 10th term at |xi|=3.5 is ~7e-3. Needs a closed form.
- Speed: autodiff is 2-20x slower than analytic eager (tables). Best eager autodiff = jacrev-family (jacrev_sum / vmap_jacrev / autograd_batched), since m=6 << nv. jacfwd variants are 2x slower than jacrev here. On GPU everything is launch-bound (time flat in B up to 4096). torch.compile of jacfwd_shared worked with no changes and is **faster than eager analytic** (and than compiled analytic) on GPU.
- Peak GPU memory grows with nv for forward modes (fwd_tangent_batch is the worst).

## Verdict
"Autodiff by default, analytic when provided" is viable functionally (no kinematics change needed for vmap/jacfwd/jacrev). Eager cost 2-5x on GPU, 2.5-20x on CPU (worst at B=1). Default: jacrev (rows <= cols) as the current problem.py:638 rule does; jacfwd only when dim > nv. To make it fast: compile the Jacobian program (works today), and fix right_jacobian_se3 before trusting analytic pose Jacobians.

## Tables (ms median, (x) = / analytic eager in same column; GPU adds peak MiB)

### proto_autodiff_cpu_panda.log

panda (fk=public)

| variant | B=1 | B=256 | B=4096 |
|---|---|---|---|
| residual | 1.07 ms (0.6x) | 4.35 ms (0.5x) | 45.23 ms (0.7x) |
| analytic | 1.80 ms (1.0x) | 8.62 ms (1.0x) | 63.49 ms (1.0x) |
| vmap_jacfwd | 23.43 ms (13.0x) | 57.52 ms (6.7x) | 206.25 ms (3.2x) |
| vmap_jacrev | 6.98 ms (3.9x) | 43.94 ms (5.1x) | 160.78 ms (2.5x) |
| jacfwd_shared | 21.30 ms (11.8x) | 55.54 ms (6.4x) | 212.04 ms (3.3x) |
| jacrev_sum | 5.86 ms (3.3x) | 46.08 ms (5.3x) | 183.57 ms (2.9x) |
| autograd_batched | 4.99 ms (2.8x) | 43.22 ms (5.0x) | 248.84 ms (3.9x) |
| autograd_loop | 13.01 ms (7.2x) | 51.06 ms (5.9x) | 364.35 ms (5.7x) |
| fwd_tangent_batch | 14.82 ms (8.2x) | 178.66 ms (20.7x) | 313.47 ms (4.9x) |
| gn_analytic | 3.02 ms (1.0x) | 15.84 ms (1.0x) | 94.26 ms (1.0x) |
| gn_jacfwd_shared | 23.04 ms (7.6x) | 66.46 ms (4.2x) | 261.99 ms (2.8x) |
| gn_vmap_jacfwd | 25.28 ms (8.4x) | 67.50 ms (4.3x) | 248.69 ms (2.6x) |

g1_freeflyer (fk=public)

| variant | B=1 |
|---|---|
| residual | 1.62 ms (0.8x) |
| analytic | 2.05 ms (1.0x) |
| vmap_jacfwd | 41.84 ms (20.4x) |
| vmap_jacrev | 11.53 ms (5.6x) |
| jacfwd_shared | 33.95 ms (16.6x) |
| jacrev_sum | 9.88 ms (4.8x) |
| autograd_batched | 8.29 ms (4.1x) |
| autograd_loop | 22.61 ms (11.0x) |
| fwd_tangent_batch | 24.97 ms (12.2x) |
| gn_analytic | 4.92 ms (1.0x) |
| gn_jacfwd_shared | 37.34 ms (7.6x) |
| gn_vmap_jacfwd | 47.42 ms (9.6x) |

### proto_autodiff_cpu_g1.log

g1_freeflyer (fk=public)

| variant | B=1 | B=256 | B=4096 |
|---|---|---|---|
| residual | 1.64 ms (0.8x) | 11.23 ms (0.7x) | 66.59 ms (0.6x) |
| analytic | 2.08 ms (1.0x) | 16.59 ms (1.0x) | 112.95 ms (1.0x) |
| vmap_jacfwd | 41.83 ms (20.2x) | 181.48 ms (10.9x) | 3441.51 ms (30.5x) |
| vmap_jacrev | 11.59 ms (5.6x) | 82.97 ms (5.0x) | 1472.89 ms (13.0x) |
| jacfwd_shared | 34.11 ms (16.4x) | 176.35 ms (10.6x) | 1755.08 ms (15.5x) |
| jacrev_sum | 9.91 ms (4.8x) | 90.07 ms (5.4x) | 1299.89 ms (11.5x) |
| autograd_batched | 8.33 ms (4.0x) | 101.50 ms (6.1x) | 1383.66 ms (12.2x) |
| autograd_loop | 22.49 ms (10.8x) | 129.35 ms (7.8x) | 557.34 ms (4.9x) |
| fwd_tangent_batch | 24.65 ms (11.9x) | 267.52 ms (16.1x) | 3342.62 ms (29.6x) |
| gn_analytic | 4.04 ms (1.0x) | 27.96 ms (1.0x) | 218.81 ms (1.0x) |
| gn_jacfwd_shared | 36.36 ms (9.0x) | 188.35 ms (6.7x) | 1859.03 ms (8.5x) |
| gn_vmap_jacfwd | 47.59 ms (11.8x) | 182.45 ms (6.5x) | 3720.05 ms (17.0x) |

### proto_autodiff_cuda_panda.log

panda (fk=public)

| variant | B=1 | B=256 | B=4096 |
|---|---|---|---|
| residual | 3.58 ms (0.5x), 0 MiB | 5.04 ms (0.7x), 3 MiB | 2.94 ms (0.4x), 41 MiB |
| analytic | 7.62 ms (1.0x), 0 MiB | 7.62 ms (1.0x), 3 MiB | 8.03 ms (1.0x), 41 MiB |
| vmap_jacfwd | 35.47 ms (4.7x), 0 MiB | 35.54 ms (4.7x), 23 MiB | 35.64 ms (4.4x), 374 MiB |
| vmap_jacrev | 15.32 ms (2.0x), 0 MiB | 15.31 ms (2.0x), 8 MiB | 17.52 ms (2.2x), 125 MiB |
| jacfwd_shared | 30.87 ms (4.1x), 0 MiB | 32.84 ms (4.3x), 23 MiB | 36.18 ms (4.5x), 377 MiB |
| jacrev_sum | 14.77 ms (1.9x), 0 MiB | 13.04 ms (1.7x), 8 MiB | 17.41 ms (2.2x), 129 MiB |
| autograd_batched | 15.89 ms (2.1x), 0 MiB | 18.71 ms (2.5x), 7 MiB | 18.57 ms (2.3x), 110 MiB |
| autograd_loop | 38.33 ms (5.0x), 0 MiB | 38.14 ms (5.0x), 3 MiB | 45.23 ms (5.6x), 56 MiB |
| fwd_tangent_batch | 25.52 ms (3.4x), 0 MiB | 24.34 ms (3.2x), 41 MiB | 26.95 ms (3.4x), 660 MiB |
| gn_analytic | 12.53 ms (1.0x), 0 MiB | 12.67 ms (1.0x), 3 MiB | 13.31 ms (1.0x), 42 MiB |
| gn_jacfwd_shared | 41.97 ms (3.3x), 0 MiB | 37.69 ms (3.0x), 23 MiB | 41.98 ms (3.2x), 376 MiB |
| gn_vmap_jacfwd | 44.19 ms (3.5x), 0 MiB | 40.43 ms (3.2x), 23 MiB | 44.99 ms (3.4x), 375 MiB |
| compiled_jacfwd_shared | 2.44 ms (0.3x), 0 MiB | 2.61 ms (0.3x), 8 MiB | 6.88 ms (0.9x), 132 MiB |
| compiled_analytic | 4.72 ms (0.6x), 0 MiB | 4.73 ms (0.6x), 1 MiB | 5.10 ms (0.6x), 23 MiB |

g1_freeflyer (fk=public)

| variant | B=1 |
|---|---|
| residual | 5.27 ms (0.7x), 0 MiB |
| analytic | 7.62 ms (1.0x), 0 MiB |
| vmap_jacfwd | 54.26 ms (7.1x), 1 MiB |
| vmap_jacrev | 29.08 ms (3.8x), 0 MiB |
| jacfwd_shared | 55.79 ms (7.3x), 1 MiB |
| jacrev_sum | 25.00 ms (3.3x), 0 MiB |
| autograd_batched | 26.97 ms (3.5x), 0 MiB |
| autograd_loop | 76.65 ms (10.1x), 0 MiB |
| fwd_tangent_batch | 39.36 ms (5.2x), 2 MiB |
| gn_analytic | 15.25 ms (1.0x), 0 MiB |
| gn_jacfwd_shared | 56.26 ms (3.7x), 1 MiB |
| gn_vmap_jacfwd | 60.95 ms (4.0x), 1 MiB |
| compiled_jacfwd_shared | 2.73 ms (0.4x), 0 MiB |
| compiled_analytic | 4.63 ms (0.6x), 0 MiB |

### proto_autodiff_cuda_g1.log

g1_freeflyer (fk=public)

| variant | B=1 | B=256 | B=4096 |
|---|---|---|---|
| residual | 4.32 ms (0.8x), 0 MiB | 5.36 ms (0.9x), 7 MiB | 5.84 ms (0.6x), 121 MiB |
| analytic | 5.23 ms (1.0x), 0 MiB | 5.84 ms (1.0x), 10 MiB | 8.99 ms (1.0x), 157 MiB |
| vmap_jacfwd | 51.97 ms (9.9x), 1 MiB | 54.98 ms (9.4x), 270 MiB | 128.12 ms (14.3x), 4313 MiB |
| vmap_jacrev | 26.06 ms (5.0x), 0 MiB | 29.00 ms (5.0x), 22 MiB | 42.74 ms (4.8x), 358 MiB |
| jacfwd_shared | 48.21 ms (9.2x), 1 MiB | 51.91 ms (8.9x), 269 MiB | 135.76 ms (15.1x), 4296 MiB |
| jacrev_sum | 25.31 ms (4.8x), 0 MiB | 23.78 ms (4.1x), 23 MiB | 42.45 ms (4.7x), 369 MiB |
| autograd_batched | 28.33 ms (5.4x), 0 MiB | 24.54 ms (4.2x), 20 MiB | 45.01 ms (5.0x), 318 MiB |
| autograd_loop | 71.38 ms (13.7x), 0 MiB | 69.62 ms (11.9x), 10 MiB | 80.50 ms (9.0x), 168 MiB |
| fwd_tangent_batch | 34.70 ms (6.6x), 2 MiB | 35.78 ms (6.1x), 522 MiB | 258.14 ms (28.7x), 8271 MiB |
| gn_analytic | 9.89 ms (1.0x), 0 MiB | 12.53 ms (1.0x), 10 MiB | 12.74 ms (1.0x), 157 MiB |
| gn_jacfwd_shared | 52.75 ms (5.3x), 1 MiB | 55.93 ms (4.5x), 270 MiB | 137.21 ms (10.8x), 4298 MiB |
| gn_vmap_jacfwd | 56.50 ms (5.7x), 1 MiB | 59.44 ms (4.7x), 271 MiB | 127.80 ms (10.0x), 4313 MiB |
| compiled_jacfwd_shared | 2.16 ms (0.4x), 0 MiB | 4.15 ms (0.7x), 89 MiB | 69.15 ms (7.7x), 1432 MiB |
| compiled_analytic | 1.10 ms (0.2x), 0 MiB | 4.82 ms (0.8x), 3 MiB | 5.72 ms (0.6x), 57 MiB |
