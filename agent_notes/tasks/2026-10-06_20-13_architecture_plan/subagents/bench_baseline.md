# Performance baseline (2026-10-06 20:19-20:33 +04), CPU only

Full tables: `../artifacts/tables_cpu.md`. Raw: `../artifacts/status_cpu_t1.json`, `status_cpu_t16.json`, `baseline_cpu_eager.json` (+ `.log`, `baseline_cpu_eager_rows.txt`).

## Environment
- BetterRobot 7e2008f73850a01a43a5ca386af0f54f3099fe23, BetterHuman 793d71607c16753249314c405874db1c17ffc91c
- torch 2.13.0+cu129, float32, eager. AMD EPYC 9654 (384 logical CPUs, load avg ~10 shared). torch threads 1 (main) and 16 (B=4096 only).
- GPU: none used. gpu_free.py (`--need-gb 8 --share-util 50`) -> "no usable GPUs": all 8 RTX PRO 6000 Blackwell busy (GPU0/3/6/7 94-96% util; GPU1/2/4/5 67% avg, 100% peak, 88.6 of 97.9 GB used). CUDA_VISIBLE_DEVICES="".
- warp: not installed. pinocchio: not importable in the BetterRobot env. robot_descriptions: not installed (URDFs read from ~/.cache/robot_descriptions directly).
- Models: Panda = example-robot-data panda.urdf (nq=nv=8, includes fingers); G1 = unitree_ros g1_29dof.urdf + free flyer (nq=36, nv=35).
- Timer: perf_counter_ns, warmup 10, median of up to 50 samples within a 4 s budget, torch.no_grad.
- Commands (from BetterRobot root, UV_CACHE_DIR=/tmp/betterrobot-uv-cache CUDA_VISIBLE_DEVICES=""):
  `uv run python agent_notes/.../subagents/bench_status.py --pinocchio --output .../artifacts/status_cpu_t1.json`
  `uv run python agent_notes/.../subagents/bench_status.py --threads 16 --batches 4096 --ik-batches 1024 --ik-samples 3 --output .../artifacts/status_cpu_t16.json`
  `uv run python benchmarks/baseline.py --label status-cpu-eager --devices cpu --lanes eager --case-timeout-seconds 180 --output .../artifacts/baseline_cpu_eager.json`

## Median ms (evals/s), CPU eager, 1 thread unless noted
| algo | Panda B=1 | Panda B=64 | Panda B=4096 | Panda 4096 @16t | G1 B=1 | G1 B=64 | G1 B=4096 | G1 4096 @16t |
|---|---|---|---|---|---|---|---|---|
| FK | 0.654 (1.5k) | 1.09 (58k) | 41.9 (98k) | 21.7 (189k) | 0.891 (1.1k) | 2.11 (30k) | 147 (28k) | 49.5 (83k) |
| FK+frames | 0.743 | 1.25 | 46.1 | 30.3 | 0.992 | 2.36 | 158 | 59.9 |
| frame Jacobian (FK+frames+joint J+get) | 1.08 (922) | 1.83 (35k) | 58.7 (70k) | 42.9 (95k) | 1.34 (749) | 3.82 (17k) | 351 (12k) | 109 (38k) |
| RNEA | 2.14 (467) | 3.09 (21k) | 70.3 (58k) | 46.3 (88k) | 4.93 (203) | 7.57 (8.5k) | 259 (16k) | 121 (34k) |
| CRBA | 1.87 (536) | 3.23 (20k) | 107 (39k) | 43.0 (95k) | 4.63 (216) | 8.63 (7.4k) | 364 (11k) | 113 (36k) |
| ABA | 6.21 (161) | 9.42 (6.8k) | 252 (16k) | 131 (31k) | 7.21 (139) | 12.2 (5.3k) | 425 (9.6k) | 161 (25k) |

solve_ik, Panda, default config (LM, max_iter 100, tol 1e-6), targets = FK of q_neutral + 0.3*randn:
| B | threads | median ms | solves/s | iters median / max | converged |
|---|---|---|---|---|---|
| 1 | 1 | 61.9 | 16.2 | 7 / 7 | 1.000 |
| 1024 | 1 | 4182 | 245 | 20 / 100 | 0.675 |
| 1024 | 16 | 5264 | 195 | 20 / 100 | 0.675 |

benchmarks/baseline.py (canonical harness, CPU eager, 1 thread): all 24 Panda cases UNSUPPORTED (robot_descriptions missing). SMPL-like 25-joint tree: FK 0.667 ms (B=1) / 76.1 ms (B=4096); RNEA 2.63 / 133.5 ms; IK (5 fixed iters) 71.1 ms at B=1, TIMEOUT (>180 s whole case) at B=4096. Total 6 min 48 s.

## Skipped / failed
- All GPU rows (no qualifying GPU). B=65536 (GPU-only in brief).
- Warp lane (not installed); torch.compile lanes (ran out of wall-time budget; baseline.py supports them via `--lanes compiled`).
- Pinocchio reference (not importable in the BetterRobot env).
- BetterHuman benchmark_bodies.py: CUDA-only (hard-codes device "cuda", line 722) and dev worktree has no `models/` dir (SMPL/SMPLX/MHR npz missing; MHR npz exist only in /home/rikhat.akizhanov/better/BetterHuman/models). Not run.

## Observations
- Launch/overhead-bound at small batch: Panda FK 0.65 ms at B=1 vs 1.09 ms at B=64 (64x work, 1.7x time). B=1 cost is ~0.65 ms FK, ~2 ms RNEA, ~6 ms ABA for a 7-dof arm; pinocchio-class C++ is typically ~1 us (not measured here).
- ABA is ~3x RNEA at B=1 on Panda (6.2 vs 2.1 ms).
- G1 frame Jacobian at B=4096 (351 ms) is slower than RNEA (259 ms).
- 16 threads give only 1.5-3.6x at B=4096.
- solve_ik B=1024: 32.5% of elements hit max_iter=100 without meeting tol 1e-6 in float32 on reachable targets; batch wall time is set by the slowest element (100 iters), and 16 threads made it slower (5.26 s vs 4.18 s).
