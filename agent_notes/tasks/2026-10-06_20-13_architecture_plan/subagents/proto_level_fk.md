# Level-batched FK prototype: decision experiment (2026-10-06)

Script: `subagents/proto_level_fk.py` (217 lines total; prototype class `LevelFK` = lines 57-118, ~60 lines, of which the two chain variants are 15 lines). Raw: `artifacts/proto_fk_{cpu,cuda,check}.jsonl`, `artifacts/proto_fk_*.log`. Tables: `subagents/tabulate.py`.

## Environment
- BetterRobot dev worktree, commit 7e2008f, its own `.venv` (uv), torch 2.13.0+cu129, float32 for timings.
- CPU: AMD EPYC 9654, 384 threads, shared machine (CPU numbers noisy, see panda lib_eager b4096 fwd: 28 ms in a first run, 178 ms in the second).
- GPU: RTX PRO 6000 Blackwell, index 6. `gpu_free.py --share-util 50` found NO usable GPU (all 62-85 % avg). Used GPU 6 (62 % avg, 0 % at start, 98 % right after the run because of another user's job). **Shared GPU, indicative only.** Even compiled + CUDA-graph lanes show a ~2.3 ms floor, so absolute GPU latency is inflated by contention; kernel counts are the robust metric.
- Warp: NOT installed in the dev env (`import warp` fails); main-checkout env has warp 1.15 but was not tried (budget). **No Warp number was measured.** Stored Warp baseline (RTX 6000 Ada, nj 25): 0.54-0.63 ms fwd.

## What the library already batches (`_fk_matrix.py`)
Local joint transforms are batched by kind (`_local_matrices`, 172-229) and parent placement is composed in one matmul (258). Only the world chain is per-joint: `for j in topo_order: world[j] = world[parent] @ local[j]` (261-266), ~1 matmul + 1 select per joint. The prototype reuses `prepare_inputs` + `_local_matrices` unchanged and replaces only that loop:
- **level**: permute joints by depth once; per depth level one `index_select` (parents in previous level) + one batched matmul; final `cat` + inverse permute.
- **jump** (pointer jumping): sentinel identity at index nj; `ceil(log2(depth+1))` steps of `acc = acc[ptr] @ acc`.

## Models
| model | njoints | nq | depth | jump steps |
|---|---|---|---|---|
| panda (URDF from robot_descriptions cache) | 14 | 8 | 11 | 4 |
| humanoid24 (synthetic SMPL-24 tree, free flyer + 23 spherical) | 25 | 99 | 9 | 4 |
| humanoid222 (humanoid24 + 2x5x3 spherical fingers + 24x7 revolute leaf chains) | 223 | 387 | 16 | 5 |
Real SMPL-X/MHR not used: model files missing in the dev BetterHuman worktree (MHR npz exists only in the main checkout; not wired up for budget).

## Correctness (CPU, 64 random configs; vs `forward_kinematics_raw(...).joint_pose_world`)
level: max abs err 0.0 (bit-identical) for all models, f32 and f64. jump: f32 <= 9.5e-7, f64 <= 1.8e-15. All pass (<1e-5 f32, <1e-12 f64). Gradients were not compared to the library (only timed).

## GPU (shared, indicative), median ms of 30 after 5 warm-up. E = eager, C = torch.compile dynamic=True, RO = reduce-overhead
| model | pass | B | lib E | level E | jump E | lib C | level C | level RO |
|---|---|---|---|---|---|---|---|---|
| panda | fwd | 1 / 256 / 4096 | 1.93 / 1.60 / 2.69 | 1.71 / 2.05 / 2.73 | 1.48 / 1.77 / 2.91 | 0.41 / 2.35 / 2.45 | 0.41 / 2.32 / 2.45 | 0.22 / 2.33 / 2.48 |
| panda | fwd+bwd | 1 / 256 / 4096 | 6.25 / 8.45 / 10.7 | 6.59 / 9.34 / 9.10 | 5.18 / 7.29 / 9.56 | 1.58 / 2.47 / 3.11 | 2.51 / 2.49 / 3.14 | 0.76 / 2.47 / 3.14 |
| humanoid24 | fwd | 1 / 256 / 4096 | 2.56 / 2.61 / 2.92 | 2.56 / 2.63 / 2.90 | 2.61 / 2.62 / 3.20 | 2.34 / 2.37 / 2.59 | 2.30 / 2.34 / 2.60 | 2.31 / 2.36 / 2.61 |
| humanoid24 | fwd+bwd | 1 / 256 / 4096 | 14.8 / 13.1 / 11.8 | 10.1 / 10.2 / 9.35 | 9.87 / 10.2 / 11.5 | 4.82 / 4.81 / 3.76 | 4.64 / 4.74 / 3.76 | 2.42 / 2.52 / 3.74 |
| humanoid222 | fwd | 1 / 256 / 4096 | 7.47 / 7.79 / 11.7 | 4.91 / 5.18 / 8.66 | 2.62 / 3.08 / 21.2 | 7.55 / 7.75 / 11.7 | 4.91 / 5.18 / 8.69 | 4.91 / 5.18 / 8.66 |
| humanoid222 | fwd+bwd | 1 / 256 / 4096 | 52.2 / 51.3 / 71.8 | 13.1 / 13.7 / 38.4 | 10.2 / 12.6 / 91.1 | 51.5 / 51.9 / 72.0 | 13.2 / 14.5 / 38.8 | 14.9 / 15.6 / 38.4 |

CUDA kernels per call, eager, B=256 (torch.profiler):
| model | lib fwd / fwd+bwd | level | jump |
|---|---|---|---|
| panda | 188 / 621 | 198 / 639 | 183 / 552 |
| humanoid24 | 219 / 796 | 221 / 724 | 203 / 650 |
| humanoid222 | 452 / 2297 | 274 / 913 | 237 / 769 |

## CPU (shared), selected, median ms (full table via tabulate.py)
humanoid222 fwd+bwd: lib E 10.4 / 116 / 2550 (B 1/256/4096), level E 3.24 / 87.7 / 640, level C 3.3 / 79.9 / 617. humanoid24 and panda: level within noise of lib.

## Verdict
1. Level-batching removes the chain cost but the chain is NOT the dominant launch count for small trees: panda and humanoid24 spend ~180-200 forward kernels outside the chain (prepare_inputs, per-kind local transforms, 4x4 assembly, matrix->quaternion). There, level = lib within noise (fwd); backward improves modestly (humanoid24 GPU 14.8 -> 10.1 ms, 796 -> 724 kernels).
2. For a 223-joint tree it matters a lot: fwd kernels 452 -> 274, fwd+bwd kernels 2297 -> 913; GPU fwd 7.5 -> 4.9 ms, fwd+bwd 52 -> 13 ms (4x); CPU fwd+bwd B4096 2.55 s -> 0.64 s. Exact (bit-identical) results, full autograd, compiles fine.
3. Pointer jumping: fewest kernels and best at small batch (222 fwd B1 2.6 ms), but O(nj log depth) work makes it 2.4x slower than level at B4096 GPU and worse on backward. Level is the better default.
4. Does it close the gap to Warp? Not shown. Warp was not runnable here, and the ~2.3 ms GPU floor under contention (even compiled + CUDA graphs) hides sub-ms differences. What the data does show: after level-batching, remaining forward cost is ~200 constant kernels of local-transform/prep/quaternion work, so the next lever is fusing that constant part (compile already does not reduce it here) or CUDA-graphing whole solver steps, not the chain. torch.compile gave no forward gain for humanoid222 (C = E), so compile is not fusing the remaining ops either.
5. Backward is where level-batching pays most (autograd replays per-joint matmul backward nodes in the lib; level collapses them per depth). Warp has no backward at all, so on fwd+bwd the level lane is strictly the better candidate.

## Not verified
Warp timings; gradients vs library (only forward values compared); real SMPL-X/MHR trees; idle-GPU absolute latency; dynamic=False compile.
Side findings: `Model.random_configuration` on a float64 humanoid model returned float32 q (free-flyer/spherical), and FK on float64 model with float32 q crashes in `_local_matrices` `index_copy` (dtype mismatch) instead of a validation error; random_configuration yields inf for revolutes without limits.
