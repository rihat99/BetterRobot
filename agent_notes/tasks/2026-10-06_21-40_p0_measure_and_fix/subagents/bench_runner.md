# bench_runner: benchmarks/run.py (in progress)

Script: /home/rikhat.akizhanov/better/dev/BetterRobot/benchmarks/run.py (new file, nothing else modified).

- jacobian = `get_frame_jacobian(model, forward_kinematics(model, q), frame_id=nframes-1)` (local_world_aligned).
  Pinocchio jacobian row = `pin.computeJointJacobians`.
- ops = top-level `aten::*` profiler CPU events (no aten ancestor), one call after 1 warm-up call.
- tree<N> = seeded (seed 0) fixed-base uniform random recursive tree of N revolute joints, mass 1, inertia 0.01 I.
- Smoke run (batches 1,64, repeats 5): all algos ran on all 4 models, no errors. ops differ by 1 between
  batch 1 and 64 for crba/aba on tree200 (14183 vs 14184, 17366 vs 17367): batch-dependent, not run-to-run.
- Heavy row probe: crba tree200 batch 4096 forward_backward took more than 10 min for warm-up+profile+timed.

## Status at hand-back (agent ran out of tokens)
- Full run: `CUDA_VISIBLE_DEVICES="" uv run python benchmarks/run.py --device cpu` -> benchmarks/results/0953f28_cpu.json (3079 s, no skipped/error rows in log).
- Second run (`--out benchmarks/results/run2_cpu.json`) and the ops comparison are still running in the background; when it finishes,
  the comparison will be in scratchpad summary.txt (written by scratchpad summarize.py run1 run2).
- Added `--max-call-s` (default 20): skips larger batches of an (algo, model, mode) after one call exceeds it, and records that in the row's error field.
- crba tree200 forward_backward: 0.86 s at B=64, 26.3 s at B=256 (superlinear). Profiler prints "gpuGetDeviceCount failed" to stderr under CUDA_VISIBLE_DEVICES="" (harmless).

## Correction after hand-back
- The rerun chain (run 1 with --max-call-s, then run2_cpu.json, then summarize.py) exited with code 144 (killed). The hand-back said
  it was still running; that was wrong. run2_cpu.json and summary.txt probably do not exist. The ops-reproducibility check was NOT done.
- To finish: rerun `CUDA_VISIBLE_DEVICES="" uv run python benchmarks/run.py --device cpu --out benchmarks/results/run2_cpu.json`,
  then `uv run python <scratchpad>/summarize.py benchmarks/results/0953f28_cpu.json benchmarks/results/run2_cpu.json`.
