# Benchmark and profile

`benchmarks/run.py` measures wall time and PyTorch operator counts for forward kinematics, the frame
Jacobian, RNEA, CRBA and ABA, across robots and batch sizes. This guide shows how to run it, how to
read its output, and how to profile one call yourself with `torch.profiler`.

## Run the benchmark

From the repository root:

```bash
CUDA_VISIBLE_DEVICES="" uv run python benchmarks/run.py --device cpu
uv run python benchmarks/run.py --device cuda --batches 1,4096 --models panda,tree200 --algos fk,rnea
```

| Flag | Default | Meaning |
|------|---------|---------|
| `--device` | `cpu` | `cpu` or `cuda` |
| `--batches` | `1,64,256,4096` | leading batch sizes `B` of `q (B, nq)` |
| `--models` | `panda,g1,tree25,tree200` | `panda`, `g1` (free flyer) from `robot_descriptions`, or `tree<N>`: a seeded random tree of `N` revolute joints |
| `--algos` | `fk,jacobian,rnea,crba,aba` | algorithms to time |
| `--repeats` | `20` | maximum timed repeats per row; fewer when a row takes more than about 2 s |
| `--threads` | `1` | `torch.set_num_threads` for the run |
| `--out` | `benchmarks/results/<git short sha>_<device>.json` | output file |

Models are float32 on the chosen device. Inputs are seeded: `q (B, nq)` is tiled from 16 random
configurations, and `v`, `a`, `tau (B, nv)` are fixed.

## What each row measures

Each row is one (algorithm, model, batch, mode) combination:

- `mode="forward"` times one call under `torch.no_grad()`; `mode="backward"` times forward plus the
  backward pass of `sum(output)` with respect to `q`.
- `median_ms` and `min_ms` are per call. CUDA rows synchronize before reading the clock.
- `ops` counts the top-level `aten::` operator calls in one call. It does not depend on the batch
  size, so it tracks Python and dispatch overhead, which dominates at small batches.
- When `pinocchio` imports, reference rows (`impl="pinocchio"`) are added for Panda and G1: float64,
  CPU, batch 1, forward only, single thread.

The JSON file holds `meta` (commit, dirty flag, torch version, device name, threads) and `rows`.
Results are only comparable on the same machine and thread count.

## Recorded results

Panda, CPU, one thread, AMD EPYC 9654, torch 2.13, forward mode. Numbers come from the JSON files in
`benchmarks/results/`; each file's `meta.commit` names the code it measured.

| Algorithm | File | `B = 1` (ms) | `B = 4096` (ms) | `ops` |
|-----------|------|-------------:|----------------:|------:|
| FK | `0953f28_cpu.json` | 0.655 | 32.9 | 246 |
| FK | `p2_lean_fk_cpu.json` | 0.551 | 20.3 | 188 |
| RNEA | `0953f28_cpu.json` | 2.11 | 53.4 | 1053 |
| RNEA | `p2_world_dynamics_cpu.json` | 1.07 | 52.0 | 379 |
| FK, Pinocchio | `0953f28_cpu.json` | 0.0005 | — | — |
| RNEA, Pinocchio | `0953f28_cpu.json` | 0.0016 | — | — |

At batch 1, Pinocchio's C++ is about three orders of magnitude faster: the cost here is operator
dispatch, which is why `ops` is reported. The library pays off at large batches and on GPU, where one
call serves thousands of configurations.

## Profile one call

`torch.profiler` shows where time goes inside one function. Warm up first so one-time setup is not
counted.

```python
import torch
import better_robot as br
from better_robot import kinematics
from robot_descriptions import panda_description

model = br.load(panda_description.URDF_PATH)
q = model.q_neutral.expand(256, -1).clone()
for _ in range(3):
    kinematics.forward_kinematics_state(model, q)

with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as prof:
    with torch.no_grad():
        kinematics.forward_kinematics_state(model, q)

print(prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=8))
```
```output
------------------------  ------------  ------------  ------------  ------------  ------------  ------------  
                    Name    Self CPU %      Self CPU   CPU total %     CPU total  CPU time avg    # of Calls  
------------------------  ------------  ------------  ------------  ------------  ------------  ------------  
               aten::bmm        33.01%     357.755us        33.13%     359.085us      23.939us            15  
                aten::mm         8.62%      93.431us         8.69%      94.152us      94.152us             1  
      aten::index_select         8.00%      86.740us         8.47%      91.827us      22.957us             4  
...
```

On GPU, add `ProfilerActivity.CUDA` to `activities` and call `torch.cuda.synchronize()` before
leaving the block. `prof.export_chrome_trace("trace.json")` writes a timeline you can open in
`chrome://tracing` or [Perfetto](https://ui.perfetto.dev).

## Benchmark tests

`tests/bench/` holds micro-benchmarks marked `bench`. They are advisory, not part of the contract, and
the default test command skips them:

```bash
uv run pytest tests/bench -m bench -q
```

## Where next

- [Testing](../contributing/testing.md): the `bench` and `cuda` markers.
- [Performance contract](../contracts/09-performance.md): the rules the benchmark guards.
