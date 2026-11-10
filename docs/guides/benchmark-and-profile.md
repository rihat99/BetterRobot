# Benchmark and profile

`benchmarks/run.py` is the one place BetterRobot measures speed. It records two kinds of numbers: counters, which
do not change from run to run and gate every push, and wall time, which does change and is recorded with the
machine's load beside it. This guide runs it, reads its output, and profiles one call with `torch.profiler`.

## Run the benchmark

From the repository root:

```bash
uv run python benchmarks/run.py --counters-only   # counter rows, checked against the thresholds: about 1 min
uv run python benchmarks/run.py                   # counter rows and the sweep, with wall time: about 7 min
uv run python benchmarks/run.py --device cuda --batches 1,4096 --models panda,tree200 --algos fk,rnea
```

| Flag | Default | Meaning |
|------|---------|---------|
| `--device` | `cpu` | `cpu` or `cuda` |
| `--batches` | `1,64,256,4096` | sweep batch sizes `B` of `q (B, nq)` |
| `--models` | `panda,g1,tree25,tree200` | sweep models: `panda`, `g1` (free flyer) from `tests/fixtures/robots/`, or `tree<N>`, a seeded random tree of `N` revolute joints |
| `--algos` | `fk,jacobian,rnea,crba,aba` | sweep algorithms |
| `--repeats` | `20` | most timed calls per row; fewer once a row has taken 2 s |
| `--threads` | `1` | Torch threads for the run |
| `--rows` | all | comma-separated row ids to measure |
| `--counters-only` | off | counter rows only, no wall time; exit 1 on a counter above its threshold |
| `--timing-only` | off | wall time only, no counters |
| `--max-call-s` | `20` | skip the larger batches of a sweep row once one call takes longer |
| `--out` | `benchmarks/results/<commit>_<device>.json` | output file; results are never committed |

Every number is `float32`, with the internal checks off: the runner stops when `BETTER_ROBOT_INTERNAL_CHECKS=1`,
because the checks add host syncs and operators. Robots load from the tracked test fixtures, so the runner works
offline.

## What each row measures

Counter rows are fixed functions; their counters gate (below). `state` is `forward_kinematics(model, q)`.

| Row | Function |
|-----|----------|
| `fk_joints/{panda,g1}/B{1,1024}` | `forward_kinematics(model, q).joint_transforms` |
| `jacobian/{panda,g1}/B1` | `frame_jacobian(model, state, frame=model.nframes - 1)` |
| `rnea/{panda,g1}/B1` | `inverse_dynamics(model, state, v=v, a=a)` |
| `crba/{panda,g1}/B1` | `mass_matrix(model, state)` |
| `aba/{panda,g1}/B1` | `forward_dynamics(model, state, v=v, tau=tau)` |
| `ik/panda/B64` | `solve_ik` to 64 reachable `panda_hand` poses, 4 restarts; also `success_fraction` and `worst_position_error_m` |

Each counter row has four counters, the same for every run with one torch version and device:

- `graph_breaks`: graph breaks `torch._dynamo.explain` finds at the row's batch.
- `recompiles`: graphs compiled when `torch.compile(fn, backend="eager")`, first called at the row's batch, is then
  called at batches 1, 2, 3 and 64.
- `ops`: top-level `aten::` operators of one call. It does not depend on the batch, so it tracks the Python and
  dispatch overhead that dominates small batches.
- `host_syncs`: calls of `aten::_local_scalar_dense` (`item`, `bool`, `float` or `int` of a tensor) and
  `aten::nonzero` (also inside boolean-mask indexing). On CUDA each one makes the host wait for the GPU queue; they
  are counted on CPU too, so the CPU gate catches a new one. The IK row has neither `ops` nor `host_syncs`: profiling
  one solve takes a minute.

Sweep rows, ids `<algo>/<model>/B<batch>/<mode>/<impl>`, time each algorithm over `--models` and `--batches`:
`mode="forward"` is one call under `torch.no_grad()`, `mode="forward_backward"` adds the backward pass of
`sum(output)` with respect to `q`. They record `ops` and `host_syncs` of one call too. When `pinocchio` imports,
reference rows (`impl="pinocchio"`) are added for Panda and G1: `float64`, CPU, batch 1, forward only.

Wall time (`median_ms`, `min_ms`) is per call, after one warm-up call; CUDA rows synchronize before reading the
clock. `page_faults` is the minor page faults per call. Every row records the load average when it started, and the
JSON's `meta` records the commit, dirty flag, command, torch and Python versions, CPU or GPU, threads, CPU affinity
and the load average at start and end. Wall time compares only on the same machine, thread count and similar load.

## The counter gate

The thresholds live in `benchmarks/thresholds/torch-<version>.json`, one file per torch version: the runner's own
`--counters-only` output, written on a clean commit. `tests/contract/test_compile_counters.py` runs
`--counters-only` in the full tier and fails when a counter is above its threshold, when a counter row has no
threshold, or when the installed torch has no thresholds file. Read the file like any result:

```python
import json
from pathlib import Path

import torch

path = Path("benchmarks/thresholds") / f"torch-{torch.__version__.split('+')[0]}.json"
thresholds = json.loads(path.read_text())
counters = ("graph_breaks", "recompiles", "ops", "host_syncs")
print(thresholds["meta"]["commit"], thresholds["meta"]["command"])
print(f"{'row':22}", *(f"{name:>12}" for name in counters))
for row in thresholds["rows"]:
    print(f"{row['row']:22}", *(f"{'-' if row[name] is None else row[name]:>12}" for name in counters))
```
```output
1ed9049 benchmarks/run.py --counters-only --out benchmarks/thresholds/torch-2.13.0.json
row                    graph_breaks   recompiles          ops   host_syncs
fk_joints/panda/B1                0            1           46            0
fk_joints/panda/B1024             0            2           46            0
jacobian/panda/B1                 0            1           69            0
rnea/panda/B1                     0            1          172            0
crba/panda/B1                     0            1          136            0
aba/panda/B1                      0            1          240            2
fk_joints/g1/B1                   0            1           67            0
fk_joints/g1/B1024                0            2           67            0
jacobian/g1/B1                    0            1           89            0
rnea/g1/B1                        0            1          188            0
crba/g1/B1                        0            1          154            0
aba/g1/B1                         0            1          255            2
ik/panda/B64                      5           12            -            -
```

When a change adds a counter, remove what added it. When the change needs it, or when a change lowers a counter,
run `uv run python benchmarks/run.py --counters-only --out benchmarks/thresholds/torch-<version>.json` on a clean
commit and commit that file alone, with a `Why:` line; a lowered counter is written down the same way, so the gate
stays tight. The two host syncs of `aba` are `torch.linalg.cholesky` and `torch.cholesky_solve`, which check their
result.

## Recorded wall time

Measured with `uv run python benchmarks/run.py` (the defaults) at the clean commit `22aed80`: AMD EPYC 9654, one
thread pinned to an idle core, torch 2.13, a shared host with load average 16 to 38 during the run (397 s). Forward
mode, ms per call; batch-4096 rows give the default-allocator time with its page faults per call, then the time with
fixed allocator thresholds (next section).

| Algorithm | Model | `B = 1` | Pinocchio `B = 1` | `B = 4096` default (faults) | `B = 4096` fixed | `ops` |
|-----------|-------|--------:|------------------:|----------------------------:|-----------------:|------:|
| FK | Panda | 0.424 | 0.0005 | 13.7 (1984) | 9.60 | 140 |
| Jacobian | Panda | 0.267 | 0.0007 | 9.72 (0) | 9.63 | 69 |
| RNEA | Panda | 0.544 | 0.0017 | 29.1 (0) | 28.8 | 172 |
| CRBA | Panda | 0.494 | 0.0012 | 42.7 (5757) | 29.2 | 136 |
| ABA | Panda | 0.786 | 0.0057 | 62.4 (5334) | 50.2 | 240 |
| FK | G1 | 0.576 | 0.0013 | 57.7 (10301) | 33.5 | 161 |
| Jacobian | G1 | 0.427 | 0.0023 | 53.7 (7184) | 37.0 | 89 |
| RNEA | G1 | 0.714 | 0.0054 | 101 (1) | 101 | 188 |
| CRBA | G1 | 0.648 | 0.0046 | 139 (14728) | 104 | 154 |
| ABA | G1 | 0.969 | 0.0187 | 241 (21418) | 187 | 255 |

At batch 1, Pinocchio's C++ is two to three orders of magnitude faster: the cost here is operator dispatch, which is
why `ops` is reported. The library pays off at large batches and on GPU, where one call serves thousands of
configurations. The sweep's `fk` returns joint poses, so its `ops` differ from those of the counter row `fk_joints`,
which returns joint transforms.

## Large batches and the allocator

At batch 1024 and above, a default-allocator time can double from page faults alone. glibc raises its mmap
threshold to the largest block the process has freed, and its trim threshold to twice that; when a call's
temporaries add up to more than the trim threshold, glibc returns the memory to the system at the end of the call
and the next call faults it back in, about 2 us per 4 KiB page. Whether that happens depends on what the process
allocated and freed before, not on the call. Set both thresholds before the process starts and it stops: an explicit
value turns glibc's adjustment off, and 128 MiB and 256 MiB keep a large batch's temporaries on the heap.

```bash
MALLOC_MMAP_THRESHOLD_=134217728 MALLOC_TRIM_THRESHOLD_=268435456 uv run python your_script.py
```

Measured in the run above, forward mode at batch 4096: FK on G1 takes 57.7 ms with 10 301 page faults per call by
default and 33.5 ms with fewer than one once the thresholds are fixed; CRBA on Panda takes 42.7 ms with 5757 faults
against 29.2 ms; FK on the 200-joint random tree 418 ms against 179 ms. The default number depends on the process,
the fixed one does not. The counter row `fk_joints/g1/B1024` took 12.2 ms with 3120 faults per call by
default in one process and 6.0 ms in another; with fixed thresholds it takes 6.0 ms in both.

The runner measures every row at batch 1024 and above both ways: it re-runs those rows in a child process with the
two variables set and records `fixed_malloc_median_ms` and `fixed_malloc_page_faults` beside the default numbers. A
"no slower" verdict at those batches reads the fixed-threshold number and its page faults.

## Profile one call

`torch.profiler` shows where time goes inside one function. Warm up first so one-time setup is not
counted.

```python
import torch
import better_robot as br
from robot_descriptions import panda_description

model = br.load(panda_description.URDF_PATH)
q = model.q_neutral.expand(256, -1).clone()
for _ in range(3):
    br.forward_kinematics(model, q)

with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as prof:
    with torch.no_grad():
        br.forward_kinematics(model, q)

print(prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=8))
```
```output
------------------------  ------------  ------------  ------------  ------------  ------------  ------------  
                    Name    Self CPU %      Self CPU   CPU total %     CPU total  CPU time avg    # of Calls  
------------------------  ------------  ------------  ------------  ------------  ------------  ------------  
...
```

On GPU, add `ProfilerActivity.CUDA` to `activities` and call `torch.cuda.synchronize()` before
leaving the block. `prof.export_chrome_trace("trace.json")` writes a timeline you can open in
`chrome://tracing` or [Perfetto](https://ui.perfetto.dev).

## Where next

- [Run the tests](run-the-tests.md): the tiers that run the counter gate.
- [Performance contract](../contracts/09-performance.md): the rules the benchmark guards.
