# Benchmark and profile

BetterRobot measures speed in two places. `benchmarks/run.py` records counters, which do not change from run to run
and gate every push. `benchmarks/speed.py` records wall time, which does change: every kinematics, dynamics and IK
function on Panda and G1, on CPU and CUDA, eager and compiled, from batch 1 to 65536, beside Pinocchio and other
libraries. This guide runs both, reads their output, and profiles one call with `torch.profiler`.

## Run the counter rows

From the repository root:

```bash
uv run python benchmarks/run.py --counters-only   # counter rows, checked against the thresholds: about 1 min
uv run python benchmarks/run.py                   # counter rows with their wall time
uv run python benchmarks/run.py --device cuda --counters-only
```

| Flag | Default | Meaning |
|------|---------|---------|
| `--device` | `cpu` | `cpu` or `cuda` |
| `--repeats` | `20` | most timed calls per row; fewer once a row has taken 2 s |
| `--threads` | `1` | Torch threads for the run |
| `--rows` | all | comma-separated row ids to measure |
| `--counters-only` | off | counter rows only, no wall time; exit 1 on a counter above its threshold |
| `--timing-only` | off | wall time only, no counters |
| `--out` | `benchmarks/results/<commit>_<device>.json` | output file; results are never committed |

Every number is `float32`, with the internal checks off: both scripts stop when `BETTER_ROBOT_INTERNAL_CHECKS=1`,
because the checks add host syncs and operators. Robots load from the tracked test fixtures, so both work offline.

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

Without `--counters-only` each counter row also gets its wall time (`median_ms`, `min_ms`) per call, after one
warm-up call, and `page_faults`, the minor page faults per call. The JSON's `meta` records the commit, dirty flag,
command, torch and Python versions, CPU or GPU, threads, CPU affinity and the load average at start and end.

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

## Measure wall time

`benchmarks/speed.py run` times every function below on Panda (fixed base) and G1 (free-flyer root) from
`tests/fixtures/robots/`, at each batch size, in each mode. On CPU, pin one idle core; on CUDA, check that no one
else uses the GPU, since every row records the other processes it finds there.

```bash
taskset -c 5 uv run python benchmarks/speed.py run --device cpu --out cpu.json
CUDA_VISIBLE_DEVICES=0 uv run python benchmarks/speed.py run --device cuda --out cuda.json
uv run python benchmarks/speed.py run --functions forward_kinematics,inverse_dynamics --robots panda --modes eager
```

| Flag | Default | Meaning |
|------|---------|---------|
| `--device` | `cpu` | `cpu` or `cuda` |
| `--dtype` | `float32` | or `float64` |
| `--functions` | all | comma-separated names from the table below |
| `--robots` | `panda,g1` | `panda`, `g1` |
| `--batches` | `1,64,1024,4096,16384,65536` | batch sizes `B` of `q (B, nq)` |
| `--modes` | `eager,compile`, and `reduce-overhead` on CUDA | `eager`; `compile` is `torch.compile(fn)`; `reduce-overhead` is `torch.compile(fn, mode="reduce-overhead")`, which replays CUDA graphs |
| `--repeats` | `15` | most samples per row |
| `--threads` | `1` | Torch threads for the run |
| `--max-call-s` | `5` | skip the larger batches of a function and mode once one call takes longer |
| `--max-compile-s` | `300` | the same, once one compile takes longer |
| `--out` | `benchmarks/results/speed_<commit>_<device>.json` | output file, rewritten after every row |

`state` is `forward_kinematics(model, q)`, and each call builds it, as a user's call does.

| Function | One call |
|----------|----------|
| `forward_kinematics` | `forward_kinematics(model, q).joint_transforms`: the state alone |
| `joint_poses` | `forward_kinematics(model, q).joint_poses` |
| `frame_jacobian` | `frame_jacobian(model, state, frame=...)` of `panda_hand` or G1's `left_rubber_hand` |
| `joint_jacobians` | `state.jacobian_columns`, the column table every joint's Jacobian masks |
| `inverse_dynamics` | `inverse_dynamics(model, state, v=v, a=a)` |
| `mass_matrix` | `mass_matrix(model, state)` |
| `forward_dynamics` | `forward_dynamics(model, state, v=v, tau=tau)` |
| `center_of_mass` | `center_of_mass(model, state)` |
| `ik_step` | one `LevenbergMarquardt.step` on `terms.ik_problem`, from `q_neutral` |
| `solve_ik` | `solve_ik(model, targets, q0=q_neutral)`: 100 iterations, one start |
| `solve_ik_multistart` | the same with `restarts=4`, five starts |

The IK targets are the poses of `panda_hand`, or of both hands and both ankles of G1, at seeded random
configurations. A solve row records `success_rate` (every target within 1 mm and 0.01 rad), `iterations` and the
median position and rotation errors.

Each row times one call under `torch.no_grad`, after warm-up calls, as the median over at most `--repeats` samples;
a sample is one or more calls, at least 10 ms long, between `torch.cuda.synchronize` calls. A compiled row starts
from `torch.compiler.reset()` and records its first call as `compile_seconds` (shorter when inductor's on-disk cache
already holds the graph), then `graph_breaks`, `graphs`, `recompiles` after the warm-up, `cudagraph_skips` and
`max_abs_diff_vs_eager`. A row that raises records its `error`; the larger batches of that function and mode are then
recorded as `skipped`, with the reason. `MODES` in the script is the one place that turns a mode into a callable.

On CPU the run adds Pinocchio rows (`float64`, batch 1, one call per sample) for every function Pinocchio has. The
run re-executes itself with the allocator thresholds of the last section fixed, so a large batch's time does not
depend on what the process freed before. `meta` records the commit, the tree of `src/` (the library measured,
whatever the benchmark's own commit), torch and CUDA versions, host, device name, threads and CPU affinity, the
allocator variables, the load average and, on CUDA, the GPU's utilization, memory and other processes at start and
end.

## Compare runs and libraries

Every row of a run, and every row of another library's timings, uses one schema, so a JSON written elsewhere
reads as a reference column:

| Field | Meaning |
|-------|---------|
| `library` | `better_robot`, `pinocchio`, or the other library's name |
| `function` | a name from the function table |
| `robot` | `panda` or `g1` |
| `device` | `cpu` or `cuda` |
| `dtype` | `float32` or `float64` |
| `batch` | samples per call |
| `mode` | `eager`, `compile`, `reduce-overhead`; another library names its own |
| `seconds_per_call` | median wall time of one call; absent when the row has `error` or `skipped` |
| `compile_seconds` | the first call of a compiled row, when there is one |

A run's rows add `seconds_min`, `seconds_spread` (interquartile range over the median), `samples`, `page_faults`
per call, `loadavg` and, on CUDA, `gpu_other_pids`: a row with any was timed on a shared GPU.

```bash
uv run python benchmarks/speed.py table after.json --before before.json
uv run python benchmarks/speed.py table cuda.json --reference cpu.json other_cuda.json --batches 1,4096
```

`table` prints one markdown line per row: microseconds per sample and compile seconds; with `--before`, the same
row of the older run and the speedup `before/after`; with `--reference`, for each other library its microseconds
per sample and the ratio `ours/<library>`, the times BetterRobot is slower. A reference row matches on function,
robot, device and batch; a Pinocchio row, one call per sample on the CPU, matches every device and batch at its
batch-1 time. `--functions`, `--robots`, `--batches` and `--modes` select rows.

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

Measured at the commit `22aed80`, one thread, forward kinematics returning joint poses at batch 4096: FK on G1 takes
57.7 ms with 10 301 page faults per call by default and 33.5 ms with fewer than one once the thresholds are fixed;
CRBA on Panda takes 42.7 ms with 5757 faults against 29.2 ms; FK on a 200-joint random tree 418 ms against 179 ms.
The default number depends on the process, the fixed one does not. The counter row `fk_joints/g1/B1024` took 12.2
ms with 3120 faults per call by default in one process and 6.0 ms in another; with fixed thresholds it takes 6.0 ms
in both. `benchmarks/speed.py` sets both thresholds for its whole run, and its rows keep `page_faults` per call.

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
