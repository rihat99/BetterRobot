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
| `ik/panda/B64` | `solve_ik` to 64 reachable `panda_hand` poses, 5 seeds; also `success_fraction` and `worst_position_error_m` |
| `ik_constrained/panda/B64` | 20 steps of `SequentialQP` to the same poses, clear of a sphere and of the arm's own sphere pairs; also `success_fraction` and `worst_violation_m` |
| `track/panda/B64` | 20 warm calls of `optim.track` with the same rows, steps of at most 5 mrad; also `success_fraction` and `worst_violation_m` |
| `qp/n7m30/B1`, `qp/n35m150/B1024` | one cold `optim.solve_qp` of seeded random programs with 7 or 35 variables and 30 or 150 rows |

Each counter row has four counters, the same for every run with one torch version and device:

- `graph_breaks`: graph breaks `torch._dynamo.explain` finds at the row's batch.
- `recompiles`: graphs compiled when `torch.compile(fn, backend="eager")`, first called at the row's batch, is then
  called at batches 1, 2, 3 and 64.
- `ops`: top-level `aten::` operators of one call. It does not depend on the batch, so it tracks the Python and
  dispatch overhead that dominates small batches.
- `host_syncs`: calls of `aten::_local_scalar_dense` (`item`, `bool`, `float` or `int` of a tensor) and
  `aten::nonzero` (also inside boolean-mask indexing). On CUDA each one makes the host wait for the GPU queue; they
  are counted on CPU too, so the CPU gate catches a new one. The IK row's solve runs a fixed number of steps with no
  host sync, so its threshold is 0.

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
d1d59ac benchmarks/run.py --counters-only --out benchmarks/thresholds/torch-2.13.0.json
row                    graph_breaks   recompiles          ops   host_syncs
fk_joints/panda/B1                0            1           40            0
fk_joints/panda/B1024             0            2           40            0
jacobian/panda/B1                 0            1           65            0
rnea/panda/B1                     0            1          165            0
crba/panda/B1                     0            1          130            0
aba/panda/B1                      0            1          238            0
fk_joints/g1/B1                   0            1           60            0
fk_joints/g1/B1024                0            2           60            0
jacobian/g1/B1                    0            1           84            0
rnea/g1/B1                        0            1          180            0
crba/g1/B1                        0            1          147            0
aba/g1/B1                         0            1          252            0
ik/panda/B64                      5           11        30373            0
ik_constrained/panda/B64            3           11        31372            0
track/panda/B64                   0            3        29324            0
qp/n7m30/B1                       0            1          560            0
qp/n35m150/B1024                  0            2          560            0
```

When a change adds a counter, remove what added it. When the change needs it, or when a change lowers a counter,
run `uv run python benchmarks/run.py --counters-only --out benchmarks/thresholds/torch-<version>.json` on a clean
commit and commit that file alone, with a `Why:` line; a lowered counter is written down the same way, so the gate
stays tight. A sync inside another operator counts too: `torch.cholesky_solve` checks its result with one, so
`forward_dynamics` and the solvers' normal equations solve with `torch.linalg.solve_triangular`.

## Measure wall time

`benchmarks/speed.py run` times every function below on three robots of `tests/fixtures/robots/`, at each batch
size, in each mode: `panda` (fixed base, nq = nv = 8, the second finger mimics the first), `panda7` (the same with
both fingers fixed, the 7-dof arm other libraries ship) and `g1` (free-flyer root, nq 36, nv 35). On CPU, pin one core
whose SMT sibling is idle; on CUDA, check that no one else uses the GPU, since every row records the other processes
it finds there. Never time CPU and CUDA at the same time: the two runs share the host.

```bash
taskset -c 5 uv run python benchmarks/speed.py run --device cpu --grad --out cpu.json
taskset -c 5 uv run python benchmarks/speed.py run --device cpu --dtype float64 --robots panda7,g1 --out cpu64.json
CUDA_VISIBLE_DEVICES=0 uv run python benchmarks/speed.py run --device cuda --grad --out cuda.json
uv run python benchmarks/speed.py run --functions forward_kinematics,inverse_dynamics --robots panda7 --modes eager
PYTHONPATH=/path/to/other/checkout/src uv run python benchmarks/speed.py run --out other.json
```

The last line times another version of the library with this script: `PYTHONPATH` puts that checkout's `src/` first,
and `meta` names the library measured.

| Flag | Default | Meaning |
|------|---------|---------|
| `--device` | `cpu` | `cpu` or `cuda` |
| `--dtype` | `float32` | or `float64` |
| `--grad` | off | add a forward-and-backward row for each kinematics and dynamics function |
| `--functions` | all | comma-separated names from the table below |
| `--robots` | `panda,panda7,g1` | any of them |
| `--batches` | `1,64,1024,4096,16384,65536` | batch sizes `B` of `q (B, nq)` |
| `--modes` | `eager,compile`, and `reduce-overhead` on CUDA | `eager`; `compile` is `kinematics.compiled(fn)`; `reduce-overhead` is `kinematics.compiled(fn, mode="reduce-overhead")`, which replays CUDA graphs. A solve, whose loop breaks the graph, and a library version without `kinematics.compiled` use `torch.compile(fn)` with the same mode |
| `--repeats` | `15` | most samples per row |
| `--threads` | `1` | Torch threads for the run |
| `--max-call-s` | `5` | skip the larger batches of a function, pass and mode once one call takes longer |
| `--max-compile-s` | `300` | the same, once one compile takes longer |
| `--out` | `benchmarks/results/speed_<commit>_<device>.json` | output file, rewritten after every row |

`state` is `forward_kinematics(model, q)`, and each call builds it, as a user's call does. Each call returns what it
computes, so compiled code cannot drop it.

| Function | One call |
|----------|----------|
| `forward_kinematics` | `forward_kinematics(model, q).joint_transforms`: the joints alone |
| `forward_kinematics_frames` | the joint and the frame transforms of the state |
| `joint_poses` | `forward_kinematics(model, q).joint_poses` |
| `frame_jacobian` | `frame_jacobian(model, state, frame=...)` of `panda_hand` or G1's `left_rubber_hand` |
| `joint_jacobians` | `state.jacobian_columns`, the column table every joint's Jacobian masks |
| `inverse_dynamics` | `inverse_dynamics(model, state, v=v, a=a)` |
| `mass_matrix` | `mass_matrix(model, state)` |
| `forward_dynamics` | `forward_dynamics(model, state, v=v, tau=tau)` |
| `center_of_mass` | `center_of_mass(model, state)` |
| `ik_step` | one `LevenbergMarquardt.step` on `terms.ik_problem`, from `init` at the start |
| `solve_ik` | `solve_ik(model, targets, q0=start)`: the default call, one start |
| `solve_ik_multistart` | the same with `seeds=32, iterations=16`; a library before `seeds=` runs `restarts=4`, five starts in sequence |
| `ik_step_constrained` | one `SequentialQP.step` on the same problem with an obstacle sphere and the robot's own sphere pairs as clearance rows |
| `solve_ik_constrained` | 50 such steps (`Phase(stop_every=0)`); also `max_violation_m` and `within_1e-4`, over every sphere pair |
| `track` | one `optim.track` call on the `ik_step` problem (its bounds the box), warm started from the call before: a control loop's steady tick |
| `track_constrained` | the same with the clearance rows of `ik_step_constrained` |

With `--grad`, a row of each kinematics and dynamics function also times
`torch.autograd.grad(sum(output.sum()), q)` (`grad` true in the row).

The IK start is the Panda's home pose (joints 0, -30, 0, -135, 0, 100 and 0 degrees), or G1 standing with its base at
height 0.793 m and every joint at zero. The targets are the poses of `panda_hand`, or of both hands and both ankles of
G1, at seeded configurations with every limited joint uniform within its limits and G1's base at the start pose;
batch `B` takes the first `B`, so a batch above 1024 solves fresh targets. A solve row records `iterations`
(`Result.iterations`, summed over sequential restarts) and `seeds` (starts per target), `success_rate` (every target within 1 mm and 0.01 rad) over
`success_targets` targets, and the median position and rotation errors. A batch below 1024 reports the accuracy of an
eager solve of the first 1024 targets, never of one sample: a solve treats each sample on its own.

Each row times one call after warm-up calls, as the median over at least 3 and at most `--repeats` samples; a sample
is one or more calls, at least 10 ms long, between `torch.cuda.synchronize` calls. Compiles are cold: the run first
compiles a throwaway function, so the compiler's one-time start-up is paid once, then each compiled row runs after
`torch.compiler.reset()` in an empty compile cache of its own and records as `compile_seconds` the warm-up calls
that compile a graph, summed (`kinematics.compiled` runs a key's first call eagerly and compiles on the second; a
solve compiles its step inside, once per key it meets), then
`graph_breaks`, `graphs`, `recompiles` after the warm-up (counted with the garbage collector off, as the counter
gate does), `cudagraph_skips` and `max_rel_diff_vs_eager`, the largest `max |compiled - eager| / max |eager|` over the
outputs of a replayed call, and `adapter`, the compiler used (`kinematics.compiled`, `torch.compile`, or for a solve
`solve_ik(step=)`: its LM step compiled once by `kinematics.compiled` and passed to every call). A row that
raises records its `error`; the larger batches of that function, pass and mode
are then recorded as `skipped`, with the reason. `adapt` in the script is the one place that turns a mode into a
callable.

On CPU the run adds Pinocchio rows (`float64`, batch 1, one call per sample, on the same three models) for every
function Pinocchio has. The run re-executes itself with the allocator thresholds of the last section fixed, so a large
batch's time does not depend on what the process freed before. `meta` records the commit of the script, the library
measured (`src_root`, its commit, the tree of its `src/` and whether it is dirty), torch and CUDA versions, host,
device name, threads, the CPU affinity and how busy the cores' SMT siblings were at start and end, the allocator
variables, the load average and, on CUDA, the GPU's utilization, memory and other processes at start and end.

## Compare runs and libraries

Every row of a run, and every row of another library's timings, uses one schema, so a JSON written elsewhere
reads as a reference column:

| Field | Meaning |
|-------|---------|
| `library` | `better_robot`, `pinocchio`, or the other library's name |
| `function` | a name from the function table |
| `robot` | `panda`, `panda7` or `g1` |
| `device` | `cpu` or `cuda` |
| `dtype` | `float32` or `float64` |
| `batch` | samples per call |
| `mode` | `eager`, `compile`, `reduce-overhead`; another library names its own |
| `seconds_per_call` | median wall time of one call; absent when the row has `error` or `skipped` |
| `compile_seconds` | the warm-up calls of a compiled row that compile a graph, summed, when there is one |

A run's rows add `grad`, `nq`, `nv`, `seconds_min`, `seconds_spread` (interquartile range over the median), `samples`,
`page_faults` per call, `loadavg` and, on CUDA, `gpu_other_pids`: a row with any was timed on a shared GPU.

```bash
uv run python benchmarks/speed.py table after.json --before before.json
uv run python benchmarks/speed.py table cpu.json cpu64.json --reference other_cpu.json --batches 1,4096
uv run python benchmarks/speed.py table cuda.json --reference cpu.json other_cuda.json
```

`table` prints the rows of one or more runs as two markdown tables, kinematics and dynamics first, then IK: for each
row its microseconds per sample, spread, sample count, page faults per call and compile seconds; an IK row adds
iterations, microseconds per iteration and success. With `--before`, the same row of older runs and the speedup
`before/after`; with `--reference`, and for the runs' own Pinocchio rows, each other library's microseconds per
sample, its dtype and mode (and `nv` when it differs from ours), on IK rows its iterations, microseconds per iteration
and success, and the ratio `ours/<library>`, the times BetterRobot is slower. A ratio within the two rows' spreads is
marked `~`, one without a spread `?`, one across dtypes `*`; a pair that computes different things reads `n/c`. A
reference row matches on function, robot, device and batch, preferring the same dtype; another library's 7-dof Panda
matches `panda7`; a Pinocchio row, one call per sample on the CPU, matches every device and batch at its batch-1
time. CPU rows above batch 4096 are flagged, since page faults dominate them. `--functions`, `--robots`, `--batches`
and `--modes` select rows.

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
