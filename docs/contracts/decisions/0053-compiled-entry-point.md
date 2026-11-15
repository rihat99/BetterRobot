# 0053 The library compiles on request, through one entry point

**Status:** Proposed (integrator, awaiting owner). **Date:** 2026-11-15

## Context
C-PERF-5 left compilation to the caller: no module calls `torch.compile`. A caller who compiles a public call meets
the same traps each time (torch 2.13, one CPU core, float32, batch 4096 unless named):

- A model's first call builds its plans. Traced, it gives a graph that rebuilds them (386 operators for the Panda's
  FK, 1085 for the G1's) and a recompile on the next call. Both memos stepped aside under `torch.compile`, so the
  placements folded into the FK map were derived again in every compiled call (78 operators instead of 33).
- Inductor leaves the batched products of small matrices (`3x4 @ 4x4`, `3x3 @ 3x1`, `6x6 @ 6x1`) as extern
  batched matrix products: compiled FK took 4.9 ms against 4.8 eager on the Panda, 17.9 against 18.9 on the G1.
- `mode="reduce-overhead"` replays CUDA graphs into one memory pool: the next call overwrites the tensors the
  previous one returned, without an error (a kept Panda inverse-dynamics result differed from its own value by 1.9
  times its largest entry after the next replay; the G1's by 1.7). A host sync inside the call (the Cholesky check of `forward_dynamics`, removed in 0.3.0)
  invalidates the capture.
- Every new batch size can recompile; compiling for every size from the start (`dynamic=True`) made the kernels 6
  to 21 % slower.
- Torch keeps at most 8 graphs per compiled function, and `fullgraph=True` turns a ninth into an error instead of a
  fall-back to eager. Ordinary use reached it: three frames by three `reference=` values through one function, three
  grad modes by two dtypes by three batch sizes, and twelve threads making their first call at once (each compiled a
  graph of its own).
- A library error raised inside the graph reaches the caller as `torch._dynamo`'s `Unsupported`, not as the
  library's class.

The owner chose opt-in compiled entry points inside the library, with eager staying the default (task of
2026-11-15, answer 2).

## Decision
We will provide one entry point, `kinematics.compiled(fn, *, mode="default" | "reduce-overhead")`, in
`src/better_robot/kinematics/compilation.py`, and `torch.compile` is called nowhere else and only when the caller
calls `compiled`. It groups calls by key: the grad mode and inference mode, and for each argument (also inside
tuples, lists, dicts and dataclasses) a tensor's dtype, device, `requires_grad` and number of axes, a model's line
(the model it was copied from) and whether it has plans and memoised values yet, a string, number, dtype, device or
enum itself (a number in a dataclass, such as a solver state's iteration, only its type, as torch makes a changing
number dynamic), and of another object only its type (a new problem of the same kind shares the key; torch's guards
tell them apart). Each key has its own `torch.compile(fn, fullgraph=True, isolate_recompiles=True)`, so
its own budget of 8 graphs, which only batch sizes use. A key's first call runs `fn` eagerly and returns that
result, which builds the plans and memoised model values; the second compiles; `fn` runs once per call. Calls that
may compile (a new key or a new shape) hold a lock. `Model._value` is read, never written, under `torch.compile`,
keyed by the grad mode alone (Dynamo cannot trace the inference-mode check, and no such value is an inference
tensor); batch sizes follow `torch.compile`'s own policy (the first size static, a second size one dynamic graph,
batch 1 its own graph). An exception from the graph makes the call run `fn` eagerly, which raises the library's own
error class; anything else is raised again unchanged. Under `"reduce-overhead"` every CUDA tensor of the result, at
any depth of tuples, lists, dicts and dataclasses, is cloned; a `Model` stays the caller's object. A solver step is
wrapped like any other function, `compiled(lambda problem, state: solver.step(problem, state))`.

The library's kernels may choose the form `compiled` traces: while it runs its graph it sets a thread-local flag,
which `fused_forms` reads through `torch.compiler.assume_constant_result` (a constant of the graph, without a
guard), and `small_matmul` then writes a small batched product as a multiply and a sum, which inductor fuses. Eager
calls and a caller's own `torch.compile` get `@`: eagerly the multiply and sum is slower, and a backend that does not
fuse it materialises it (the G1 mass matrix at batch 4096 under `torch.compile(backend="eager")` took 294 to 302 ms
with the fused forms and 96 to 97 ms without them, as eager, on one CPU core).

Still forbidden: compiling at import, decorating a library function with `torch.compile`, changing a process-wide
setting (an inductor or Dynamo config, a decomposition table, the compiler's stance: `src` names neither
`torch._inductor` nor `torch._dynamo`, and of `torch.compiler` only `is_compiling` and `assume_constant_result`),
`torch.jit` and `torch.library`, and any custom kernel or second backend (record [0001](0001-torch-only.md)).

## Example
```python { .no-run }
from better_robot.kinematics import compiled, forward_kinematics

fk = compiled(lambda q: forward_kinematics(model, q).joint_transforms)
fk(q)   # eager, returns that result
fk(q)   # compiles (seconds); later calls run the graph
step = compiled(lambda q, v, tau: forward_dynamics(model, forward_kinematics(model, q), v=v, tau=tau),
                mode="reduce-overhead")   # CUDA graphs; each result is the caller's own tensor
```

## Consequences
On one CPU core (float32, torch 2.13) at batch 4096 the compiled Panda FK runs in 0.92 ms against 4.8 eager, 6.5
eager before this work (commit 3654884) and 5.0 for `torch.compile` of the same call without the changes above; the
G1's in 4.9 ms (18.7, 25.6, 17.7). Inverse dynamics: Panda 6.7 ms (26.5, 28.7, 13.2), G1 28.8 ms (93.5, 100.8, 47.7).
At batch 1 every one of the six functions runs 2.9 to 5 times faster than eager (Panda FK 0.041 ms against 0.12).
The mass matrix and forward dynamics gain least at batch 4096 (1.1 to 1.8 times): their composite inertias are
memory-bound. On CUDA (RTX PRO 6000) under `"reduce-overhead"` a call costs 0.12 to 0.28 ms from batch 1 to 1024,
3 to 11 times less than eager; the clone costs about 0.03 ms of that at batch 1. A compiled result differs from
eager by at most 1e-6 of its largest entry (1.2e-4 for float32 forward dynamics, whose mass matrix is ill
conditioned). Eager code is unchanged but for `small_matmul`'s branch (0.17 us per product; eager calls measured
0.97 to 1.02 of before). A key's first call is an eager call, its second the compile: 2 to 16 s per graph on one
core. Choosing the key costs about 1.5 us per call with three tensor arguments; at batch 1 a compiled call takes 1
to 4 us more than before the keys (Panda FK 0.044 ms against 0.041).

The limits that remain: a key holds at most 8 graphs, and batch sizes take three of them (the first size, one graph
for every later size, batch 1; a Panda FK key called with every batch size from 1 to 39 compiled 3). A value `fn`
reads from a closure or global variable, such as a frame name, is not in the key: each new value takes a graph of
the key and the ninth raises (a closure string changed nine times raised; a closure float became dynamic after one
recompile). A Python number argument that changes on every call (an iteration count) makes a key and a compile per
value, up to torch's 256 graphs per function. Each function `compiled` returns keeps its own graphs, so one built
per model or per call compiles again. A copy made by `with_values` and passed as an argument enters cold: its first
calls use a graph that derives the memoised values inside it; once an eager call has filled its memo it shares the
graphs of its line.

## Alternatives
- A `compile=` keyword on each public function: every function grows a keyword and its own cache of graphs, and a
  composed call (FK then dynamics, a solver step) still needs its own compile. Rejected for one wrapper.
- A process-wide change of inductor's decomposition table (the study's scratch patch, 2 to 2.8x on RNEA): it changes
  every compile in the user's process and uses a private table. Rejected.
- A custom inductor backend passing the same decompositions to `torch._inductor.compile_fx` for the entry point
  only: private API, and a caller's own `torch.compile` gains nothing. Rejected for the in-kernel form.
- Leaving the result of a CUDA-graph replay in the pool and documenting "valid until the next call": cheaper by one
  copy, but a result kept across calls is silently replaced. Rejected.
- `dynamic=True` from the first call: one graph for every size, 6 to 21 % slower kernels. Rejected for torch's
  default policy.
- Inductor's `cpp_wrapper` option, which replaces the Python wrapper of a compiled graph by C++: on CUDA it made
  the default mode about a fifth faster at batches 1 and 1024 (Panda FK 0.115 to 0.092 ms, RNEA 0.28 to 0.23) and on
  CPU at batch 1, but cost up to 9 s more compile, did not combine with CUDA graphs in our measurement (the same time
  with `triton.cudagraphs` on), and was not checked with autograd. Not taken; `mode` stays the one choice.
- A home other than `kinematics`: see the open question.

## Compatibility
One new public name, `kinematics.compiled`. No result changes; a caller's own `torch.compile` of a public call now
reads the model memo, and traces the same forms as eager.

## Enforcement
C-PERF-5: `tests/contract/test_performance_rules.py::test_only_the_entry_point_compiles_and_nothing_builds_a_kernel`;
`tests/kinematics/test_compilation.py` (one graph from a cold model; every supported call equal to eager in float32
and float64, on a floating base and a batched model; the library's error classes on a warmed call; nine values and
three grad modes by two dtypes by three sizes without an error; concurrent first calls compile one graph; one side
effect per call; the clone's traversal; CUDA-graph capture of `forward_dynamics` with each result kept, also inside a
dataclass and a tuple of states); `tests/contract/test_caches.py::test_a_compiled_call_reads_the_model_memo_and_writes_nothing`.

## Open questions
- Where `compiled` lives. The rules the repository enforces give each home these facts:
  (a) `kinematics`, as now: no gate changes; `compilation.py` imports torch, `exceptions` and `model`, all in
  `kinematics`' row of the edge table, and `dynamics` reaches `small_matmul` and `fused_forms` over its allowed edge
  to `kinematics`. The cost is the name: `kinematics.compiled` also wraps dynamics functions and solver steps.
  (b) The root, `better_robot.compiled`: `FIRST_LEVEL` of `tests/contract/test_public_surface.py` grows to fifteen
  names (gate data) against records 0019, 0024 and 0035, which keep the root to the names a first script needs; the
  code can stay in `kinematics`.
  (c) A package of its own, `better_robot.compilation`: `SUBPACKAGES` grows to thirteen and the edge table `ALLOWED`
  of `tests/contract/test_layer_dependencies.py` gets a row (`exceptions`, `model`), with `compilation` added to the
  rows of `kinematics` and `dynamics` (gate data); a package of one module, against principle 1.2's few deep modules.
  (d) `optim`, where the solver steps live: `optim` imports only torch and `exceptions` and knows nothing about
  robots, but `compiled` reads `Model` (cold or warm in the key, kept as it is in the clone), and the flag
  `fused_forms` reads must be importable by `kinematics` and `dynamics`, which may not import `optim`. It would need
  a robot-free rule for both, or a change of the edge table; `small_matmul` stays in `kinematics`, and C-PERF-5's
  check would name `optim`'s module.
