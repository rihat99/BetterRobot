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

The owner chose opt-in compiled entry points inside the library, with eager staying the default (task of
2026-11-15, answer 2).

## Decision
We will provide one entry point, `kinematics.compiled(fn, *, mode="default" | "reduce-overhead")`, in
`src/better_robot/kinematics/compilation.py`, and `torch.compile` is called nowhere else and only when the caller
calls `compiled`. It compiles `fn` into one graph (`fullgraph=True`) on the first call; it first runs `fn` eagerly
once under each grad mode, so plans and memoised model values exist before the trace; `Model._value` is read, never
written, under `torch.compile`, keyed by the grad mode alone (Dynamo cannot trace the inference-mode check, and no
such value is an inference tensor); batch sizes follow `torch.compile`'s own policy (the first size static, a second
size one dynamic graph, batch 1 its own graph); under `"reduce-overhead"` every CUDA tensor of the result is cloned:
in tuples, lists and dicts, and in the fields of a dataclass result (a `KinematicState`, a solver `State`) and their
containers, while a field that is itself a dataclass (the state's model) stays the caller's object. A solver step is
wrapped like any other function, `compiled(lambda state: solver.step(problem, state))`.

The library's kernels may choose the form `torch.compile` traces: `small_matmul` writes a small batched product as a
multiply and a sum under `torch.compiler.is_compiling()`, which inductor fuses, and as `@` in eager mode, where that
form is slower. A caller's own `torch.compile` of a public call gets the same forms.

Still forbidden: compiling at import, decorating a library function with `torch.compile`, changing a process-wide
setting (an inductor or Dynamo config, a decomposition table: `src` imports neither `torch._inductor` nor
`torch._dynamo`), and any custom kernel or second backend (record [0001](0001-torch-only.md)).

## Example
```python { .no-run }
from better_robot.kinematics import compiled, forward_kinematics

fk = compiled(lambda q: forward_kinematics(model, q).joint_transforms)
fk(q)   # eager once, then compiles (seconds); later calls run the graph
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
0.97 to 1.02 of before). The first call costs one eager call plus the compile: 2 to 16 s per graph on one core. A
call under a grad mode not seen before runs eagerly once more and compiles a graph for that mode.

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
reads the model memo and traces the multiply-and-sum forms.

## Enforcement
C-PERF-5: `tests/contract/test_performance_rules.py::test_only_the_entry_point_compiles_and_nothing_builds_a_kernel`;
`tests/kinematics/test_compilation.py` (one graph from a cold model, equal to eager; CUDA-graph capture of
`forward_dynamics` with each result kept); `tests/contract/test_caches.py::test_a_compiled_call_reads_the_model_memo_and_writes_nothing`.

## Open questions
- Where `compiled` lives. It sits in `kinematics` because `small_matmul` must be importable by `kinematics` and
  `dynamics` (`dynamics` may import `kinematics`, not the reverse) and C-PERF-5's check names one module; but it
  wraps dynamics functions and solver steps too, so `kinematics.compiled` reads oddly beside `dynamics` and `optim`.
  The alternatives, each a rule change for the owner: (a) the root, `better_robot.compiled`, a fifteenth first-level
  name: `FIRST_LEVEL` of `tests/contract/test_public_surface.py` grows (gate data) against records 0019, 0024 and
  0035, which keep the root to the names a first script needs; (b) a package of its own, `better_robot.compilation`:
  `SUBPACKAGES` grows to thirteen and the edge table `ALLOWED` of `tests/contract/test_layer_dependencies.py` gets a
  row importing only `exceptions`, with `kinematics` and `dynamics` allowed to import it, for a package of one module
  and two functions (principle 1.2 prefers few deep modules); (c) `optim`, which already imports only torch and
  `exceptions` and is where the solver steps live: `compiled` would move there and `small_matmul` stay in
  `kinematics`, and C-PERF-5's check would name `optim`'s module instead.
