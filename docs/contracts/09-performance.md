# 09 Performance

Performance rules are the ones that cannot be retrofitted: a host sync or a Python loop over the batch inside
a hot path forces a rewrite later. Everything else (fusing ops, caching tables) is measured before it is done.

## No host synchronisation in hot paths: see `C-TENS-4`

There is no `C-PERF-1`: the first performance rule lives in `02` as `C-TENS-4`, so this page starts at `C-PERF-2`.

The performance reason for `C-TENS-4` (`02`): `.item()`, `.cpu()`, `bool(tensor)` and data-dependent Python
control flow stall the GPU queue. Hot paths are the per-call code under `lie`, `spatial`, `kinematics`, `dynamics`
and `terms`, the solver step in `src/better_robot/optim/solvers.py`, and the linearization and linear algebra the
step runs every iteration, `src/better_robot/optim/linearization.py` and `src/better_robot/optim/linear.py`; this is
the list `tools/hot_path_lint.py` watches, and `tests/contract/test_hot_path_lint.py` keeps the two equal.
Once-per-model setup (plan building) is not a hot path and is exempted with `# bench-ok`. So is the `LBFGS` line
search, which reads one flag per trial: a trial is a full loss evaluation,
dearer than the sync that skips the ones left once every problem has its step. This section has no rule ID of its
own; `C-TENS-4` and its enforcer cover it.

## C-PERF-2 Batch, do not loop (MUST)

Work over a batch axis is one tensor op. A Python loop is allowed only over a *structural* axis whose length
is a Python int fixed by the model (joints in a kinematic chain, LM iterations), never over batch elements.

```python { .no-run }
# good  (illustrative; real entry point: src/better_robot/kinematics/forward.py::forward_kinematics)
for joint in model.joints:                 # structural, len = model.njoints
    X_WB = X_WB @ joint_placement(joint, q[..., joint.idx_q])   # batched over `...`
# bad
for b in range(q.shape[0]): ...
```

Enforcement: `tests/contract/test_performance_rules.py::test_a_batch_dispatches_the_same_operators_at_every_size`
(every entry point of the contract tests' public-ops registry dispatches as many torch operators for a batch of
two as for a batch of seven).

## C-PERF-5 The library compiles nothing and ships no custom kernel (MUST)

Compilation is the caller's choice: no module calls or decorates with `torch.compile`; the public kinematics and
dynamics trace as one graph when the caller compiles them (`tests/kinematics/test_fk_compile.py`). Torch is the only
implementation (record [0001](decisions/0001-torch-only.md)): no C++ extension, Triton, Numba or CuPy kernel.

Enforcement: `tests/contract/test_performance_rules.py::test_src_compiles_nothing_and_builds_no_kernel`.

## C-PERF-3 The counters of the hot entry points do not grow (MUST)

`benchmarks/run.py --counters-only` measures, on CPU, the graph breaks, recompiles, top-level operators and
host-syncing operators of every hot entry point and compares them with `benchmarks/thresholds/torch-<version>.json`;
a counter above its threshold fails. Wall time is a measurement, not a gate: an optimisation shows
`benchmarks/run.py` before and after in its review ([Benchmark and profile](../guides/benchmark-and-profile.md)).
`C-PERF-4` ("Measure before optimising") is retired and its ID is not reused.

Enforcement: `tests/contract/test_compile_counters.py`.
