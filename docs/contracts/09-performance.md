# 09 Performance

Performance rules are the ones that cannot be retrofitted: a host sync or a Python loop over the batch inside
a hot path forces a rewrite later. Everything else (fusing ops, caching tables) is measured before it is done.

## No host synchronisation in hot paths: see `C-TENS-4`

The performance reason for `C-TENS-4` (`02`): `.item()`, `.cpu()`, `bool(tensor)` and data-dependent Python
control flow stall the GPU queue. Hot paths are the per-call code under `lie`, `kinematics`, `dynamics` and the
solver step in `optim/solvers.py`; once-per-model setup (plan building) is not a hot path and is exempted with
`# bench-ok`. This section has no rule ID of its own; `C-TENS-4` and its enforcer cover it.

## C-PERF-2 Batch, do not loop (MUST)

Work over a batch axis is one tensor op. A Python loop is allowed only over a *structural* axis whose length
is a Python int fixed by the model (joints in a kinematic chain, LM iterations), never over batch elements.

```python { .no-run }
# good  (illustrative; real entry point: kinematics/state.py:151)
for joint in model.joints:                 # structural, len = model.njoints
    X_WB = X_WB @ joint_placement(joint, q[..., joint.idx_q])   # batched over `...`
# bad
for b in range(q.shape[0]): ...
```

Enforcement: `[review]`; `tests/bench/` guards the entry points below.

## C-PERF-3 Hot paths have a benchmark and are not slowed silently (SHOULD)

`forward_kinematics_state`, `frame_jacobian`, `rnea`/`crba`, and `optim.solve` have a benchmark under
`tests/bench/` (`-m bench`). A change that slows one by more than 20 % needs a reason in the PR.

Enforcement: `tests/bench/` (`uv run pytest -m bench`); `[review]`.

## C-PERF-4 Measure before optimising (SHOULD)

An optimisation PR shows the benchmark before and after. No speculative caching, no `torch.compile` in the
library, no custom kernels without an ADR.

Enforcement: `[review]`.
