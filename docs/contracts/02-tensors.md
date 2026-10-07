# 02 Tensors

Every public tensor has *trailing event axes* (the last axes carry the quantity: `(..., 7)` for a pose,
`(..., 6)` for a twist, `(..., nq)` for a configuration) and *arbitrary leading batch axes* (`...`). Batch axes
combine by Array API broadcasting, the same rule as `torch` elementwise ops: axes are aligned from the right, a
size-1 axis stretches to match, and a missing axis counts as size 1.

## C-TENS-1 Preserve dtype and device (MUST)

An output has the dtype and device of its tensor inputs. A function never casts, moves or squeezes silently.
Model parameters live on one device; a query tensor on another device is an error raised at the boundary
(`06`), not a hidden `.to()`.

```python no-run
# good  (illustrative; real entry point: kinematics/state.py:151)
out = torch.empty((*q.shape[:-1], 7), dtype=q.dtype, device=q.device)
# bad
out = torch.empty((*q.shape[:-1], 7))          # default dtype, cpu
q = q.float()                                   # silent downcast of float64 inputs
```

Enforcement: `tests/contract/test_dtype_device.py` (calls every public kinematics/dynamics/lie entry point on
float32 and float64 inputs and asserts the output dtype; skips cuda when absent).

## C-TENS-2 Leading batch axes broadcast; no fixed batch rank (MUST)

A function accepts any number of leading axes and broadcasts them across its tensor arguments. Never
`unsqueeze` to a fixed rank, never loop over `q.shape[0]`.

```python no-run
# good
X = compose(X_WA[..., None, :], X_AB)          # (B, 1, 7) with (J, 7) -> (B, J, 7)
# bad
for b in range(q.shape[0]):
    out[b] = step(q[b])
```

Enforcement: `tests/contract/test_dtype_device.py` (runs each entry point with shapes `(n,)`, `(2, n)`,
`(3, 1, n)` and compares); `[review]` for new code.

## C-TENS-3 Shapes in docstrings (MUST)

Every public function or field that takes or returns a tensor documents the shape with trailing-axes
notation. The numpy-style `Parameters`/`Returns` lines carry `(..., n)` so the shape is greppable.

```python no-run
# good
"""Return frame poses.

Returns
-------
Tensor, shape (..., nframes, 7)
    Poses ``[tx, ty, tz, qx, qy, qz, qw]`` of every frame in the world frame.
"""
# bad
"""Return frame poses."""
```

Enforcement: `tests/contract/test_docstrings.py` (every `__all__` symbol has a docstring; if its return
annotation is `Tensor`, the docstring contains `shape (`).

## C-TENS-4 No host synchronisation in library paths (MUST)

`.item()`, `.cpu()`, `.numpy()`, `float(t)`, `bool(t)` and `if t:` on a tensor force the GPU to finish and break
`torch.func` tracing. They are allowed only in `viewer`, `io` export paths, and tests. Convergence tests in
`optim` use tensor masks, not Python branches (`04`).

```python no-run
# good
converged = (step_norm < tol)               # tensor mask, stays on device
# bad
if step_norm.item() < tol: break
```

Enforcement: `tests/contract/test_hot_path_lint.py` (ast scan of `kinematics`, `dynamics`, `lie` and
`optim/solvers.py`; a line marked `# bench-ok` is exempt, e.g. once-per-model plan building in
`kinematics/_fk_lean.py`). Widening the scan to `spatial` and `terms` is `[planned: L-4]`.

## C-TENS-5 Optimisation variables carry exactly one batch axis (MUST)

An `optim` variable is a tensor `(B, *event)` and every variable of a `Problem` shares the same `B`
(`optim/problem.py:143`); the `B` problems are independent, so a term sees `(B, ...)` inputs and returns
`(B, m)` residuals and `Problem.loss` returns `(B,)`. Unlike the rest of the library, there are no arbitrary
leading batch axes here: flatten extra axes into `B` before building the problem. A rejected batch element keeps
its old value through `torch.where`, so one bad element never stalls the others.

Enforcement: `tests/optim/test_problem.py`; `[review]`.
