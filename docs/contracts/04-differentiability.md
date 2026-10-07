# 04 Differentiability

Everything that maps floats to floats is a `torch.autograd` graph and a `torch.func` (`vmap`, `grad`,
`jacrev`) transformable function. This is the reason the library exists; a non-differentiable public path is a
documented exception, never a default. See ADR [0001](decisions/0001-torch-only.md).

## C-DIFF-1 Every public float operation is differentiable (MUST)

Gradients flow from every output to every float input (`q`, `v`, model parameters) unless the docstring has a
`Differentiability` section saying what is cut and why. No `.detach()`, `torch.no_grad()` or
`requires_grad_(False)` inside library math. Documented exceptions: `optim.Problem` detaches while solving
(`optim/problem.py:278-303`; `solve` is not differentiable, `solve_layer` is, and both docstrings say so),
and `dynamics/state_manifold.py:64,72` detaches a manifold projection, which needs a `Differentiability`
section or a fix `[planned: L-6]`.

```python { .no-run }
# good  (illustrative)
"""... Differentiability: `solve` returns detached values; use `solve_layer` to backpropagate
through the solution (implicit function theorem)."""
# bad
return poses.detach()          # inside forward_kinematics_state
```

Enforcement: `tests/autograd/` and `tests/lie/test_torch_backend_gradcheck.py` (`torch.autograd.gradcheck`
in float64 on every `lie`/`spatial`/`kinematics` entry point); `[review]` for new entry points.

## C-DIFF-2 `torch.func`-safe: no Python branching on tensor values (MUST)

`vmap` traces a function once per call; `if t.any():`, `while err > tol:` with a tensor `err`, and data-
dependent shapes make the trace invalid. Use `torch.where`, boolean masks, and fixed iteration counts with
masked updates. Loop bounds come from Python ints (`model.nv`, `max_iters`), never from tensor contents.

```python { .no-run }
# good  (lie/so3.py)
small = theta < eps
coef = torch.where(small, taylor(theta), exact(theta))
# bad
if theta < eps: return taylor(theta)
```

Enforcement: `tests/autograd/test_vmap.py` [planned: L-8] (wraps each entry point in `torch.func.vmap` over a leading
batch axis and compares to the broadcast result); `[review]`.

## C-DIFF-3 Singularities have a documented safe gradient (MUST)

Functions with a removable singularity (`log` near identity, `exp` near zero, `acos` at ±1, normalisation of a
near-zero vector) use a Taylor branch under `torch.where` *and* mask the unused branch's inputs so the
gradient is finite (the "double-where" pattern). The docstring names the switch threshold.

```python { .no-run }
# good
safe_theta = torch.where(small, torch.ones_like(theta), theta)   # keeps NaN out of the dead branch's grad
coef = torch.where(small, 1 - theta**2 / 6, torch.sin(safe_theta) / safe_theta)
# bad
coef = torch.where(small, 1 - theta**2 / 6, torch.sin(theta) / theta)   # grad is NaN at theta == 0
```

Enforcement: `tests/lie/test_torch_backend_gradcheck.py` (gradcheck at the singular points);
`[review]` for new functions.

## No in-place ops on inputs, no global autograd switches: see `C-STATE-3` and `C-DEP-3`

The autograd reason for `C-STATE-3` (`03`): in-place on a leaf tensor raises at backward time. The autograd
reason for `C-DEP-3` (`07`): `torch.set_grad_enabled`, `torch.set_default_dtype`, `torch.set_default_device` and
`torch.use_deterministic_algorithms` belong to the user's program, never to the library. This section has no
rule ID of its own; those two rules and their enforcers cover it.
