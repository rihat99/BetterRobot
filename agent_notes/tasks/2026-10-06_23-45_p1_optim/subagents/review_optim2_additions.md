# Review 2: optim2 additions after 65b63bc (mars_aux, clean context, 2026-10-07)

Scope: `ArrowNormal`/`ArrowSink`, band refactor, shared-variable Jacobian pieces, `solve_constrained`,
`TorchOptim(scheduler=)`, `Phase.on_step`. Reviewer scripts were in /tmp/claude-1003/review_optim2b/ (not kept).

## Findings and dispositions
1. MAJOR. Shared-variable columns of an autodiff time-local term come from differentiating a backward pass.
   Through an op whose backward is not differentiable (custom `Function` with a `no_grad` backward, a Warp
   kernel) they came out zero, silently. Reproduced: gradient `[0,0,0]` instead of `[-1.77,-0.66,-0.34]`.
   FIXED: one ordinary backward pass checks the result; mismatch raises `ValueError` asking for a `jac`.
   Test `test_shared_columns_through_a_once_differentiable_op_fail_loudly`.
2. MINOR. `TorchOptim(scheduler=)` calls `step()` without arguments, so `ReduceLROnPlateau` fails.
   DOCUMENTED in the docstring (not supported).
3. MINOR. `solve_constrained` returned a loss that included the penalty terms.
   FIXED: `Result.loss` is the loss of the other terms; asserted in the constrained test.

## Verified correct by the reviewer (against `dense=True` and plain autograd)
Two time variables of different per-knot size, three shared variables in mixed order (one unread), term reading
a shared variable only through a cached value, window 3 with padded last block, shared-only dense-rows term,
scalar term, Huber kernels, B=1, float32, free masks (per-knot, shared, per-batch), active bounds on time and
shared columns, `predicted_decrease`, `solve(g=)` differentiability, `check_jacobian` on `ArrowNormal`,
`solve_constrained` leaving the problem unmodified.

## Not checked by the reviewer
LM conditioning as the penalty grows; inequality constraints with analytic `jac`; windowed constraints under the
band layout; multipliers at active bounds; `x=` of `solve_constrained`; `Phase.on_step`; performance.
