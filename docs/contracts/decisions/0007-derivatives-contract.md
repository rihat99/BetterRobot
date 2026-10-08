# 0007 Derivatives contract: autograd first, analytic later behind the same signature

**Status:** Accepted · **Date:** 2026-10-08

## Context
Trajectory optimization, DDP/MPC and contact-implicit users need partial derivatives of kinematics and
dynamics (`∂τ/∂q`, `∂τ/∂v`, `∂a/∂τ`, …). Today they get them only by running `torch.func.jacrev` on the public
functions themselves: for a batched call that returns cross-batch blocks `(B, nv, B, nv)`, and for a free-flyer
model it differentiates the ambient `nq` coordinates instead of the `nv` tangent space. Pinocchio ships analytic
derivatives (Carpentier-Mansard), which are much faster at batch 1, but writing them first would freeze an API
before the shapes and conventions are settled.

## Decision
We will fix the derivatives contract now and implement it with autograd first:

- **Tangent-space partials.** Every derivative is taken with respect to the tangent of the configuration
  (`nv`), through `Model.integrate` (ADR [0005](0005-nq-nv.md)), never the ambient `nq` coordinates. Shapes are
  per sample: `(B..., n_out, nv)`, no cross-batch blocks.
- **One public entry per quantity**, named `<quantity>_derivatives(model, state, ...)` (for example
  `inverse_dynamics_derivatives`, `forward_dynamics_derivatives`), taking the same arguments as the quantity
  and returning a tuple of partials in a documented order.
- **Analytic implementations are a later drop-in** behind the same signature. The autograd version stays as
  the test oracle; an analytic version must match it (and Pinocchio, where it exists) before it replaces it.
- **Batch first; compile-clean is the path to batch 1.** The public FK, RNEA and CRBA paths trace as one
  `torch.compile(fullgraph=True)` graph (tested in `tests/kinematics/test_fk_compile.py`); compile and CUDA
  graphs, not a second implementation, are the first answer to batch-1 latency.
- **Warp stays a reserved option.** A kernel backend is not planned; if one lands it sits behind the same
  signatures and raises `BackendNotAvailableError` when absent (ADR [0001](0001-torch-only.md) still holds
  until a new ADR supersedes it).

## Consequences
The `*_derivatives` names are reserved and their output shapes fixed before any implementation lands. Callers
that use `jacrev` today keep working; the derivative entries will replace that idiom without a signature change
when analytic code arrives. Any change to the placement or inertia caches must keep the fullgraph test green.

## Alternatives considered
- Analytic derivatives first: fastest at batch 1, but freezes conventions before autograd has validated them.
- No derivatives API (users call `jacrev`): wrong shapes for batches and ambient gradients for free flyers.
- A Warp or C++ kernel now: contradicts ADR 0001 and adds a second implementation to keep in parity.

## Enforcement
`C-DIFF-1`..`C-DIFF-3` ([04](../04-differentiability.md)); `tests/kinematics/test_fk_compile.py` (fullgraph
FK/RNEA/CRBA on the Panda); Pinocchio parity tests under `tests/test_pinocchio/` once an analytic version lands.
