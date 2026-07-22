> **Implementation log (2026-07-22): Order 05 scope complete on `dev`.** Layout-aware first-order state, `resume()`, scheduler ownership, the staged-fit guide, and vertical slice v2 are implemented.
> **Validation:** 344 optimizer CPU tests, 5 optimizer CUDA tests, strict HTML, and all 31 doctests pass; Python source growth is net `+55` lines. Full-tree gates currently fail only in concurrent Warp FK/RNEA work outside this order.
> **Finding:** The phase driver is 90 lines and uses plain data; phase objects, live kernel scale, auto-skip smoothness, and a metrics framework remain intentionally rejected. See [`05_results.md`](05_results.md).

# Order 05 — TorchOptimizer phases, staged-fit guide, vertical slice v2

Read `plan/README.md` and `DESIGN_RULES.md` first. Orders 01–04 must be
landed. This order assembles the round: the optimizer knobs a staged fit
needs, the documentation that makes the phase pattern official, and one
in-tree proof that the wishlist plumbing dissolved.

## Motivation (verified)

- `TorchOptimizer` exposes no learning-rate control after construction;
  BVR decays LR by poking `opt._optimizer.param_groups[0]["lr"]`
  (`optimizer.py:391-393`).
- Three current semantics make a multi-phase loop with one optimizer
  impossible, which is why BVR hand-rolled its runner:
  - **every** `problem.update()` bumps `_update_serial` and
    `TorchOptimizer._ensure_current_layout` responds by rebuilding the
    inner optimizer (`optimizers.py:159-161`) — swapping a static scene
    cloud between phases silently discards Adam moments even though the
    tangent layout did not change;
  - terminal statuses stick: `step()` early-returns once nothing is
    RUNNING (`optimizers.py:211-213`) and `optimize()` converts RUNNING to
    MAXITER (`optimizers.py:242-248`), so a second `optimize()` call is a
    no-op — phase 2 never runs;
  - there is no way to resume without `reset()`, which discards moments by
    design.

## Work items

1. **Layout-aware refresh.** `_ensure_current_layout` compares the actual
   layout (trainable names, `free_dim`s, batch shape, dtype/device)
   instead of the raw serial: a same-layout `problem.update()` (static
   swap, target change) keeps the inner optimizer, buffers, and scheduler,
   and only the memo invalidation that `update()` already did takes
   effect. A layout change rebuilds as today. LM's serial-based warm-start
   handling is already value-aware — leave it alone.
2. **`resume()`.** On the `Optimizer` base: return every terminal batch
   element to RUNNING while keeping variable values, iteration counts
   (cumulative), and — for `TorchOptimizer` — moments, buffers, and
   scheduler state. This is the official way to continue after `optimize()`
   hit MAXITER or after a phase flipped `enabled`/weights. LM's `resume()`
   keeps variable values and drops only its damping/acceptance state
   (document why: stale damping from another phase's objective is not a
   warm start).
3. **LR scheduler.** `TorchOptimizer(..., scheduler=None)` where
   `scheduler` is a callable `torch.optim.Optimizer → LRScheduler`
   (factory, mirroring `optimizer_cls`). Built in `reset()` after the
   inner optimizer; stepped exactly once per `step()` that ran an
   optimizer update (not on early-return paths — pin with a test);
   preserved by the layout-aware refresh; rebuilt on real layout change.
   No milestone DSL, no BR-owned schedule types. Validate the factory's
   return like `optimizer_cls` is validated.
4. **Staged-fit how-to** (`docs/`, How-To quadrant, follow `sphinx-docs` /
   `diataxis-docs` conventions): one curriculum walked end to end — build
   problem once; phase table as plain data; per phase set `enabled` /
   weights, swap static inputs via `problem.update()` (moments survive),
   `resume()` between phases; a frozen-variable warm-up phase as a
   separate problem; one `TorchOptimizer` with a scheduler across Adam
   phases; L-BFGS polish as a second optimizer; `problem.term_costs()`
   for logging. State explicitly why there is no Phase object (plan
   decision 4) so the next downstream doesn't wish for one.
5. **Vertical slice v2.** Extend or replace the existing
   `tests/optim/test_vertical_slice.py` + `slice_support.py` scenario with
   the BVR shape: a synthetic articulated "body" posed by a node chain
   (order 02), feeding gated scene + chamfer + point-projection residuals
   (order 03) with outer weights and `mean_active` (order 01), a
   `ScalarCost` prior, a frozen-root warm-up phase (order 04), an LR
   schedule across two Adam phases joined by `resume()`, and
   `term_costs()` assertions. Keep the evaluation-counting discipline of
   the current slice (it caught the double-forward bug this round fixed).
   The phase-driving code must stay under ~150 lines and read like a
   description of the fit — that number is this round's success
   criterion, mirrored from the wishlist's closing paragraph.
6. **Docs/roadmap/changelog sync.** `docs/reference/roadmap.md`: resolve
   the jerk-smoothness entry, record the state-dependent smoothness
   blocks deferral from order 04, add nothing speculative;
   `docs/CHANGELOG.md` entry for the round; consolidate `MIGRATION.md`
   rows from orders 01–04 into one coherent section; sweep
   `optim/CLAUDE.md`, `residuals/CLAUDE.md`, and the concept pages for
   contract lines any order falsified — including the optimizer-lifecycle
   lines this order itself changes (`reset` vs `resume`, layout-aware
   refresh).

## Acceptance

- Same-layout `problem.update()` provably preserves Adam moments (compare
  `state` tensors) and scheduler state; batch-shape update still rebuilds
  (the existing regression tests for that stay green).
- `resume()` after MAXITER continues and converges; `resume()` after an
  `enabled` flip re-runs only what the new objective demands; LM
  `resume()` drops damping but keeps values.
- Scheduler: decay observable across `optimize()`; no step on terminal /
  no-op paths; works under L-BFGS; survives same-layout updates.
- The staged-fit guide builds clean under strict Sphinx and its code
  fences execute (doctest or the marked-fence mechanism the docs already
  use).
- Vertical slice v2 passes with exact evaluation counts and converges;
  its phase-driving section is < ~150 lines.
- Full CPU gate green; CUDA suite green; net src growth ≤ +150.
- Results file closes the round: per-wishlist-item disposition table
  (shipped / partially shipped / rejected with pointer), so the BVR side
  can be updated against it.
