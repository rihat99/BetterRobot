> **Implementation log (2026-07-22): complete on `dev`.** Nodes now compose recursively, static bool/int inputs are supported, and `ScalarCost` contributes exact non-negative scalar penalties.
> **Validation:** 1,618 CPU tests and 49 CUDA tests pass; strict HTML and all 30 doctests pass; source is 22,622 → 22,724 lines (`+102`).
> **Finding:** No required behavior was dropped; `ScalarCost` is also re-exported from `better_robot.optim` for API consistency. See [`02_results.md`](02_results.md).

# Order 02 — Node composition, static-input contract, scalar costs

Read `plan/README.md` and `DESIGN_RULES.md` first. Order 01 must be landed.

## Motivation (verified)

A `Node` input must expose `tensor/name/trainable`
(`residuals/nodes.py:15-19`), and `Problem._freeze` requires every node's
`variables` to be leaf `Variable`s (`optim/problem.py:84-89,127`). A value
*computed* from a variable — the posed body mesh — therefore cannot feed
another node or a shipped residual, which is the root cause of BVR's
`SceneSDFNode.__init__` reaching into `body_node.variables[0]` and forking
every mesh-reading residual. Scalar penalties have the same shape problem:
`w·f(x)` is only expressible by faking a `√(2·loss)` row.

## Target semantics

1. **A `Node` input may be a `Node`.** `Node.__init__(*inputs)` accepts a
   mix of Variable-likes and `Node`s. `node.variables` remains the tuple of
   **leaf** Variables, now collected transitively (order-stable, deduped);
   direct child nodes are kept in `node.nodes`. `compute()` may call
   `child.value()`; within one evaluation epoch the child memo guarantees
   one compute.
2. **`Problem` harvests transitively.** Freeze walks `residual.nodes` and
   each node's `nodes` recursively (cycle → `ValueError` naming the
   cycle), registers **every** discovered node for evaluation scoping —
   this is the load-bearing piece: an unregistered child would keep a memo
   across candidate values, which is the exact bug class the epoch contract
   forbids. Merging by `merge_key` applies to children too (two residuals
   each building an equivalent posed-points node over the same `q` merge);
   a child shared by two parents is registered once (dedup by id, as the
   existing `_nodes` handling already does).
3. **The static-input contract.** Anything a caller will change mid-solve
   (candidate clouds, labels, targets, masks) is passed as an explicitly
   constructed named non-trainable `Variable` and swapped with
   `problem.update()`, which already reaches static variables and
   invalidates node memos (`problem.py:267-278`). Two gaps must close to
   make this real, because masks and labels are bool/int:
   - `Variable` accepts non-floating dtypes **only when**
     `trainable=False` (`variables.py:68-71` currently rejects them);
     trainable stays floating-only. A non-floating static has no tangent
     and no retraction — assert it never reaches those paths.
   - `Problem._validate_values` (`problem.py:203-209`) relaxes to: all
     *floating* variables share one dtype; *all* variables share one
     device.
   Raw tensors passed to residual/node constructors stay what they are
   today — captured at construction, not updatable, and that is now
   *documented* at the constructors. There is no auto-wrapping: residuals
   sit below optim in the DAG and cannot construct `optim.Variable`, and
   `static_value` (`residuals/utils.py:54-63`) reads values, it does not
   wrap. No setter APIs anywhere.
4. **`ScalarCost`** in `residuals/`: a `dim=1` residual
   `ScalarCost(fn, *reads, weight=1.0, name=None)` where `reads` are
   Variables and/or Nodes and `fn` returns a batch-shaped scalar with
   domain `f ≥ 0` (a documented contract, not a runtime value scan). Its
   row is the double-`where` safe square root of `2f`: exactly
   `√(2f)` where `f > 0`, exactly `0` where `f ≤ 0`, with no NaN in the
   backward. No epsilon, no dead zone. Under order 01's outer weight the
   objective contribution is exactly `w·f` for `f ≥ 0`. Document two
   honest caveats in the docstring: the gradient at exactly `f = 0` is `0`
   (the masked branch), and LM's Gauss--Newton column `∇f/√(2f)` grows as
   `f → 0`, so `ScalarCost` suits penalties that stay away from zero or
   terms whose vanishing means convergence — damping covers the tail.
   `ScalarCost` problems are **implicit-ineligible** (join order 01's
   eligibility guard; actionable error).

## Boundaries

- `RobotState` is unchanged as the standard FK node; this order adds no
  new node types beyond `ScalarCost`'s needs. Order 03 is the consumer
  that makes shipped residuals accept node inputs.
- Keep validation at the public boundary: a node input that is neither
  Variable-like nor `Node` raises `TypeError` with the offending type.
- The evaluation-scope machinery (`_begin_evaluation`/`_end_evaluation`/
  `_invalidate`) should not grow states — reuse it per discovered node.

## Acceptance

- A counting test with a two-level chain (leaf `q` → posed-points node →
  consumer node → two residuals) proves exactly one compute per node per
  evaluation epoch, and recomputation across epochs (extend the pattern of
  the existing vertical-slice counters). A shared-child variant (one child
  reached from two parent nodes) computes once.
- A stale-memo attack test: change `q` between two `objective()` calls and
  assert the nested node's output moved (fails closed if child
  registration is missing).
- Merge test: two equivalent nested nodes constructed independently merge
  at freeze; the counter shows one compute.
- `problem.update()` on a static **bool** mask Variable feeding a nested
  node changes the next evaluation; mixed float/bool problems validate;
  a trainable bool Variable raises.
- `ScalarCost`: objective equals `w·f` analytically; gradcheck (float32)
  away from zero; exact-zero `f` produces zero cost, zero gradient, no
  NaN; works under LM and `TorchOptimizer`; the implicit-ineligibility
  error fires.
- Cycle construction raises; `residuals/CLAUDE.md` + concepts page updated
  (node composition contract, the static-input contract). Full gate
  green; net src growth ≤ +150.
