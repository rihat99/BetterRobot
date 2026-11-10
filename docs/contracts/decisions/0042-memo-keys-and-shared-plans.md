# 0042 The model memo is keyed by the grad flags, copies share their plans, and the world geometry is cheaper

**Status:** Accepted (integrator, 2026-11-10). **Date:** 2026-11-10

## Context
Record [0040](0040-model-value-memo.md) keys `Model._memo` by `(kind, grad mode)` and says it is dropped "like
`_plans`" by `dataclasses.replace`, `to` and `with_values`. Two failures followed: `requires_grad_()` on a model
tensor after a query was served a graph-free value, and a value memoised under `torch.inference_mode` is an
inference tensor that a later autograd call cannot save for backward. Rebuilding every plan for each `to` or
`with_values` copy cost a fit loop that swaps values every iteration. Record [0011](0011-caches-and-plans.md)'s
measurements and the operator counts in the Consequences of [0041](0041-world-geometry-on-the-state.md) predate the
kernel work that followed.

## Decision
We will key the memo by `(kind, torch.is_grad_enabled(), flags)`, with `flags` the `requires_grad` flag of every
tensor field, neither read nor write it under `torch.inference_mode` or `torch.compile` (the value is derived per
call there), and let a `to` or `with_values` copy share the plans of the first model of its line through
`Model._plan_owner`, while its value memo still starts empty.

## Example
```python { .no-run }
def _value(self, kind, derive):
    if torch.compiler.is_compiling() or torch.is_inference_mode_enabled():
        return derive(self)                                   # never stored: no inference tensor in the memo
    flags = tuple(getattr(self, name).requires_grad for name in _TENSOR_FIELDS)
    key = (kind, torch.is_grad_enabled(), flags)
    ...

light = model.with_values(body_masses=0.5 * model.body_masses)   # light._plan_owner is model: no plan is rebuilt
```

## Consequences
A model tensor switched to `requires_grad_()` after a query still yields a gradient, and inference-mode callers
never poison the memo; the cost is one tuple of flags per lookup. The plan link is strong: a copy keeps the first
model of its line (its tensors and its value memo) alive and pickles it along; a weak link would make every copy
unpicklable and rebuild the plans once the first model is dropped. A `dataclasses.replace` that changes a static
field has another `structure_key` and builds its own plans. The world geometry of 0041 is now derived by a column
table of 10 operators (one gather of `[R | p]` and one matmul against `JacobianPlan.column_subspace`) and a shift of
13; its results are unchanged (the 1032 Pinocchio parity cases are identical), and `rnea`, `crba` and `aba` at
batch 1 on CPU are within 5 % of the kernels at commit d10f2a9 (largest: `crba` on the Panda, +4.7 %).

## Alternatives
- Clear the memo whenever a tensor's flag changes: needs a hook on every tensor, the hidden input 0011 removed.
- Store inference-mode values under their own key: they would still be served to a later `torch.enable_grad` call
  inside inference mode, which is what `optim.Problem.gradient` does.
- Copy `_plans` into each copy: `_plans` is `init=False`, so `dataclasses.replace` cannot hand it over; a weak
  reference to the owner: see Consequences.

## Compatibility
No public name changes. A caller that edits a model tensor in place is still outside C-STATE-1; only the
`requires_grad_()` case is made harmless.

## Enforcement
C-STATE-4; `tests/contract/test_caches.py` (`test_requires_grad_after_a_query_still_yields_a_gradient`,
`test_nothing_is_memoised_under_inference_mode`, `test_values_with_a_graph_are_never_memoised`);
`tests/model/test_with_values.py::test_a_copy_shares_its_source_plans_and_computes_as_a_fresh_model`;
`tests/kinematics/test_fk_compile.py` (the public FK, RNEA and CRBA trace as one graph).

## Open questions
None.
