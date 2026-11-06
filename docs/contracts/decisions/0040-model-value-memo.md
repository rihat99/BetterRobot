# 0040 A value-side memo on `Model`, keyed by grad mode

**Status:** Accepted (owner). **Date:** 2026-11-06

## Context
Record [0011](0011-caches-and-plans.md) removed the `_version`-keyed caches and converts the `(njoints, 7)` joint
placements, the `(nframes, 7)` frame placements and the packed `(nbodies, 10)` inertias to matrices on every call,
with a threshold: a value-side memo on `Model`, keyed by grad mode, is added by its own record when the FK benchmark
shows that conversion above 10 % of a batch-1 FK call on CPU. R1 measured it (CPU, one thread, float32, one idle
core, the R0 tree at 046f61f interleaved with the R1 tree, twice each): `placement_matrices` costs 0.17 ms per call,
44 % of a batch-1 Panda FK call (0.189 -> 0.392 ms; G1 fixture 0.227 -> 0.44..0.52 ms) and under 1.2 % at batch 4096
(Panda 14.3 ms, G1 55 ms, inside the baseline spread). Batch-1 RNEA on the Panda went 0.536 -> 0.846 ms; the
0.11 ms beyond the FK delta is the per-call `packed_inertias_to_6x6`. `se3.to_matrix` is 110 us per call, about 25
small eager ops, against 140 us for the FK kernel itself.

## Decision
We will add `Model._memo: dict` (`init=False`, `repr=False`, `compare=False`; dropped by `dataclasses.replace`,
`to` and `with_values` like `_plans`) holding values derived from the tensor fields: the placement matrices
`(I..., njoints, 4, 4)` and `(I..., nframes, 4, 4)` read by `src/better_robot/kinematics/_fk_lean.py::placement_matrices`, and the
packed spatial inertias `(I..., nbodies, 6, 6)` read by `src/better_robot/dynamics/_world.py::world_kinematics`. One method,
`Model._value(kind, derive)`, serves all three: the key is `(kind, torch.is_grad_enabled(), flags)` with `flags`
the `requires_grad` flag of every tensor field; an entry is written only when its value carries no autograd graph
(`not value.requires_grad`) and is read only under the grad mode and flags it was made with. A value with a graph is
recomputed per call, so each differentiable call still owns its graph. Under `torch.inference_mode` the memo is
neither read nor written: the value is derived per call; so under `torch.compile`, where the derivation is traced
into the graph (Dynamo cannot trace the inference-mode check). No `_version`, no functorch probe: a model whose tensors are
edited in place after construction is outside C-STATE-1 (treat a model as read-only); the memo then serves the old
value, as any plan would. `requires_grad_()` on a model tensor is such an edit; the flags in the key make it
harmless anyway.

## Example
```python { .no-run }
def placement_matrices(model: Model) -> tuple[torch.Tensor, torch.Tensor]:
    joints = model._value("joint_matrices", lambda m: se3.to_matrix(m.joint_placements))
    frames = model._value("frame_matrices", lambda m: se3.to_matrix(m.frame_placements))
    return joints, frames

with torch.no_grad():
    forward_kinematics(model, q)        # memoises the matrices under ("joint_matrices", False, flags)
forward_kinematics(model_with_grad, q)  # placements with a graph: converted per call, never memoised
```

## Consequences
Batch-1 FK on CPU returns to R0's cost (measurement below). The memo changes cost, never a result: the sweep in
`tests/contract/test_caches.py` keeps forbidding `object.__setattr__`, `_version` and `torch._C`, and gains the two
memo checks (empty after `replace`/`to`/`with_values`; a grad-carrying value is never memoised and a graph-free one is
served under its own grad mode). Contract 03 C-STATE-4 lists `_memo` next to `_plans`; the pytree registration keeps
`_memo` out of the leaves, as `_plans`. Under `torch.inference_mode` nothing is memoised: a value made there is an
inference tensor, and `torch.enable_grad` inside inference mode (what `optim.Problem.gradient` does) would file it
under the grad-mode key, where a later autograd call cannot save it for backward (found by the R1+R2 review).

Measurement after the memo (same script, core and interleaving as R1's; CPU, one thread, float32):

| model | batch | FK at R0 (046f61f), interleaved with each | FK before the memo (ef66e6a) | FK with the memo |
|---|---|---|---|---|
| Panda | 1 | 0.1896 / 0.1905 ms; 0.1914 / 0.1910 ms | 0.3926 / 0.3959 ms | 0.1805 / 0.1806 ms |
| Panda | 4096 | 13.73 / 13.83 ms; 13.94 / 13.77 ms | 14.17 / 14.03 ms | 13.63 / 13.65 ms |

Batch-1 RNEA with FK on the Panda: 0.534 / 0.544 ms at R0, 0.848 / 0.859 ms before, 0.538 / 0.534 ms with the memo.
`placement_matrices` alone: 0.171 ms before, 0.0005 ms with the memo (0.3 % of the batch-1 call). The G1 fixture at
batch 1: 0.427 ms before, 0.214 ms with the memo.

## Alternatives
A fused conversion in `_fk_lean` (outer product and one constant matmul, about five ops) cuts the cost to roughly
0.04 ms, still 15 % at batch 1, and duplicates `lie.se3.to_matrix` (two ways to do one thing). Per-call conversion
with `torch.compile` for batch-1 speed (the roadmap's stated route): the eager batch-1 call stays 2x slower than R0
until then. A memo keyed by tensor version: the hidden input 0011 removed.

## Compatibility
No public signature changes. A caller that mutates a model's tensors in place after a query sees the memoised
value; such a model was already outside C-STATE-1.

## Enforcement
`tests/contract/test_caches.py` (cache fields are not constructor arguments; empty after `replace`; grad-carrying
values are never memoised; `requires_grad_()` after a query still yields a gradient; a leaf stepped in place is not
seen; nothing is memoised under inference mode); `tests/kinematics/test_fk_compile.py` (the public FK, RNEA and CRBA
still trace as one graph).

## Open questions
None.
