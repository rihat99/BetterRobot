# 0052 The memos keep inference-mode values under a key of their own

**Status:** Proposed (integrator, awaiting owner). **Date:** 2026-11-15

## Context
Record [0042](0042-memo-keys-and-shared-plans.md) bypasses `Model._value` and `KinematicState._memoised` under
`torch.inference_mode`: a value made there is an inference tensor, which a later autograd call cannot save for
backward. It rejected a key of their own for inference-mode values because such a value "would still be served to a
later `torch.enable_grad` call inside inference mode". That premise does not hold: inside inference mode autograd
records nothing, `enable_grad` or not (torch 2.13: an operation on a tensor that requires grad returns a tensor
that does not, and `torch.autograd.grad` raises), so an inference tensor served there breaks nothing. `torch.func`
transforms work there, with inference tensors as constants. The bypass made every batch-1 call under inference mode
rebuild the placement matrices, the folded FK map and the 6x6 inertias: 0.07 to 0.15 ms more than the same call under
`torch.no_grad` (Panda FK 0.276 against 0.155 ms; one CPU thread, float32).

## Decision
We will key both memos by `(name, mode, ...)`, with `mode` the grad mode, or `"inference"` under
`torch.inference_mode`, and store inference-mode values like the others. A value made under inference mode is then
served only under inference mode; one made outside it never is. `torch.compile` still bypasses both memos.

## Example
```python { .no-run }
key = (kind, "inference" if torch.is_inference_mode_enabled() else torch.is_grad_enabled(), flags)

with torch.inference_mode():
    forward_kinematics(model, q)   # memoises the folded map under ("joint_matrices", "inference", flags)
forward_kinematics(model, q.requires_grad_()).frame_translations.sum().backward()   # a fresh value under True
```

## Consequences
A batch-1 call under inference mode costs what it costs under `torch.no_grad`. A model or state read in both modes
holds one entry per mode. The rule that an inference tensor never reaches an autograd call stands; it is now kept by
the key instead of by not storing.

## Alternatives
- Keep the bypass and document it: inference mode stays slower than `torch.no_grad` at small batch.
- Derive a value under `torch.inference_mode(False)` and `torch.no_grad` and keep it under the no-grad key, as
  `Model._plan` does, so the memo never holds an inference tensor. It is correct in every mode order tried (reads
  under inference mode, no-grad and grad in either order, model tensors that require grad, `torch.func` inside
  inference mode, a model built inside inference mode) and free for the model memo, which derives once per model.
  But every state-memo derivation then runs outside inference mode: batch-1 calls under inference mode that read
  the state memo are 8 to 28 % slower (`frame_jacobian` 0.212 against 0.255 ms on the G1, 0.161 against 0.205 ms
  on the Panda; one CPU thread, float32, median of three interleaved rounds), and keeping the inference key for the
  state memo alone leaves two rules. Deriving under `torch.inference_mode(False)` alone turns grad mode on, so a
  read inside inference mode of a state with a graph would return, and keep under the no-grad key, a value with a
  graph.

## Compatibility
No public name changes; results are unchanged.

## Enforcement
C-STATE-4; `tests/contract/test_caches.py` (`test_inference_mode_values_are_served_only_under_inference_mode`,
`test_state_memo_keeps_inference_mode_values_apart`).

## Open questions
None.
