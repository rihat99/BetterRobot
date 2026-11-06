# 0028 States broadcast `B...` against model values `I...`; `optim` keeps `(B, *event)`

**Status:** Accepted (owner). **Date:** 2026-11-01

## Context
Kinematics takes `(B..., event)`; `optim` takes one batch axis `(B, *event)`, for a documented reason. BetterHuman
bakes per identity, so model values carry leading axes (`joint_placements (I..., nj, 7)`, `body_inertias
(I..., nb, 10)`, `frame_placements (I..., nf, 7)`), and forward kinematics, Jacobians and `inverse_dynamics`
broadcast `q (B...)` against them. `Model`'s field table marks `(I..., ...)`, but a shape standard that says only
`(B...)` leaves the rule unwritten.

## Decision
We will let model tensor fields carry leading value axes `I...`, define every state's batch as `broadcast(B..., I...)`,
raise `ShapeError` for shapes that do not broadcast, give limits, `q_neutral` and `gravity` no batch, and keep
`(B, *event)` in `optim` as the one documented exception.

## Example
```python { .no-run }
human = bh.load("smplx"); identity = human.bake(betas=betas)       # betas (2, 10): identity batch I = (2,)
model = identity.model                                              # joint_placements (2, nj, 7)
state = br.forward_kinematics(model, q)                             # q (3, 1, nq): B = (3, 1)
state.joint_poses.shape                                             # (3, 2, nj, 7) = broadcast((3, 1), (2,)) + event
br.forward_kinematics(model, q_bad)                                 # q (3, 5, nq): raises ShapeError
```

## Consequences
Identity batches (many subjects) and pose batches (many frames) compose without loops. Every BetterRobot kernel that
reads placements or inertias must broadcast them; this is a cost on BetterRobot for BetterHuman's sake.

## Alternatives
Forbid batched model values: BetterHuman loops over identities and identity batches end. Brax's `System` and pyroki's
`Robot` are single-instance pytrees batched by `vmap` from outside; here the batch axes are explicit in the shapes so
that torch broadcasting does the work.

## Compatibility
Adds the rule; no existing call breaks. Removing it later breaks every identity batch. Reversible: no.

## Enforcement
The coupling tests (0026, `tests/contract/test_coupling_standard.py`) run `I = (2,)` against `B = (3, 1)` and assert
the `(3, 2, ...)` result and the `ShapeError`; `tests/contract/test_dtype_device.py` and the shape docstring test
(`tests/contract/test_docstrings.py`) document the axes.

## Open questions
None.
