# 0002 A forward-kinematics query returns an immutable state

**Status:** Accepted · **Date:** 2026-10-07 (recorded; decision taken earlier)

## Context
The old design was Pinocchio's: `Model.create_data()` made a mutable `Data`, and `forward_kinematics(model,
data, q)`, `rnea`, `crba`, `aba` wrote results into it. In PyTorch this caused stale reads (nothing tied a field
to the `q` that produced it), autograd hazards (in-place writes into cached tensors break the graph or raise
version-counter errors), and hidden coupling (which fields were valid depended on call order).

## Decision
We will return a frozen `KinematicState` from `forward_kinematics_state(model, q)`; Jacobians and
`dynamics.state` functions take that state as input. No in/out data argument anywhere.

## Consequences
States are valid by construction and never change under the caller; two queries never share storage. Each
query allocates, which is cheap next to the kernels. Caches must be explicit (C-STATE-4).

## Alternatives considered
Mutable `Data` with version tags (still hidden coupling); lazy properties on a mutable object (same autograd
hazards).

## Enforcement
C-STATE-1..3; `tests/contract/test_frozen.py`, `tests/contract/test_no_inplace.py`.
