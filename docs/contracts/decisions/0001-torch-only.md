# 0001 Torch is the only implementation

**Status:** Accepted · **Date:** 2026-10-07 (recorded; decision taken earlier)

## Context
Every computation (Lie groups, spatial algebra, kinematics, dynamics, collision, optimization, offline
rendering) is a direct PyTorch formula. Earlier versions carried NVIDIA Warp lanes next to the torch code for
forward kinematics. Measured on one CPU at batch 4096: Warp won forward-only FK (2.3 ms vs 6.6 ms Panda,
5.9 ms vs 27.8 ms G1) but lost with gradients (106 ms vs 23 ms Panda, 317 ms vs 75 ms G1). At batch 1 torch won
both. A second backend doubled every test and every contract.

## Decision
We will implement everything in PyTorch, with no second backend and no custom kernels.

## Consequences
Autograd everywhere; one dtype/device story; torch is the only hard dependency. Forward-only FK is slower than a
Warp kernel could be. No GPU measurement was taken; a large GPU win *with gradients* may reopen this.

## Alternatives considered
Warp lanes beside torch (removed); JAX (no in-place model, different ecosystem); custom CUDA kernels (lose
autograd, double the maintenance).

## Enforcement
C-DIFF-1, C-DEP-4; `tests/contract/test_import_dag.py`; `tests/autograd/`.
