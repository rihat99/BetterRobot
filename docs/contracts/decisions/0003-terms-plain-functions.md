# 0003 Terms are plain functions

**Status:** Accepted · **Date:** 2026-10-07 (recorded; decision taken earlier)

## Context
The old `better_robot.residuals` package had one class per residual and `better_robot.tasks` wrapped them for the
old optimizer; the function-based optimizer was developed as `optim2` beside it. Each new residual needed a
subclass, state lived in instances, and the solver had to discover what a residual read.

## Decision
We will make a term a function of named variables and data, registered on a `Problem` with a name, weight,
kernel and optional `window`. Robot terms in `better_robot.terms` are factories returning such functions.
`optim` imports only torch.

## Consequences
Argument names *are* the reads: `Problem` inspects them, hands each term exactly what it names, and can tell
which terms are time-local for the banded solver. No class hierarchy to extend. The class-based packages were
removed and `optim2` renamed `optim`.

## Alternatives considered
Residual classes with `residual()`/`jacobian()` (removed); Theseus-style cost-function objects (state in
instances, heavier to compose).

## Enforcement
C-EXT-1, C-DEP-1; `tests/contract/test_protocols.py`, `tests/contract/test_import_dag.py`.
