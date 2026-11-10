# 10 Docs and tests

Documentation is part of the API: a tensor shape in a docstring is a contract users rely on, and a code
snippet in the docs is a test. Tests exist to pin contracts and parity, not to count.

## C-DOC-1 Every public object has a docstring that names its tensor shapes (MUST)

Every public object, a name in a package `__all__` and each method and property of a public class, has a
docstring. It names the shape of each tensor parameter next to the parameter, `q: (B..., nq)`, and the shape of a
tensor result (`C-TENS-3`). Numpy `Parameters` / `Returns` sections are one way to write that, not a requirement.
Private helpers need a docstring only when the maths is not obvious.

```python { .no-run }
# good
def integrate(model: Model, q: Tensor, v: Tensor) -> Tensor:
    """Step ``q: (B..., nq)`` along ``v: (B..., nv)`` on the configuration manifold; ``(B..., nq)``."""
# bad: no shape for q, v or the result
def integrate(model: Model, q: Tensor, v: Tensor) -> Tensor:
    """Advance a configuration by a tangent step."""
```

Enforcement: `tests/contract/test_ruff_ratchets.py` (ruff's `D1` rules on `src/`: every public name has a
docstring); `tests/contract/test_docstrings.py` (the docstring and its shapes, over the whole `__all__` tree).

## C-DOC-2 Every documentation snippet executes (MUST)

Python blocks in `docs/` and `README.md` run under `tests/docs/` (markdown code-block runner) unless the fence
carries the `{ .no-run }` attribute (```` ```python { .no-run } ````). A snippet that prints is followed by an
```` ```output ```` fence holding the real output.

Enforcement: `tests/docs/test_snippets.py` (runs every snippet in page order and checks the output fences).

## C-DOC-3 Every contract rule has an enforcer (MUST)

Each rule in `docs/contracts/` names a `ruff` code, a test file, or `[review]`. The enforcement matrix in
`00-index` lists them; a rule whose enforcer file is missing is a ledger item, not a waiver.

Enforcement: `tests/contract/test_contract_matrix.py`.

## C-TEST-1 Tests pin contracts and parity; no trivial tests (SHOULD)

A test is justified when it fixes a bug, guards an invariant (`tests/contract/`), checks Pinocchio parity
(`tests/parity/`), or is the cheapest verification of a change. Never weaken, skip or `xfail` a contract or
parity test to land a change; fix the code or open an ADR.

Enforcement: `tests/contract/test_no_skips.py` (no skip or xfail, whatever spells it); `[review]`.

## C-TEST-2 Type hints on every public signature (MUST)

Public functions annotate every parameter and the return type; `Tensor` from `torch`, ids as `int | str`.

Enforcement: `[ruff:ANN201]` on `src/`; `tests/contract/test_ruff_ratchets.py` (ruff's `ANN001`); `tests/contract/test_pyright_ratchet.py`.

## C-DOC-4 No Sphinx roles in docstrings (MUST)

The docs are built with MkDocs, where `` :class:`Model` `` renders as literal text. Write plain backticks
(`` `Model` ``) or an mkdocstrings cross-reference (`` [Model][better_robot.Model] ``).

Enforcement: `tests/contract/test_docstrings.py` (no role in any file under `src/`).
