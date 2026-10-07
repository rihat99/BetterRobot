# 10 Docs and tests

Documentation is part of the API: a tensor shape in a docstring is a contract users rely on, and a code
snippet in the docs is a test. Tests exist to pin contracts and parity, not to count.

## C-DOC-1 Numpy-style docstrings on new and rewritten public symbols (MUST)

Every name in a package `__all__` has a docstring whose first line sits on the opening `"""`. Code written
or rewritten from now on adds numpy-style `Parameters` / `Returns` sections (functions) or a field list
(dataclasses); existing public symbols need only the docstring and the shape line (`C-TENS-3`), and are
upgraded when touched. Private helpers need a docstring only when the maths is not obvious.

```python { .no-run }
# good
def integrate(model: Model, q: Tensor, v: Tensor) -> Tensor:
    """Advance a configuration by a tangent step.

    Parameters
    ----------
    q : Tensor, shape (..., nq)
    v : Tensor, shape (..., nv)

    Returns
    -------
    Tensor, shape (..., nq)
    """
```

Enforcement: `[ruff:D]` [planned: L-7] (numpy convention, `D1xx` on public names only via per-file ignores for `_`
modules and tests); `tests/contract/test_docstrings.py`.

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

Enforcement: `[review]`.

## C-TEST-2 Type hints on every public signature (MUST)

Public functions annotate every parameter and the return type; `Tensor` from `torch`, ids as `int | str`.

Enforcement: `[ruff:ANN201]` on `src/`; `[ruff:ANN001]` [planned: L-7].

## C-DOC-4 No Sphinx roles in docstrings (MUST)

The docs are built with MkDocs, where `` :class:`Model` `` renders as literal text. Write plain backticks
(`` `Model` ``) or an mkdocstrings cross-reference (`` [Model][better_robot.Model] ``). The 50 existing roles are
ledger item L-9.

Enforcement: `tests/contract/test_docstrings.py` [planned: L-9]; `[review]`.
