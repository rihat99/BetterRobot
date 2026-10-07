# Docstring rewrite brief (shared by the docstring agents)

Read `context.md` in this folder first (rules: no commits, no rm, CPU only, never weaken tests).

Goal: public docstrings that read well in the mkdocstrings API reference and tell a user what a thing is
for, its tensor shapes, its contract, and one tiny example when the call is not obvious. No behaviour
change: `uv run pytest <package tests> -q` identical before and after; `uv run ruff check` + `ruff format`
clean on every file you touch.

Style (the repo's own):
- First line on the opening `"""`, one sentence, imperative or noun phrase, ends with a period. Then a blank
  line and short paragraphs. Line length 120.
- Shapes inline in double backticks: ``q (B, nq)``, ``(..., J, 3, 4)``; leading batch axes written as
  ``B...`` or ``...``; say which axes broadcast.
- Contracts stated as facts: dtype/device preserved, pose order ``[x, y, z, qx, qy, qz, qw]``, twist
  ``[linear, angular]``, frames of reference by name (``LOCAL``, ``WORLD``, ``LOCAL_WORLD_ALIGNED``).
- Cross-references as plain backticked names (``KinematicState``); mkdocstrings autolinks them. Do not use
  Sphinx roles (``:class:``, ``:func:``) — remove any you find.
- Sections, when needed, as plain bold-free paragraphs or a short bullet list; an example as a fenced
  ```python block of ≤ 8 lines that actually runs (check it with `uv run python -c` or a scratch script).
- Module docstrings: 3–8 lines: what the package/module does and what is public in it.
- Private helpers (`_name`): keep or shorten; never expand. Do not add docstrings to private functions.
- Do not touch code outside docstrings and `__all__` ordering. Do not rename anything.

What "public" means: every name in the package `__all__` and every attribute/method of those classes that
does not start with `_`. `tests/contract/test_docstrings.py` checks that public names have docstrings —
keep it passing.

Final message (≤ 25 lines): files touched, how many docstrings rewritten, examples you ran, anything in the
code that you found inconsistent with its docstring (file:line) and did NOT change.
