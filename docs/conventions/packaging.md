# Packaging

> **Status:** normative.
> Pins the install surface and the SemVer scope per-symbol once we
> reach 1.0.

The principle, pre-1.0: `pip install better-robot` should give you
everything the library needs at runtime. We are still in active
development; the cost of "your script crashed because you forgot to
install a parser" is much higher than the cost of a slightly heavier
install. Once the public surface stabilises and the use-cases
differentiate, we can revisit slicing the install surface into
lean-core + opt-in extras. For now: one install line, everything works.

## 1 · Dependencies

Core (always installed):

| Package | Purpose |
|---------|---------|
| `torch` | tensors, autograd, the SE3/SO3 backend |
| `numpy` | interop and reference math |
| `rich` | error formatting |
| `yourdfpy` | URDF parsing |
| `mujoco` | MJCF parsing |
| `trimesh` | mesh loading + collision geometry |
| `viser` | browser-based viewer |
| `robot_descriptions` | example URDFs (Panda, G1, …) |

`[dev]`: `pytest`, `pytest-cov`, `pytest-xdist`, `pytest-benchmark`,
`hypothesis`, `pin` (Pinocchio reference oracle), `pyperf`, the
Sphinx docs stack (`sphinx`, `myst-parser`, `myst-nb`,
`sphinx-book-theme`, `sphinx-design`, `sphinx-autodoc2`,
`sphinx-copybutton`, `sphinxcontrib-bibtex`, `sphinx-tabs`), plus
`ruff`, `pyright`, `mypy`, `pre-commit`, `jaxtyping`.

## 2 · Adding a dependency

If you can implement the feature with what's already in the core list,
do that. If you genuinely need a new dependency, add it to
`pyproject.toml::project.dependencies` (or `[dev]` if it's only for
contributors) and bump the version per §3.

## 3 · SemVer pre / post 1.0

Until 1.0 every minor bump may break. Once 1.0 is cut:

- **Major** — removing a public symbol; changing
  `[tx, ty, tz, qx, qy, qz, qw]` ordering; changing the layered DAG;
  changing a Protocol's required members.
- **Minor** — adding a public symbol; renaming with a deprecation
  shim; tightening (never loosening) a numerical tolerance.
- **Patch** — bug fixes; perf improvements within tolerance.

The frozen `EXPECTED` set in `tests/contract/test_public_api.py`
(currently 26 symbols) is the SemVer-bound contract. The per-symbol
stability tier is in {doc}`contracts` §7.3.

## 4 · Deprecation mechanism

```python
import warnings
warnings.warn(
    "Old API is deprecated; use new API. Will be removed in vX.Y.",
    DeprecationWarning,
    stacklevel=2,
)
```

Each deprecation gets:

- A `DeprecationWarning` with the replacement and the removal version.
- An entry in `CHANGELOG.md` under the current release.
- A test that the warning fires under `pytest.warns()`.

The shim is removed in the named version; verified by
`tests/contract/test_deprecations.py`.

`BR_STRICT=1` promotes deprecation warnings to errors.

## 5 · `__version__`

```python
# src/better_robot/_version.py
__version__ = "0.2.0"
```

Imported in `__init__.py`. Single source of truth that tests read; kept
in step with `pyproject.toml::project.version`.

## 6 · Cross-references

- {doc}`contracts` §7 — SemVer scope and deprecation policy.
- {doc}`naming` §5 — the rename schedule this operationalises.
- {doc}`testing` §4.5–§4.6 — regression oracles and gate-promotion
  ladder.
- {doc}`extension` §14 — the Muscle / joint extension seam.
