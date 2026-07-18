# Packaging

> **Status:** normative.
> Pins the install surface and the SemVer scope per-symbol once we
> reach 1.0.

The core install is deliberately small. It provides tensor algorithms and URDF
loading; integrations with heavyweight or specialised runtimes are explicit
extras. The source of truth is `[project.dependencies]` and
`[project.optional-dependencies]` in `pyproject.toml`. This page explains that
surface but must not invent dependencies that are absent there.

## 1 · Dependencies

Core (always installed):

| Package | Purpose |
|---------|---------|
| `torch>=2.4` | tensors, autograd, and the eager Torch compute lane |
| `numpy>=2.0` | interop and reference math |
| `yourdfpy>=0.0.14` | URDF parsing |

`yourdfpy` currently requires ``trimesh[easy]`` directly, so ``trimesh`` is
present transitively in a core environment. That does not make BetterRobot's
mesh-facing surface part of the core contract.

Optional runtime extras:

| Extra | Dependency | Purpose |
|-------|------------|---------|
| `[viewer]` | `viser` | Browser-based visualisation |
| `[io-mjcf]` | `mujoco` | MJCF loading |
| `[meshes]` | `trimesh` | Direct mesh APIs |
| `[demos]` | `robot_descriptions` | Panda, G1, and other example assets |
| `[warp]` | `warp-lang` | CUDA-validated opt-in fused FK lane |

`[dev]`: `pytest`, `pytest-cov`, `pytest-xdist`, `pytest-benchmark`,
`scipy`, `hypothesis`, `pin` (Pinocchio reference oracle), `pyperf`, the
Sphinx docs stack (`sphinx`, `myst-parser`, `myst-nb`,
`sphinx-book-theme`, `sphinx-design`, `sphinx-autodoc2`,
`sphinx-copybutton`, `sphinxcontrib-bibtex`, `sphinx-tabs`, `ghp-import`), plus
`ruff`, `pyright`, `mypy`, `pre-commit`, `jaxtyping`.

## 2 · Adding a dependency

If you can implement the feature with what is already installed, do that. Add a
dependency to the smallest surface that needs it: core only when ordinary
runtime use cannot work without it, a named extra for an optional integration,
or `[dev]` for contributor tooling. Regenerate `uv.lock`, verify a core-only
installation, and test that importing `better_robot` does not eagerly import an
optional runtime.

## 3 · SemVer pre / post 1.0

Until 1.0 every minor bump may break. Once 1.0 is cut:

- **Major** — removing a public symbol; changing
  `[tx, ty, tz, qx, qy, qz, qw]` ordering; changing the layered DAG;
  changing a Protocol's required members.
- **Minor** — adding a public symbol; renaming with a deprecation
  shim; tightening (never loosening) a numerical tolerance.
- **Patch** — bug fixes; perf improvements within tolerance.

`tests/contract/test_public_api.py` pins the required core symbols, their
resolution and docstrings, and the absence of duplicate exports; it does not
freeze a symbol count while the project is pre-1.0. Once the 1.0 surface is
frozen, removals follow the SemVer policy above. The per-symbol stability tier
is in {doc}`contracts` §7.3.

## 4 · Deprecation mechanism

```python
import warnings
warnings.warn(
    "Old API is deprecated; use new API. Will be removed in vX.Y.",
    DeprecationWarning,
    stacklevel=2,
)
```

Once a compatibility shim is intentionally introduced, it gets:

- A `DeprecationWarning` with the replacement and the removal version.
- An entry in `CHANGELOG.md` under the current release.
- A test that the warning fires under `pytest.warns()`.

The shim is removed in the named version. There is currently no generic
`test_deprecations.py` harness and no package-wide environment flag that turns
deprecation warnings into errors; a change that introduces either must add the
implementation and tests before this document claims it. Pre-1.0 removals that
land without a shim are recorded explicitly in the changelog and the removed-
surface migration ledgers.

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
- {doc}`testing` §5.4 and §7 — regression oracles and benchmark evidence.
- {doc}`extension` §2 — the joint extension seam; §14 is only a future
  actuator-design sketch.
