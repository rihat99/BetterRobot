# Packaging, Extras, Releases

> **Status:** normative. Operational form of
> [17_CONTRACTS.md §7](17_CONTRACTS.md). Pins the install surface, the
> SemVer scope per-symbol, the deprecation mechanism, and the release
> checklist. The principle: `pip install better-robot` gives a small
> core; everything heavy or format-specific is opt-in.

## 1 · Extras taxonomy

The core dependency list is **lean**. Anything that is only needed for
parsing a specific format, examples, visualisation, or cross-checking
against another library lives in an extra.

```toml
# pyproject.toml
[project]
name        = "better-robot"
version     = "0.1.0"          # pre-1.0 — semver applies after 1.0
description = "PyTorch-native robotics library: kinematics, dynamics, optimisation."
requires-python = ">=3.10"

dependencies = [
    "torch>=2.4",
    "numpy>=1.26",
    "pypose>=0.6,<0.8",        # removed in v1.2 — see P10 (L-D) of UPDATE_PHASES.md
    "rich>=13",                # nice repr / errors
]

[project.optional-dependencies]
# Format-specific parsers.
urdf = [
    "yourdfpy>=0.0.55",        # URDF parsing — was core, now opt-in.
]
mjcf = [
    "mujoco>=3.1",             # MJCF parser.
]

# Visualisation and geometry.
viewer = [
    "viser>=0.1",              # 3D viewer.
    "trimesh>=4.0",            # mesh loading.
]
geometry = [
    "trimesh>=4.0",            # convex meshes / collision SDFs.
]

# Examples and tutorials need the standard robot zoo.
examples = [
    "yourdfpy>=0.0.55",        # examples ship URDFs.
    "robot_descriptions>=1.7", # the standard robot zoo.
]

# Backend acceleration.
warp = [
    "warp-lang>=1.0",          # NVIDIA Warp; CUDA-only.
]

# Build / test / develop.
docs = [
    "sphinx>=7",
    "myst-parser>=4",
    "myst-nb>=1.1",
    "sphinx-design>=0.6",
    "furo>=2024.5",
    "sphinx-autodoc2>=0.5",   # PyPI name; import is `autodoc2`
]
bench = [
    "pytest-benchmark>=4",
    "pyperf>=2.7",
]
test = [
    "pytest>=8",
    "pytest-cov>=5",
    "pytest-xdist>=3",
    "hypothesis>=6.100",
    "pin>=2.7",                # Pinocchio for cross-check tests; opt-in only.
]
dev = [
    "better-robot[urdf,mjcf,viewer,geometry,examples,test,docs,bench]",
    "mypy>=1.10",
    "pyright>=1.1.380",
    "ruff>=0.6",
    "pre-commit>=3.7",
    "jaxtyping>=0.2.30",
]
all = [
    "better-robot[urdf,mjcf,viewer,geometry,warp,docs,bench]",
]
# Note: no [human] extra in Phase 1 — see §1.1 below.
```

### 1.1 The `[human]` extra is phased

`better_robot_human` is not yet on PyPI. **Phase 1** (now) — no
`[human]` extra; a placeholder test in `tests/io/test_smpl_like_skeleton.py`
demonstrates the SMPL-skeleton topology can be built via the programmatic
builder (no `chumpy`, no SMPL data). **Phase 2** (when the package is
published) — add the extra:

```toml
human = [
    "better_robot_human>=0.1.0",
    "chumpy>=0.78",            # only for legacy SMPL .pkl
]
```

Until then, the seam is documented in
[15_EXTENSION.md §15](15_EXTENSION.md) but no real dependency is
declared.

### 1.2 What lives in core (and what does not)

Core (`pip install better-robot`) is small enough that:

- `import better_robot` does not pull `yourdfpy`, `mujoco`, `viser`,
  `trimesh`, `warp`, `sphinx`, `robot_descriptions`, or `pinocchio`.
- The contract test `tests/contract/test_optional_imports.py` enforces
  this on every PR.

```python
def test_core_import_does_not_pull_optional_deps():
    import sys
    forbidden = (
        "warp", "mujoco", "viser", "trimesh", "sphinx",
        "yourdfpy", "robot_descriptions", "pinocchio",
    )
    for name in list(sys.modules):
        if name == "better_robot" or name.startswith("better_robot."):
            del sys.modules[name]
    for name in forbidden:
        sys.modules.pop(name, None)
    import better_robot  # noqa: F401
    for name in forbidden:
        assert name not in sys.modules, f"`import better_robot` pulled in {name!r}"
```

The README documents `pip install better-robot[urdf]` as the conventional
install line for a robotics user (URDF is the most common format).

## 2 · Imports respect extras

The discipline at module level:

| Module | Imports |
|--------|---------|
| `better_robot.lie`, `.spatial`, `.data_model`, `.kinematics`, `.dynamics`, `.residuals`, `.costs`, `.optim`, `.tasks` | Only `torch`, `numpy`, `pypose` (until P10/L-D of [UPDATE_PHASES](../UPDATE_PHASES.md)). |
| `better_robot.io.parsers.urdf` | `yourdfpy` (lazy import inside the parser body). Raises `BackendNotAvailableError` if missing — install `better-robot[urdf]`. |
| `better_robot.io.parsers.mjcf` | `mujoco` (lazy import). Raises if missing — install `better-robot[mjcf]`. |
| `better_robot.viewer` | `viser`, `trimesh` (lazy import inside `Visualizer.__init__`). Raises if missing — install `better-robot[viewer]`. |
| `better_robot.backends.warp` | `warp` (lazy import). Raises `BackendNotAvailableError`. |
| `examples/*.py` | May import `robot_descriptions` and `yourdfpy`; documented in `examples/README.md` as needing `[examples]`. |

## 3 · SemVer pre/post 1.0

Until 1.0 every minor bump may break. Once 1.0 is cut:

- **Major** — removing a public symbol; changing
  `[tx, ty, tz, qx, qy, qz, qw]` ordering; changing the layered DAG;
  changing a Protocol's required members.
- **Minor** — adding a public symbol; renaming with a deprecation
  shim; tightening (never loosening) a numerical tolerance.
- **Patch** — bug fixes; perf improvements within tolerance.

The frozen `EXPECTED` set in `tests/contract/test_public_api.py`
(currently 26 symbols) is the SemVer-bound contract; see
[01_ARCHITECTURE.md §Public API contract](../design/01_ARCHITECTURE.md).
The per-symbol stability tier is in
[17_CONTRACTS.md §7.3](17_CONTRACTS.md).

A CI tripwire `tests/contract/test_semver_compat.py` runs on
`release/*` and `main` branches:

> On a non-major-bump branch, fail if anything in `__all__` has been
> removed or had its signature changed in a backwards-incompatible way
> relative to the latest released tag.

## 4 · Deprecation mechanism

```python
import warnings
warnings.warn(
    "Data.oMi is deprecated; use Data.joint_pose_world. "
    "Will be removed in v1.1.",
    DeprecationWarning,
    stacklevel=2,
)
```

Each deprecation gets:

- A `DeprecationWarning` with the replacement and the removal version.
- An entry in `CHANGELOG.md` under the current release.
- An entry in `docs/site/reference/changelog.md` (auto-generated).
- A test that the warning fires under `pytest.warns()`.

The shim is removed in the named version; verified by
`tests/contract/test_deprecations.py`:

```python
import importlib

def test_data_oMi_removed_in_v1_1():
    if better_robot.__version__ >= "1.1":
        from better_robot.data_model.data import Data
        d = Data(_model_id=0, q=torch.zeros(7))
        with pytest.raises(AttributeError):
            d.oMi
```

`BR_STRICT=1` promotes deprecation warnings to errors; CI's contract
job sets it.

## 5 · Release process

A `RELEASING.md` at repo root documents the human steps:

1. Update `CHANGELOG.md` (Keep-a-Changelog format).
2. Bump `pyproject.toml` version.
3. Update `__version__` in `src/better_robot/_version.py`.
4. `git tag vX.Y.Z`.
5. `uv build && uv publish` (or the GitHub Actions release workflow runs
   automatically on tag push).
6. Sphinx site rebuilds with the new version.

Steps 3–5 are scripted; the human responsibility is the changelog.

## 6 · `__version__`

```python
# src/better_robot/_version.py
__version__ = "0.1.0"
```

Imported in `__init__.py`. Source of truth that tests read.

## 7 · Wheels and source distributions

`uv build` produces both. Pure Python — no C extensions in v1
(all heavy lifting via PyTorch / Warp kernels), so wheels are
universal (`py3-none-any`). PyPI publish via the GitHub Actions release
workflow + `pypa/gh-action-pypi-publish`.

## 8 · Bug-bash policy

Pre-1.0:
- Yanked release if a regression slips through to PyPI; document in
  CHANGELOG.
- Patch release for any cross-check failure (Pinocchio/Drake) on a
  released version.

Post-1.0:
- Same plus deprecation across a full minor cycle for any rename or
  signature change.

## 9 · Out of scope

- **Conda-forge packaging** — deferred until 1.0 stabilises on PyPI.
- **ROS integration** — never; per
  [00 §Non-goals](../design/00_VISION.md).
- **C-extension modules** — not planned. Warp is the perf path; bridging
  via `torch.autograd.Function` keeps the user-facing surface
  `torch.Tensor`-only.

## 10 · Acceptance criteria

- `pyproject.toml` declares the extras above.
- `pip install better-robot` does not install `warp`, `viser`, `mujoco`,
  `sphinx`, `yourdfpy`, `robot_descriptions`, `pinocchio`, or any human
  extras.
- `tests/contract/test_optional_imports.py` passes.
- A user who runs `import better_robot; better_robot.load("foo.urdf")`
  without `[urdf]` installed gets a `BackendNotAvailableError` whose
  message names `pip install better-robot[urdf]`.
- `RELEASING.md` exists at repo root and documents the process.
- `__version__` is in `src/better_robot/_version.py` and re-exported
  from `__init__.py`.
- `BR_STRICT=1` is set in the CI contract job; a PR using a deprecated
  alias fails.

## 11 · Cross-references

- [17 §7](17_CONTRACTS.md) — SemVer scope and deprecation policy.
- [13 §5](13_NAMING.md) — the rename schedule this operationalises.
- [14 §4.2](14_PERFORMANCE.md) — bench gate ladder.
- [16 §4.5–§4.6](16_TESTING.md) — regression oracles + gate-promotion ladder.
- [15 §15](15_EXTENSION.md) — the `[human]` Muscle/joint seam.
- [../UPDATE_PHASES.md](../UPDATE_PHASES.md) — implementation phases that
  produce the artefacts in this doc.
