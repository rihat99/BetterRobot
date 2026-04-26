# 15 · Packaging, extras, and the release discipline

★ **Hygiene.** Pyproject extras, SemVer policy, deprecation
mechanism. Operational form of
[17 §7](../../conventions/17_CONTRACTS.md).

## Problem

`pyproject.toml` today is minimal. There are no optional extras,
no documented release process, no tooling for deprecation. As the
proposals in this folder land — Warp backend, docs site, human-body
extras, Pinocchio cross-check — the dependency footprint grows.
Without extras, every install pulls every dependency, including
viser (heavy), warp (CUDA-only), mujoco (large), and the docs
toolchain.

The fix is mostly mechanical. The point of writing it down is to
make the discipline survive contributor turnover.

## The proposal

### 15.A Extras taxonomy

The core dependency list is **lean**. Anything that is only needed
for parsing a specific format, examples, visualisation, or
cross-checking against another library lives in an extra.

```toml
# pyproject.toml
[project]
name        = "better-robot"
version     = "0.1.0"          # pre-1.0 — semver applies only after 1.0
description = "PyTorch-native robotics library: kinematics, dynamics, optimisation."
requires-python = ">=3.10"

dependencies = [
    "torch>=2.4",
    "numpy>=1.26",
    "pypose>=0.6,<0.8",        # removed in v1.2 — see Proposal 03
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

# Visualisation, mesh loading, geometry.
viewer = [
    "viser>=0.1",              # 3D viewer.
    "trimesh>=4.0",            # mesh loading.
]
geometry = [
    "trimesh>=4.0",            # convex meshes / collision SDFs.
]

# Examples and tutorials need the standard robot zoo. Not pulled
# in by `pip install better-robot`.
examples = [
    "yourdfpy>=0.0.55",        # examples ship URDFs.
    "robot_descriptions>=1.7", # the standard robot zoo.
]

# Backend acceleration.
warp = [
    "warp-lang>=1.0",          # NVIDIA Warp; CUDA-only.
]

# Domain extensions.
human = [
    "better-robot-human>=0.1", # placeholder until package is published;
                                # see Proposal 09 §9.E for the phasing.
]

# Build / test / develop.
docs = [
    "sphinx>=7",
    "myst-parser>=4",
    "myst-nb>=1.1",
    "sphinx-design>=0.6",
    "furo>=2024.5",
    "autodoc2>=0.5",
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
    "better-robot[urdf,mjcf,viewer,geometry,warp,human,docs,bench]",
]
```

**What changed from an earlier draft**, per the gpt-plan review:

- `yourdfpy` was a core dependency. Moved to `[urdf]`. Calling
  `better_robot.load("foo.urdf")` raises a clear
  `BackendNotAvailableError` if the user has not installed
  `better-robot[urdf]`. URDF is the most common format, so the
  README documents `pip install better-robot[urdf]` as the
  conventional install line for a robotics user.
- `robot_descriptions` was a core dependency "for examples; small".
  It is *only* used by examples and tests, not by core algorithms.
  Moved to `[examples]`.
- `pin` (Pinocchio) is used by cross-check tests only. Moved to
  `[test]`. Tests `skipif` when it is unavailable.
- `[urdf]` and `[mjcf]` are split because the two parsers have
  different transitive deps and the user often only wants one.
- `[geometry]` exists separately from `[viewer]` because some users
  want collision SDFs without `viser`.
- `[dev]` is now a meta-extra that pulls in everything a
  contributor needs.

### 15.B Imports respect extras

The extras are real only if `import better_robot` doesn't pull them.
The discipline:

| Module | Imports |
|--------|---------|
| `better_robot.lie`, `better_robot.spatial`, `better_robot.data_model`, `better_robot.kinematics`, `better_robot.dynamics`, `better_robot.residuals`, `better_robot.costs`, `better_robot.optim`, `better_robot.tasks` | Only `torch`, `numpy`, `pypose` (until [Proposal 03](03_replace_pypose.md)). |
| `better_robot.io.parsers.urdf` | `yourdfpy` (lazy import inside the parser body). Raises `BackendNotAvailableError` if missing — install `better-robot[urdf]`. |
| `better_robot.io.parsers.mjcf` | `mujoco` (lazy import inside the parser body). Raises if missing — install `better-robot[mjcf]`. |
| `better_robot.viewer` | `viser`, `trimesh` (lazy import inside `Visualizer.__init__`). Raises if missing — install `better-robot[viewer]`. |
| `better_robot.backends.warp` | `warp` (lazy import). Raises `BackendNotAvailableError`. |
| `examples/*.py` | May import `robot_descriptions` and `yourdfpy`; these are documented in `examples/README.md` as needing `[examples]`. |

The contract test
`tests/contract/test_optional_imports.py`:

```python
def test_core_import_does_not_pull_optional_deps():
    """Verify import better_robot doesn't drag in heavy / format-specific deps."""
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

### 15.C SemVer pre/post 1.0

Until 1.0, every minor bump may break. Once 1.0 is cut:

- **Major** — removing a public symbol; changing
  `[tx, ty, tz, qx, qy, qz, qw]` ordering; changing the layered DAG;
  changing a Protocol's required members.
- **Minor** — adding a public symbol; renaming with a deprecation
  shim; tightening (never loosening) a numerical tolerance.
- **Patch** — bug fixes; perf improvements within tolerance.

Already in [17 §7.1](../../conventions/17_CONTRACTS.md). The
operational addition: a CI job that diffs `__all__` and the
docstrings of public symbols against the previous tag and fails the
build if a major-bump-only change appears in a minor or patch
branch.

```python
# tests/contract/test_semver_compat.py
"""On a non-major-bump branch, fail if anything in __all__ has been
removed or had its signature changed in a backwards-incompatible
way relative to the latest released tag."""
```

This test runs only on release branches (`release/*` or `main`)
because diffing requires `git fetch`. It's a tripwire, not a daily
check.

### 15.D Deprecation mechanism

Already documented in [17 §7.2](../../conventions/17_CONTRACTS.md):

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

- A `DeprecationWarning` with the replacement and the removal
  version.
- An entry in `CHANGELOG.md` under the current release.
- An entry in `docs/site/reference/changelog.md` (auto-generated
  from `CHANGELOG.md`).
- A test that ensures the warning fires under `pytest.warns()`.

The shim itself is removed in the named version, which is verified
by a test:

```python
# tests/contract/test_deprecations.py
import importlib

def test_data_oMi_removed_in_v1_1():
    if better_robot.__version__ >= "1.1":
        from better_robot.data_model.data import Data
        d = Data(_model_id=0, q=torch.zeros(7))
        with pytest.raises(AttributeError):
            d.oMi
```

### 15.E Release process

A `RELEASING.md` at repo root:

1. Update `CHANGELOG.md` (Keep-a-Changelog format).
2. Bump `pyproject.toml` version.
3. Update `__init__.py` `__version__` constant.
4. `git tag vX.Y.Z`.
5. `uv build && uv publish` (or the GitHub Actions release workflow
   does this automatically on tag push).
6. Sphinx site rebuilds with the new version.

Steps 3–5 are scripted; the human responsibility is the changelog.

### 15.F `__version__`

```python
# src/better_robot/_version.py
__version__ = "0.1.0"
```

Imported in `__init__.py`. Source of truth that tests read.

### 15.G Wheels and source distributions

`uv build` produces both. Pure Python — no C extensions in v1
(all our heavy lifting is via PyTorch / Warp kernels), so wheels are
universal (`py3-none-any`). PyPI publish via the GitHub Actions
release workflow + `pypa/gh-action-pypi-publish`.

### 15.H Bug-bash policy

For pre-1.0:

- Yanked release if a regression slips through to PyPI; document in
  CHANGELOG.
- Patch release for any cross-check failure (Pinocchio/Drake) on a
  released version.

For post-1.0:

- Same plus deprecation across a full minor cycle for any rename or
  signature change.

## Out of scope

- Conda-forge packaging: deferred until 1.0.
- ROS integration: never; per
  [00 §Non-goals](../../design/00_VISION.md).
- C-extension modules: not planned. Warp is the perf path.

## Tradeoffs

| For | Against |
|-----|---------|
| `pip install better-robot` is small; `[viewer]` and `[warp]` are opt-in. | Users must pick the right extras, but the README documents this. |
| SemVer enforcement via tests means a removal cannot land in a minor cycle by accident. | The semver-compat test only runs on release branches. Mitigation: it's a strong tripwire; PRs that intend to break compat must explicitly target a major-bump branch. |
| Deprecation discipline preserves user code across renames. | Adds shim code temporarily. Mitigation: shims are tracked in [13_NAMING.md §5](../../conventions/13_NAMING.md) and removed on schedule. |

## Acceptance criteria

- `pyproject.toml` declares the extras above.
- `pip install better-robot` does not install `warp`, `viser`,
  `mujoco`, `sphinx`, `yourdfpy`, `robot_descriptions`, `pinocchio`,
  or any human extras.
- `tests/contract/test_optional_imports.py` passes.
- A user who runs `import better_robot; better_robot.load("foo.urdf")`
  without `[urdf]` installed gets a `BackendNotAvailableError` whose
  message names `pip install better-robot[urdf]`.
- `RELEASING.md` exists at repo root and documents the process.
- `__version__` is in `src/better_robot/_version.py` and re-exported
  from `__init__.py`.

## Cross-references

- [17 §7](../../conventions/17_CONTRACTS.md) — SemVer scope and
  deprecation policy.
- [13 §5](../../conventions/13_NAMING.md) — the rename schedule that
  this proposal operationalises.
- [Proposal 11](11_quality_gates_ci.md) — wires the
  `test_optional_imports.py` and semver tests into CI.
- [Proposal 09](09_human_body_extension_lane.md) — `[human]` extra
  hosts the SMPL-family loaders.
- [Proposal 10](10_user_docs_diataxis.md) — `[docs]` extra hosts
  the Sphinx + MyST tooling.
