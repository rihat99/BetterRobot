# Packaging

`pyproject.toml` is the source of truth for versions, dependencies, and
extras. This page explains how to change that surface without making optional
integrations part of every installation.

## Install surfaces

The core installation contains the tensor algorithms and URDF loader:

| Dependency | Purpose |
|---|---|
| `torch>=2.4` | tensors and automatic differentiation |
| `numpy>=2.0` | array interchange and reference calculations |
| `yourdfpy>=0.0.14` | URDF parsing |

Optional features have named extras:

| Extra | Main dependency | Purpose |
|---|---|---|
| `viewer` | `viser` | browser viewer |
| `io-mjcf` | `mujoco` | MJCF loading |
| `meshes` | `trimesh` | direct mesh APIs |
| `demos` | `robot_descriptions` | example robot assets |
| `warp` | `warp-lang` | opt-in fused FK on supported CUDA systems |
| `dev` | test, docs, lint, type, and benchmark tools | repository development |

`yourdfpy` currently brings in `trimesh` itself. That transitive install
does not make BetterRobot's mesh APIs part of the core contract.

## Adding a dependency

Put a dependency in the smallest surface that needs it:

- core, only when ordinary library use cannot work without it;
- a named extra, when the feature is optional; or
- `dev`, when only contributors need it.

Then regenerate `uv.lock`, test a core-only installation, and check that
`import better_robot` does not import an optional runtime.

Optional imports stay near the function that uses them. Importing the package
must not require a viewer, MJCF parser, example asset package, or Warp.

## Versions

The package version appears in two places that must agree:

- `project.version` in `pyproject.toml`;
- `__version__` in `src/better_robot/_version.py`.

BetterRobot is below 1.0. A minor release may still reshape the public API,
but the release notes must say what changed and what callers should use
instead.

After 1.0:

| Release | Appropriate change |
|---|---|
| major | remove or rename a stable symbol; change pose storage; change a required protocol member |
| minor | add a compatible public feature |
| patch | fix a bug or improve speed within the documented numerical contract |

The stability table in {doc}`contracts` says which surfaces become bound by
that promise.

## Removing or replacing an API

Before 1.0, remove unused surfaces cleanly. Do not leave a second
implementation, forwarding wrapper, or warning-only alias unless the release
itself explicitly promises a transition period. Record the replacement in the
changelog.

After 1.0, a compatibility period is a release decision. If one is chosen, it
needs a `DeprecationWarning`, a removal version, a focused warning test, and
a changelog entry.

## Release checklist

1. Update the version sources and the Unreleased changelog section.
2. Run the full supported test suite and strict documentation build.
3. Verify `uv lock --check`.
4. Build the wheel and source archive in a clean environment.
5. Test the wheel once with core dependencies only and once with the relevant
   extras.
6. Check third-party notices against {doc}`source_and_license`.

Public exports are checked by `tests/contract/test_public_api.py`. The test
requires the documented core to resolve and rejects duplicate exports; it
does not freeze an exact symbol count before 1.0.
