# 0038 No network downloader and no parser registry in `io`

**Status:** Accepted (owner). **Date:** 2026-11-01

## Context
`io/assets.py::CachedDownloadResolver` downloads with `urllib.request.urlretrieve` inside a model loader
(`# noqa: S310`), names cache files by the URL's last segment, and has no test; the four other resolvers have no
caller outside tests. `io.register_parser` mutates a module-global registry and accepts `".sdf"`, which never matches;
`parse(format="builder")` raises a bare `KeyError`.

## Decision
We will remove `CachedDownloadResolver`, the other four asset resolvers and `register_parser`, keep mesh URIs as strings
on `Mesh` shapes resolved by `viewer.robot_meshes(geometry, *, root=)` (0016), and add a format by writing a function
that returns an `IRModel` (0025).

## Example
```python { .no-run }
geometry = br.io.load_geometry("panda.urdf", model)                     # mesh URIs stay strings
meshes = br.viewer.robot_meshes(geometry, root=Path("~/robots/panda"))   # the caller names the asset root
model = br.io.build_model(parse_sdf(Path("robot.sdf")))                  # a new format: a function, no registration
```

## Consequences
A model loader opens no network connection. Asset lookup is a path the caller provides. No global table can be filled
by import order. Cost: users of remote URDFs download them first, which they did through `robot_descriptions` anyway.

## Alternatives
Keep the downloader with a shipped stand-in and a test (1.8), and keep a validated typed registry: both are code to
maintain for no caller. A registry can be added later without breaking anyone. MuJoCo's plugin registry is
process-global, the shape being avoided.

## Compatibility
Breaks `AssetResolver`, `FilesystemResolver`, `PackageResolver`, `CompositeResolver`, `CachedDownloadResolver` and
`register_parser`; no app uses them. Adding any of them later is compatible.

## Enforcement
`tests/contract/test_public_surface.py`; ruff `S310` (the `noqa` is gone and no `urllib.request` import remains in
`src`); `tests/contract/test_extension_points.py` (0025: no module-level registry).

## Open questions
None.
