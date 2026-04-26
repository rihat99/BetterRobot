# 17 · IO ergonomics: builder helpers, IR versioning, asset resolver

★ **Hygiene.** Three small, mostly-mechanical fixes the gpt-plan
review surfaced. None of them is a foundational decision; together
they remove user-side friction that compounds as the parser surface
grows.

## Problem

### 17.1 `kind="revolute_z"` vs `kind="revolute"` mismatch

The `ModelBuilder` documentation in
[04_PARSERS.md §6](../../design/04_PARSERS.md) shows:

```python
b.add_joint(name="shoulder", kind="revolute_z", parent="base", child="link_1")
```

The `build_model` IR-to-`Model` factory expects:

```python
IRJoint(kind="revolute", axis=(0, 0, 1), ...)
```

The two strings are documented in adjacent files. A user copy-pasting
the builder example writes `kind="revolute_z"`; the parser later
rejects it. The error message ("unknown kind 'revolute_z'") leaves
the user guessing.

This is the kind of bug that examples *define* — once a tutorial
ships with `revolute_z`, every contributor copy-pastes it.

### 17.2 No IR schema version

`IRModel` is the single contract between every parser
(URDF, MJCF, programmatic builder, future SMPL, future OpenSim) and
the `build_model` factory. We will iterate on this contract during
the v1 cycle: adding mimic-joint support, anatomical-joint metadata,
muscle attachment points, etc. Without a `schema_version` field on
`IRModel`, breaking changes are invisible to downstream
`build_model` and to any cached `.npz` IR dumps.

### 17.3 Mesh / asset paths are stringly typed

URDF and MJCF mesh paths come in three flavours:

- absolute filesystem paths;
- relative paths against the URDF's directory;
- `package://` URLs that need a ROS-style package map.

`yourdfpy` resolves the first two. `mujoco` resolves a different
subset. Generated collision spheres / capsules need their own asset
resolution. Today each parser hand-rolls its own logic. A
`Visualizer` that wants to load meshes ends up duplicating the
URDF's path logic — the user passes the model and a directory and
hopes.

## The proposal

### 17.A `ModelBuilder` kind helpers — convenience methods over a stringly typed enum

Replace the documented free-form `kind="..."` API on user-facing
helpers with explicit named methods:

```python
class ModelBuilder:
    """Programmatic IR builder.

    Prefer the named ``add_*`` helpers over ``add_joint(kind=...)``.
    The string-keyed form is kept for advanced uses (custom joint
    kinds registered via the [JointModel](../../conventions/15_EXTENSION.md)
    extension seam)."""

    # Named helpers — these are the documented API.
    def add_revolute(self, name, parent, child, axis, **kw): ...
    def add_revolute_x(self, name, parent, child, **kw):
        return self.add_revolute(name, parent, child, axis=(1, 0, 0), **kw)
    def add_revolute_y(self, name, parent, child, **kw):
        return self.add_revolute(name, parent, child, axis=(0, 1, 0), **kw)
    def add_revolute_z(self, name, parent, child, **kw):
        return self.add_revolute(name, parent, child, axis=(0, 0, 1), **kw)
    def add_prismatic(self, name, parent, child, axis, **kw): ...
    def add_prismatic_x(self, name, parent, child, **kw): ...
    def add_prismatic_y(self, name, parent, child, **kw): ...
    def add_prismatic_z(self, name, parent, child, **kw): ...
    def add_spherical(self, name, parent, child, **kw): ...
    def add_planar(self, name, parent, child, **kw): ...
    def add_free_flyer_root(self, name, child, **kw):
        """Convenience for floating-base robots — adds a free-flyer
        joint between the universe and the supplied child body."""

    # Catch-all — for custom joint kinds.
    def add_joint(self, *, name, parent, child, kind: "JointModel", **kw): ...
```

Note `add_joint(kind=...)` now takes a `JointModel` *instance*, not
a string — strings stop being valid for the `add_joint` path. The
named helpers internally know which `JointModel` to construct.

This removes the `revolute_z` vs `revolute(axis=(0,0,1))` confusion
by making the helper *be* the friendly name. Existing tests / docs
that used the string form get a one-time migration; the new examples
are unambiguous.

### 17.B `IRModel.schema_version`

```python
@dataclass(frozen=True)
class IRModel:
    schema_version: int = 1                # NEW — bump on breaking change
    name: str
    bodies: tuple[IRBody, ...]
    joints: tuple[IRJoint, ...]
    frames: tuple[IRFrame, ...]
    geoms: tuple[IRGeom, ...]
    meta: dict
```

`build_model` validates:

```python
def build_model(ir: IRModel, *, free_flyer: bool = False) -> Model:
    if ir.schema_version != IRModel.schema_version:
        raise IRSchemaVersionError(
            f"IR has schema_version={ir.schema_version}; "
            f"this build of better_robot expects "
            f"{IRModel.schema_version}. Re-parse the source asset."
        )
    ...
```

The version is a single-int counter, bumped in the same PR that
changes the IR shape, with a one-line entry in
`docs/CHANGELOG.md` describing the change. Cached IR dumps
(`.npz` artifacts in `tests/io/`) carry the version; old fixtures
re-generate when the version bumps.

### 17.C `AssetResolver` Protocol

```python
# src/better_robot/io/assets.py
from typing import Protocol
from pathlib import Path

class AssetResolver(Protocol):
    """Resolves a mesh / asset URI to a local file path.

    Parsers and visualisers should use a resolver instead of
    hand-rolling path logic. The resolver may consult a package
    map (ROS-style ``package://``), a base directory, an embedded
    asset bundle, or a downloader cache."""

    def resolve(self, uri: str, *, base_path: Path | None = None) -> Path: ...
    def exists(self, uri: str, *, base_path: Path | None = None) -> bool: ...
```

Concrete resolvers:

```python
class FilesystemResolver(AssetResolver):
    """Filesystem-only resolver. Resolves absolute paths and paths
    relative to ``base_path``. Default for URDFs loaded from local
    disk; closure-bound to the URDF's directory."""

class PackageResolver(AssetResolver):
    """ROS-style ``package://`` URL resolver. Takes a dict
    {package_name: package_root_path} and translates URLs to local
    paths. Falls back to ``FilesystemResolver`` for non-URL URIs."""

class CompositeResolver(AssetResolver):
    """Tries each child resolver in order; returns the first hit.
    Useful for URDFs that mix package URLs with local meshes."""

class CachedDownloadResolver(AssetResolver):
    """For ``robot_descriptions``-style packages that ship asset URLs;
    downloads to ``~/.cache/better_robot/assets/`` on first access."""
```

Parsers accept an optional `resolver=` kwarg:

```python
def parse_urdf(source, *, resolver: AssetResolver | None = None) -> IRModel:
    ...
```

If `resolver=None`, the parser builds a `FilesystemResolver` rooted
at the URDF's directory — preserving today's behaviour. The
`Visualizer` and any future `RobotCollision` mesh loader take the
same kwarg.

A `Model.meta["asset_resolver"]` slot stores the resolver used at
parse time, so downstream consumers (viewer, collision SDF
generation) inherit it.

### 17.D Examples and tests update

```
docs/site/tutorials/01_install_and_panda_fk.md
docs/site/guides/load_urdf.md
docs/site/guides/add_a_joint_kind.md          # renamed: was "add a custom joint kind"
examples/programmatic_panda.py                # uses add_revolute_z, no string kinds
tests/io/test_builder_helpers.py              # asserts each named helper round-trips
tests/io/test_ir_schema_version.py            # asserts version mismatch raises
tests/io/test_asset_resolver.py               # filesystem + package + composite cases
```

## Cross-references and integration

This proposal is **complementary**:

- It is a precondition for [Proposal 09 §9.D SMPL parsers](09_human_body_extension_lane.md):
  the SMPL parser uses the `AssetResolver` to find the body's mesh
  and weights without hard-coding paths.
- It cooperates with [Proposal 15 §15.A](15_packaging_extras_releases.md):
  `[urdf]` and `[mjcf]` extras gate parser availability;
  `AssetResolver` is in core because it is a Protocol with no heavy
  dependencies.
- It is not a precondition for [Proposal 02](02_backend_abstraction.md)
  — the asset / parser layer sits above the backend layer.

## Tradeoffs

| For | Against |
|-----|---------|
| The `revolute_z` / `revolute(axis=(0,0,1))` mismatch becomes impossible at the source. | Existing examples migrate; the change is mechanical but visible in diffs. |
| `IRModel.schema_version` makes IR breaking changes loud. | One more field to set. Default value covers the common case. |
| `AssetResolver` centralises path logic; the viewer no longer reimplements it. | One Protocol to learn. Mitigation: parsers default to `FilesystemResolver`, so casual users do not see it. |

## Acceptance criteria

- `ModelBuilder` exposes `add_revolute_z`, `add_prismatic_x`,
  `add_free_flyer_root`, etc. — each a one-liner that round-trips
  through `build_model` and produces the expected joint kind.
- `add_joint(kind="revolute_z")` raises a typed error pointing at
  `add_revolute_z`.
- `IRModel.schema_version` exists; `build_model(IRModel(schema_version=99, ...))`
  raises `IRSchemaVersionError`.
- `AssetResolver` Protocol and the four concrete resolvers exist.
- `parse_urdf(source, resolver=None)` defaults to the filesystem
  resolver rooted at `Path(source).parent`.
- `Visualizer` reads `model.meta["asset_resolver"]` to find meshes
  rather than re-implementing path logic.
- `examples/programmatic_panda.py` runs cleanly using only named
  helpers — no string `kind="..."` kwargs anywhere.
- `tests/io/test_builder_helpers.py`, `test_ir_schema_version.py`,
  and `test_asset_resolver.py` pass.
