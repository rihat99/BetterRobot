"""``URDFMeshMode`` routes mesh paths through the active ``AssetResolver``.

The constructor accepts an explicit ``resolver=`` override; otherwise
the parse-time resolver carried on ``Model.meta["asset_resolver"]`` is
used. When no resolver is configured the raw ``IRGeom.params["path"]``
is used unchanged (legacy behaviour).

See ``docs/concepts/parsers_and_ir.md §6`` and ``docs/concepts/viewer.md §17``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from better_robot.io.assets import AssetResolver
from better_robot.io.ir import IRGeom
from better_robot.viewer.render_modes.urdf_mesh import _load_geom, _resolve_mesh_path


class _RecordingResolver:
    """Test double: records every path it was asked to resolve."""

    def __init__(self, mapping: dict[str, str] | None = None, *, fail: bool = False) -> None:
        self.mapping = mapping or {}
        self.fail = fail
        self.calls: list[str] = []

    def resolve(self, uri: str) -> Path:
        self.calls.append(uri)
        if self.fail:
            raise FileNotFoundError(uri)
        return Path(self.mapping.get(uri, uri))


def test_resolver_protocol_runtime_check():
    """``_RecordingResolver`` satisfies the ``AssetResolver`` Protocol."""
    assert isinstance(_RecordingResolver(), AssetResolver)


def test_resolve_mesh_path_passthrough_when_resolver_is_none():
    assert _resolve_mesh_path("foo/bar.obj", None) == "foo/bar.obj"


def test_resolve_mesh_path_uses_resolver():
    r = _RecordingResolver({"package://x/m.obj": "/abs/m.obj"})
    out = _resolve_mesh_path("package://x/m.obj", r)
    assert out == "/abs/m.obj"
    assert r.calls == ["package://x/m.obj"]


def test_resolve_mesh_path_falls_back_on_resolver_failure():
    """A failing resolver must not break load — fall back to the raw path."""
    r = _RecordingResolver(fail=True)
    out = _resolve_mesh_path("package://x/m.obj", r)
    assert out == "package://x/m.obj"
    assert r.calls == ["package://x/m.obj"]


def test_resolve_mesh_path_empty_string_short_circuits():
    """Empty path means "no mesh" — return immediately, never call resolver."""
    r = _RecordingResolver()
    assert _resolve_mesh_path("", r) == ""
    assert r.calls == []


def test_load_geom_routes_mesh_kind_through_resolver(monkeypatch: pytest.MonkeyPatch):
    """``_load_geom`` for ``kind="mesh"`` consults the resolver. We swap
    in a resolver that maps ``foo`` → a path we know does not exist;
    ``trimesh.load`` will fail and ``_load_geom`` should return
    ``(None, rgba)``. The point of this test is to verify the resolver
    saw the call.
    """
    import torch

    geom = IRGeom(
        kind="mesh",
        params={"path": "foo.obj", "scale": [1.0, 1.0, 1.0]},
        origin=torch.zeros(7),
    )
    r = _RecordingResolver({"foo.obj": "/definitely/not/here.obj"})

    mesh, rgba = _load_geom(geom, resolver=r)

    assert r.calls == ["foo.obj"]
    # The mapped path doesn't exist, so trimesh.load fails and we get None.
    assert mesh is None
    # rgba defaults applied.
    assert rgba == (0.7, 0.7, 0.7, 1.0)


def test_load_geom_primitives_ignore_resolver():
    """``box``/``cylinder``/``sphere``/``capsule`` don't have file paths,
    so the resolver must not be consulted for them.
    """
    import torch

    r = _RecordingResolver()
    geom = IRGeom(
        kind="box",
        params={"size": [0.1, 0.2, 0.3]},
        origin=torch.zeros(7),
    )
    mesh, _rgba = _load_geom(geom, resolver=r)
    assert mesh is not None  # trimesh.creation.box succeeded
    assert r.calls == []
