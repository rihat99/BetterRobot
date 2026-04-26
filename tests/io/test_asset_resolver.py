"""``AssetResolver`` Protocol + concrete resolvers."""

from __future__ import annotations

from pathlib import Path

import pytest

from better_robot.io.assets import (
    AssetResolver,
    CompositeResolver,
    FilesystemResolver,
    PackageResolver,
)


@pytest.fixture
def asset_tree(tmp_path: Path) -> Path:
    """Create a tiny directory with one mesh-like file under each package."""
    (tmp_path / "robot").mkdir()
    (tmp_path / "robot" / "link.stl").write_text("dummy")

    (tmp_path / "third_party_pkg" / "meshes").mkdir(parents=True)
    (tmp_path / "third_party_pkg" / "meshes" / "wheel.obj").write_text("dummy")
    return tmp_path


def test_filesystem_resolver_resolves_relative_path(asset_tree):
    resolver = FilesystemResolver(base_path=asset_tree / "robot")
    found = resolver.resolve("link.stl")
    assert found == (asset_tree / "robot" / "link.stl").resolve()


def test_filesystem_resolver_missing_raises(asset_tree):
    resolver = FilesystemResolver(base_path=asset_tree / "robot")
    with pytest.raises(FileNotFoundError):
        resolver.resolve("missing.stl")


def test_filesystem_resolver_rejects_url_scheme(asset_tree):
    resolver = FilesystemResolver(base_path=asset_tree / "robot")
    with pytest.raises(FileNotFoundError):
        resolver.resolve("package://foo/bar.stl")


def test_package_resolver_resolves_with_known_package(asset_tree):
    resolver = PackageResolver({"third_party_pkg": asset_tree / "third_party_pkg"})
    found = resolver.resolve("package://third_party_pkg/meshes/wheel.obj")
    assert found == (asset_tree / "third_party_pkg" / "meshes" / "wheel.obj").resolve()


def test_package_resolver_unknown_package_raises(asset_tree):
    resolver = PackageResolver({"known": asset_tree / "robot"})
    with pytest.raises(FileNotFoundError, match="not registered"):
        resolver.resolve("package://unknown/foo.stl")


def test_composite_resolver_first_hit_wins(asset_tree):
    composite = CompositeResolver([
        PackageResolver({"third_party_pkg": asset_tree / "third_party_pkg"}),
        FilesystemResolver(base_path=asset_tree / "robot"),
    ])
    # First resolver wins.
    out_pkg = composite.resolve("package://third_party_pkg/meshes/wheel.obj")
    assert out_pkg.name == "wheel.obj"
    # Second resolver picks up the relative-path case the first one rejects.
    out_fs = composite.resolve("link.stl")
    assert out_fs.name == "link.stl"


def test_composite_resolver_all_fail_raises(asset_tree):
    composite = CompositeResolver([
        FilesystemResolver(base_path=asset_tree / "robot"),
    ])
    with pytest.raises(FileNotFoundError):
        composite.resolve("absent.stl")


def test_protocol_runtime_check(asset_tree):
    resolver = FilesystemResolver(base_path=asset_tree)
    assert isinstance(resolver, AssetResolver)
