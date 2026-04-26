"""``AssetResolver`` Protocol + concrete resolvers.

Mesh / texture references inside URDF/MJCF files are paths or
package URIs that can only be turned into a real file once you know the
search root. The library defers that decision to a pluggable
:class:`AssetResolver`. The resolver is set on the ``IRModel.meta`` (and
forwarded onto ``Model.meta["asset_resolver"]``) so the viewer and
collision modules can find meshes after parsing without re-reading the
URDF.

See ``docs/concepts/parsers_and_ir.md §6`` and
``docs/concepts/viewer.md §17``.
"""

from __future__ import annotations

import os
import urllib.request
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, runtime_checkable


@runtime_checkable
class AssetResolver(Protocol):
    """Resolve a parser-emitted asset reference (URI / relative path) to a
    concrete absolute filesystem path.

    Implementations must raise :class:`FileNotFoundError` if the asset
    cannot be located — never return ``None`` for missing assets.
    """

    def resolve(self, uri: str) -> Path: ...


class FilesystemResolver:
    """Resolve relative paths against a fixed base directory.

    The default for URDFs is ``FilesystemResolver(base_path=Path(source).parent)``.
    """

    def __init__(self, base_path: Path | str) -> None:
        self.base_path = Path(base_path)

    def resolve(self, uri: str) -> Path:
        if "://" in uri:
            raise FileNotFoundError(
                f"FilesystemResolver does not handle URIs with schemes: {uri!r}"
            )
        candidate = (self.base_path / uri).resolve()
        if not candidate.exists():
            raise FileNotFoundError(
                f"Asset {uri!r} not found under base_path={self.base_path}"
            )
        return candidate


class PackageResolver:
    """Resolve ``package://<pkg>/<rel>`` URIs against an explicit package
    map (``{pkg_name: root_dir}``).

    ROS-style ``package://`` paths are common in URDFs; this resolver
    avoids depending on a live ``rospack`` install by accepting the map
    explicitly.
    """

    _SCHEME = "package://"

    def __init__(self, packages: dict[str, Path | str]) -> None:
        self._packages = {k: Path(v) for k, v in packages.items()}

    def resolve(self, uri: str) -> Path:
        if not uri.startswith(self._SCHEME):
            raise FileNotFoundError(
                f"PackageResolver expects 'package://' URIs, got {uri!r}"
            )
        rest = uri[len(self._SCHEME):]
        pkg, sep, rel = rest.partition("/")
        if not sep:
            raise FileNotFoundError(f"Malformed package URI: {uri!r}")
        if pkg not in self._packages:
            raise FileNotFoundError(
                f"Package {pkg!r} not registered with this PackageResolver "
                f"(known: {list(self._packages)})"
            )
        candidate = (self._packages[pkg] / rel).resolve()
        if not candidate.exists():
            raise FileNotFoundError(f"Asset not found: {candidate}")
        return candidate


class CompositeResolver:
    """Try a sequence of resolvers in order; first hit wins.

    Useful for "URDFs that mix ``package://`` and relative paths" — pair a
    :class:`PackageResolver` with a :class:`FilesystemResolver`.
    """

    def __init__(self, resolvers: list[AssetResolver]) -> None:
        self._resolvers = list(resolvers)

    def resolve(self, uri: str) -> Path:
        last_error: Exception | None = None
        for r in self._resolvers:
            try:
                return r.resolve(uri)
            except FileNotFoundError as exc:
                last_error = exc
                continue
        raise FileNotFoundError(
            f"None of {len(self._resolvers)} resolvers could find {uri!r} "
            f"(last error: {last_error})"
        )


class CachedDownloadResolver:
    """Resolve ``http(s)://`` URIs by downloading to a local cache directory.

    Files already present in ``cache_dir`` are reused — this is **not** an
    HTTP-level cache (no ETag handling). Failures raise
    :class:`FileNotFoundError`.
    """

    def __init__(self, cache_dir: Path | str) -> None:
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def resolve(self, uri: str) -> Path:
        if not uri.startswith(("http://", "https://")):
            raise FileNotFoundError(
                f"CachedDownloadResolver only handles http(s) URIs, got {uri!r}"
            )
        # Hash-free local name — last URL component is good enough for
        # the typical URDF mesh case; collisions are the caller's problem.
        local_name = uri.rsplit("/", 1)[-1] or "asset.bin"
        target = self.cache_dir / local_name
        if not target.exists():
            try:
                urllib.request.urlretrieve(uri, target)  # noqa: S310
            except OSError as exc:
                raise FileNotFoundError(
                    f"Failed to download {uri!r} to {target}: {exc}"
                ) from exc
        return target


__all__ = [
    "AssetResolver",
    "FilesystemResolver",
    "PackageResolver",
    "CompositeResolver",
    "CachedDownloadResolver",
]
