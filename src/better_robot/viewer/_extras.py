"""``needs_extra``: an optional package that is not installed raises ``OptionalDependencyError`` with its install line."""

from __future__ import annotations

import contextlib
from collections.abc import Iterator

from ..exceptions import OptionalDependencyError


@contextlib.contextmanager
def needs_extra(extra: str) -> Iterator[None]:
    """Turn a ``ModuleNotFoundError`` raised inside the block into ``OptionalDependencyError`` naming the missing
    package and ``uv sync --extra <extra>``, the extra of ``pyproject.toml`` that installs it."""
    try:
        yield
    except ModuleNotFoundError as error:
        raise OptionalDependencyError(error.name or extra, install=f"uv sync --extra {extra}") from error
