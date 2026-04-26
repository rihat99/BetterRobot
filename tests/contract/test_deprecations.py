"""Stub for version-gated deprecation removal tests.

When v1.1 is cut, deprecated aliases such as ``oMi`` / ``oMf`` / ``liMi``
will be removed and assertions added here. Until then, this file just
documents the intent.

See ``docs/conventions/13_NAMING.md §4`` and
``docs/conventions/16_TESTING.md §4.6``.
"""

from __future__ import annotations


def test_deprecations_stub_passes() -> None:
    """Placeholder until v1.1 removes deprecated aliases."""
    # When the v1.1 release branch lands, replace this with assertions
    # that the deprecated symbols are gone.
    assert True
