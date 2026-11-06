"""No test skips or xfails, whatever spells it: ``tests/conftest.py`` fails each one (principle 6.5, record 0033).

A test that needs what the fast and full tiers lack (a GPU) carries the ``cuda`` marker, which deselects it there; a
missing oracle, fixture or browser is an error. The hooks read the test reports, so an aliased ``pytest``, a
``unittest`` skip or a non-strict ``xfail`` fails like a plain ``pytest.skip``. These cases plant each spelling in a
scratch suite that runs under a copy of this repository's ``tests/conftest.py``.
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytest_plugins = ["pytester"]

CONFTEST = Path(__file__).resolve().parents[1] / "conftest.py"
PLANTED = """
import unittest

import pytest
import pytest as pt
from pytest import mark


def test_skip_call():
    pytest.skip("planted")


@pt.mark.skipif(True, reason="planted")
def test_skipif_aliased_pytest():
    pass


@mark.skip(reason="planted")
def test_skip_imported_mark():
    pass


def test_importorskip():
    pytest.importorskip("a_module_that_does_not_exist")


@pytest.fixture
def skipping_fixture():
    pytest.skip("planted")


def test_skip_in_fixture(skipping_fixture):
    pass


@unittest.skip("planted")
def test_unittest_skip_decorator():
    pass


def test_raise_skiptest():
    raise unittest.SkipTest("planted")


class TestCase(unittest.TestCase):
    def test_skiptest_method(self):
        self.skipTest("planted")


@pytest.mark.xfail(reason="planted")
def test_xfail_that_fails():
    raise AssertionError


@pytest.mark.xfail(reason="planted", strict=False)
def test_xfail_that_passes():
    pass


def test_xfail_call():
    pytest.xfail("planted")
"""


def test_every_skip_or_xfail_spelling_fails(pytester: pytest.Pytester) -> None:
    """All eleven planted tests fail; the three that skip in setup (two marks, a fixture) report as errors."""
    pytester.makeconftest(CONFTEST.read_text())
    pytester.makepyfile(test_planted=PLANTED)
    result = pytester.runpytest("-p", "no:cacheprovider")
    assert result.parseoutcomes() == {"failed": 8, "errors": 3}, result.outlines


def test_a_module_skipped_at_collection_fails(pytester: pytest.Pytester) -> None:
    """A module-level ``importorskip`` is a collection error, not a skipped module."""
    pytester.makeconftest(CONFTEST.read_text())
    pytester.makepyfile(test_planted='import pytest\n\npytest.importorskip("a_module_that_does_not_exist")\n')
    result = pytester.runpytest("-p", "no:cacheprovider")
    assert result.parseoutcomes() == {"errors": 1}, result.outlines
