"""No test skips or xfails, whatever spells it: the plugin ``better_robot.testing.no_skips`` fails each one
(principle 6.5, record 0033).

A test that needs what the fast and full tiers lack (a GPU) carries the ``cuda`` marker, which deselects it there; a
missing oracle, fixture or browser is an error. The hooks read the test reports, so an aliased ``pytest``, a
``unittest`` skip or a non-strict ``xfail`` fails like a plain ``pytest.skip``. These cases plant each spelling in a
scratch suite that loads only the plugin, as this repository's ``tests/conftest.py`` and BetterHuman's do. Two ways
to drop a test from every tier without a skip are refused too: ``collect_ignore`` in a conftest, and the ``bench``
marker (deselected in every tier) outside ``tests/bench/``.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

pytest_plugins = ["pytester"]

CONFTEST = 'pytest_plugins = ["better_robot.testing.no_skips"]\n'
REPO = Path(__file__).resolve().parents[2]
COLLECT_IGNORE = {"collect_ignore", "collect_ignore_glob"}
SKIPS = """
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
"""
XFAILS = """
import pytest


@pytest.mark.xfail(reason="planted")
def test_xfail_that_fails():
    raise AssertionError


@pytest.mark.xfail(reason="planted", strict=True)
def test_strict_xfail_that_fails():
    raise AssertionError


@pytest.mark.xfail(reason="planted", strict=False)
def test_xfail_that_passes():
    pass


def test_xfail_call():
    pytest.xfail("planted")
"""


@pytest.mark.parametrize(
    ("planted", "outcomes"),
    [(SKIPS, {"failed": 5, "errors": 3}), (XFAILS, {"failed": 4})],
    ids=["skips", "xfails"],
)
def test_every_skip_or_xfail_spelling_fails(pytester: pytest.Pytester, planted: str, outcomes: dict[str, int]) -> None:
    """Every planted test fails and pytest exits nonzero; the three that skip in setup (two marks, a fixture) report
    as errors. The xfails run alone, as pytest's exit code ignores a failed report that still carries ``wasxfail``
    (a strict xpass fails without the plugin, so it is not planted)."""
    pytester.makeconftest(CONFTEST)
    pytester.makepyfile(test_planted=planted)
    result = pytester.runpytest("-p", "no:cacheprovider")
    assert result.parseoutcomes() == outcomes, result.outlines
    assert result.ret == pytest.ExitCode.TESTS_FAILED, result.outlines


def test_a_module_skipped_at_collection_fails(pytester: pytest.Pytester) -> None:
    """A module-level ``importorskip`` is a collection error, not a skipped module."""
    pytester.makeconftest(CONFTEST)
    pytester.makepyfile(test_planted='import pytest\n\npytest.importorskip("a_module_that_does_not_exist")\n')
    result = pytester.runpytest("-p", "no:cacheprovider")
    assert result.parseoutcomes() == {"errors": 1}, result.outlines
    assert result.ret == pytest.ExitCode.INTERRUPTED, result.outlines


def _tier_drops(path: str, source: str) -> list[str]:
    """``collect_ignore`` named in a conftest, and the ``bench`` marker outside ``tests/bench/``, in ``source``."""
    found = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Name) and node.id in COLLECT_IGNORE and path.endswith("conftest.py"):
            found.append(f"{path}: {node.id}")
        if isinstance(node, ast.Attribute) and node.attr == "bench" and not path.startswith("tests/bench/"):
            owner = node.value  # ``pytest.mark`` or a ``mark`` imported from pytest
            if (isinstance(owner, ast.Attribute) and owner.attr == "mark") or (
                isinstance(owner, ast.Name) and owner.id == "mark"
            ):
                found.append(f"{path}: mark.bench")
    return found


def test_no_test_is_dropped_from_every_tier() -> None:
    """Every Python file under ``tests/``, committed or not, as pytest collects it."""
    files = sorted((REPO / "tests").rglob("*.py"))
    assert [drop for file in files for drop in _tier_drops(file.relative_to(REPO).as_posix(), file.read_text())] == []


@pytest.mark.parametrize(
    ("path", "source"),
    [
        ("tests/conftest.py", 'collect_ignore = ["test_problem.py"]\n'),
        ("tests/parity/pinocchio/conftest.py", 'collect_ignore_glob = ["*_gpu.py"]\n'),
        ("tests/optim/test_problem.py", "import pytest\n\npytestmark = pytest.mark.bench\n"),
        ("tests/lie/test_se3.py", "from pytest import mark\n\n@mark.bench\ndef test_compose(): ...\n"),
    ],
)
def test_a_planted_tier_drop_is_found(path: str, source: str) -> None:
    """The review's plants: both dropped tests from every tier and passed every check."""
    assert len(_tier_drops(path, source)) == 1
