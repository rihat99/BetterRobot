"""One switch guards every check that reads tensor values (decision record 0021, principle 6.8).

``better_robot._internal_checks`` reads ``BETTER_ROBOT_INTERNAL_CHECKS`` once, at import, and no other module reads
it. A value-reading check is an ``if`` that raises and whose test reads tensor values (``bool``, ``float`` or ``int``
of an expression, ``.item()``, ``.any()``, ``.all()``, ``.tolist()``, ``torch.allclose``), or a
``torch._assert_async``. In ``src`` each one is guarded: it sits under ``if _internal_checks.ENABLED:`` or under the
caller's opt-in ``if check_<what>:``, or inside a ``_check_<what>`` helper whose every call is so guarded. Every call
of an ``_internal_checks`` function sits under ``if _internal_checks.ENABLED:``.

Permanent exceptions to the value-check rule (calls of ``_internal_checks`` are guarded everywhere):
``_internal_checks.py`` itself (its functions are the guarded invariants), ``io`` (it checks a robot description's
values once, at load, where it parses them), and numpy reads (no device sync). The unguarded
checks that exist today sit in ``tests/contract/ratchets/value_checks.json`` (``path::function`` to the count of checks
and the reason); it only shrinks.
"""

from __future__ import annotations

import ast
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from better_robot import _internal_checks
from better_robot._internal_checks import check_unit_quaternions

from ._ratchet_sites import compare

SRC = Path(__file__).resolve().parents[2] / "src" / "better_robot"
RATCHET = Path(__file__).parent / "ratchets" / "value_checks.json"
EXEMPT = {
    "_internal_checks.py": "its functions are the invariants the switch guards",
    "io": "checks a robot description's values once, at load, where it parses them",
}
VALUE_METHODS = {"item", "any", "all", "tolist", "allclose", "_assert_async"}


def _reads_values(node: ast.AST) -> bool:
    for sub in ast.walk(node):
        if not isinstance(sub, ast.Call):
            continue
        function = sub.func
        if isinstance(function, ast.Name) and function.id in {"bool", "float", "int"}:
            if sub.args and not isinstance(sub.args[0], ast.Constant):
                return True
        if isinstance(function, ast.Attribute) and function.attr in VALUE_METHODS:
            if not (isinstance(function.value, ast.Name) and function.value.id in {"np", "numpy"}):
                return True
    return False


def _is_guard(test: ast.expr) -> bool:
    names = [
        sub.id if isinstance(sub, ast.Name) else sub.attr
        for sub in ast.walk(test)
        if isinstance(sub, (ast.Name, ast.Attribute))
    ]
    return any(name == "ENABLED" or name.startswith("check_") for name in names)


def _is_value_check(node: ast.AST) -> bool:
    if isinstance(node, ast.If):
        return _reads_values(node.test) and any(
            isinstance(sub, ast.Raise) for item in node.body for sub in ast.walk(item)
        )
    return (
        isinstance(node, ast.Expr)
        and isinstance(node.value, ast.Call)
        and _reads_values(node.value)
        and (isinstance(node.value.func, ast.Attribute) and node.value.func.attr == "_assert_async")
    )


def _walk(node: ast.AST, scope: list[str], guarded: bool, found: dict[str, list[tuple[str, int, bool, str]]]) -> None:
    """Collect value checks (``"check"``) and calls (``"call:<name>"``) with their function and guard state."""
    for child in ast.iter_child_nodes(node):
        if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            _walk(child, [*scope, child.name], False if isinstance(child, ast.ClassDef) else guarded, found)
            continue
        if _is_value_check(child):
            found.setdefault(".".join(scope), []).append(("check", child.lineno, guarded, ""))
        if isinstance(child, ast.Call):
            function = child.func
            name = function.attr if isinstance(function, ast.Attribute) else getattr(function, "id", "")
            owner = (
                function.value.id
                if isinstance(function, ast.Attribute) and isinstance(function.value, ast.Name)
                else ""
            )
            found.setdefault(".".join(scope), []).append(("call", child.lineno, guarded, f"{owner}.{name}"))
        _walk(child, scope, guarded or (isinstance(child, ast.If) and _is_guard(child.test)), found)


def _unguarded() -> dict[str, list[int]]:
    """``path::function`` of each unguarded value check or internal-check call, to its offending lines."""
    sites: dict[str, dict[str, list[tuple[str, int, bool, str]]]] = {}
    for path in sorted(SRC.rglob("*.py")):
        relative = path.relative_to(SRC).as_posix()
        functions: dict[str, list[tuple[str, int, bool, str]]] = {}
        _walk(ast.parse(path.read_text()), [], False, functions)
        if relative in EXEMPT or relative.split("/")[0] in EXEMPT:
            functions = {name: [item for item in items if item[0] == "call"] for name, items in functions.items()}
        sites[f"src/better_robot/{relative}"] = functions
    helpers = {
        function.rsplit(".", 1)[-1]
        for functions in sites.values()
        for function, items in functions.items()
        if function.rsplit(".", 1)[-1].startswith("_check_") and any(kind == "check" for kind, *_ in items)
    }
    offenders: dict[str, list[int]] = {}
    for path, functions in sites.items():
        for function, items in functions.items():
            for kind, line, guarded, callee in items:
                owner, _, name = callee.rpartition(".")
                bad = (kind == "check" and not guarded and not function.rsplit(".", 1)[-1].startswith("_check_")) or (
                    kind == "call" and not guarded and (name in helpers or owner == "_internal_checks")
                )
                if bad:
                    offenders.setdefault(f"{path}::{function}", []).append(line)
    return offenders


def test_the_switch_is_read_once_by_its_module() -> None:
    """Only ``_internal_checks.py`` names the variable, and ``ENABLED`` follows it in a fresh interpreter."""
    readers = [
        path.relative_to(SRC).as_posix()
        for path in SRC.rglob("*.py")
        if "BETTER_ROBOT_INTERNAL_CHECKS" in path.read_text()
    ]
    assert readers == ["_internal_checks.py"]
    script = "import better_robot._internal_checks as checks; print(checks.ENABLED)"
    for value, expected in (("1", "True"), ("", "False")):
        environment = {**os.environ, "BETTER_ROBOT_INTERNAL_CHECKS": value}
        result = subprocess.run(
            [sys.executable, "-c", script], env=environment, capture_output=True, text=True, check=True
        )
        assert result.stdout.strip() == expected


def test_the_test_tiers_run_with_the_switch_on() -> None:
    """``tests/conftest.py`` turns the switch on before better_robot imports, unless the environment sets it."""
    assert _internal_checks.ENABLED == (os.environ["BETTER_ROBOT_INTERNAL_CHECKS"] == "1")


def test_value_checks_sit_behind_the_switch_or_an_opt_in_argument() -> None:
    """A new unguarded value check fails: guard it, make it an opt-in ``check_*`` argument, or move it to io."""
    unguarded = _unguarded()
    over, _ = compare(RATCHET, {site: len(lines) for site, lines in unguarded.items()})
    new = [f"{entry}, lines {unguarded[entry.rsplit(': ', 1)[0]]}" for entry in over]
    assert not new, "value-reading checks outside their guard:\n  " + "\n  ".join(new)


def test_value_check_ratchet_has_no_stale_entries() -> None:
    """An entry whose check is now guarded or gone leaves the ratchet with the fix."""
    _, under = compare(RATCHET, {site: len(lines) for site, lines in _unguarded().items()})
    assert not under, "lower these counts in tests/contract/ratchets/value_checks.json (drop an entry at 0):\n  " + (
        "\n  ".join(under)
    )


def test_a_quaternion_off_unit_norm_fails_and_a_non_finite_one_passes() -> None:
    """The invariant fires on the library's own bug, not on a caller's NaN."""
    poses = torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]] * 2, dtype=torch.float64)
    check_unit_quaternions(poses, name="joint_placements")
    with pytest.raises(AssertionError, match="joint_placements: a quaternion is 0.1 from unit norm"):
        check_unit_quaternions(poses * 1.1, name="joint_placements")
    check_unit_quaternions(torch.full((2, 7), torch.nan, dtype=torch.float64), name="joint_placements")
