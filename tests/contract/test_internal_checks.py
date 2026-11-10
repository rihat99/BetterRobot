"""One switch guards every check that reads tensor values (decision record 0021, principle 6.8).

``better_robot._internal_checks`` reads ``BETTER_ROBOT_INTERNAL_CHECKS`` once, at import, and no other module reads
it. A value-reading check is an ``if`` that raises and whose test reads tensor values (``bool``, ``float`` or ``int``
of an expression, ``.item()``, ``.any()``, ``.all()``, ``.tolist()``, ``torch.allclose``, ``torch.equal``, or the
implicit truth of a tensor: ``if x.abs().max() > tol``, ``if x.max()``), or a ``torch._assert_async``. In ``src`` each one is guarded: it sits under ``if _internal_checks.ENABLED:`` or under the
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

# Private on purpose: record 0021's switch is an environment variable and this module, with no public name; this is
# its contract test.
from better_robot import _internal_checks
from better_robot._internal_checks import check_unit_quaternions

from ._ratchet_sites import compare

SRC = Path(__file__).resolve().parents[2] / "src" / "better_robot"
RATCHET = Path(__file__).parent / "ratchets" / "value_checks.json"
EXEMPT = {
    "_internal_checks.py": "its functions are the invariants the switch guards",
    "io": "checks a robot description's values once, at load, where it parses them",
}
VALUE_METHODS = {"item", "any", "all", "tolist", "allclose", "equal", "_assert_async"}
# methods whose result is a tensor: testing its truth (alone or in a comparison) reads its values
TENSOR_METHODS = {"abs", "amax", "amin", "count_nonzero", "det", "isfinite", "isinf", "isnan", "max", "mean", "min"}
TENSOR_METHODS |= {"norm", "prod", "sum", "eq", "ne", "gt", "ge", "lt", "le"}
NOT_VALUES = {"is_tensor", "is_floating_point", "is_complex", "is_grad_enabled", "is_inference_mode_enabled"}


def _root(node: ast.expr) -> ast.expr:
    """The name a chain of calls, attributes and subscripts starts from (``np`` in ``np.abs(x).max()``)."""
    while isinstance(node, (ast.Call, ast.Attribute, ast.Subscript)):
        node = node.func if isinstance(node, ast.Call) else node.value
    return node


def _is_tensor_call(node: ast.AST) -> bool:
    if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
        return False
    root = _root(node)
    if isinstance(root, ast.Name) and root.id in {"np", "numpy", "math"}:
        return False
    if isinstance(node.func.value, ast.Name) and node.func.value.id == "torch":
        return node.func.attr not in NOT_VALUES
    return node.func.attr in TENSOR_METHODS


def _tests_tensor_truth(test: ast.expr) -> bool:
    """Whether ``if test`` calls a tensor's ``__bool__``: the test, a ``not``/``and``/``or`` operand, or an operand of
    a value comparison in it, is a tensor-valued call."""
    if isinstance(test, ast.BoolOp):
        return any(_tests_tensor_truth(value) for value in test.values)
    if isinstance(test, ast.UnaryOp) and isinstance(test.op, ast.Not):
        return _tests_tensor_truth(test.operand)
    if isinstance(test, ast.Compare):
        if all(isinstance(op, (ast.Is, ast.IsNot, ast.In, ast.NotIn)) for op in test.ops):
            return False
        return any(_is_tensor_call(sub) for operand in (test.left, *test.comparators) for sub in ast.walk(operand))
    return _is_tensor_call(test)


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
        return (_reads_values(node.test) or _tests_tensor_truth(node.test)) and any(
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


@pytest.mark.parametrize(
    ("test", "reads"),
    [
        ("state.q.abs().max() > 1e6", True),
        ("x.max()", True),
        ("not torch.equal(a, b)", True),
        ("x.shape[0] != 3 or x.dtype is not torch.float32", False),
        ("not math.isfinite(c) or c <= 0.0", False),
        ("np.abs(points).max() > 1.0", False),
    ],
)
def test_a_value_check_is_found_however_it_reads_the_values(test: str, reads: bool) -> None:
    """An implicit ``__bool__`` of a tensor and ``torch.equal`` read values; shape, Python and numpy tests do not."""
    assert _is_value_check(ast.parse(f"if {test}:\n    raise ValueError\n").body[0]) == reads


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
