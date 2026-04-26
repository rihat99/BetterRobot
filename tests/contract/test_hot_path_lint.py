"""Forbid hot-path patterns that break ``torch.compile`` and CUDA throughput.

AST-walks ``kinematics/``, ``dynamics/``, and ``optim/optimizers/`` and
fails the test if any of these forbidden idioms appear:

* ``.item()`` — forces a CUDA-host sync.
* ``.cpu()`` — forces a CUDA-host sync.
* ``torch.zeros`` / ``torch.ones`` / ``torch.empty`` *inside a Python loop*
  — should be allocated once outside; allocates per-iter trash garbage.
* ``if x.dim() == N`` — branches on rank, kills compile.

A line may exempt itself with ``# bench-ok: <reason>``.

See ``docs/conventions/performance.md §1`` and
``docs/conventions/testing.md §3``.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2] / "src" / "better_robot"
WATCHED = ("kinematics", "dynamics", "optim/optimizers")
ALLOC_FNS = ("zeros", "ones", "empty", "full", "rand", "randn")


def _walk(node, parents=None) -> list:
    parents = parents or []
    yield node, parents
    for child in ast.iter_child_nodes(node):
        yield from _walk(child, parents + [node])


def _find_hot_path_files() -> list[Path]:
    out: list[Path] = []
    for sub in WATCHED:
        for p in (ROOT / sub).rglob("*.py"):
            if "__pycache__" in p.parts:
                continue
            out.append(p)
    return out


def _is_in_loop(parents) -> bool:
    return any(isinstance(p, (ast.For, ast.While)) for p in parents)


def _exempt(line: str) -> bool:
    return "# bench-ok" in line


@pytest.mark.parametrize("file", _find_hot_path_files(), ids=lambda p: str(p.relative_to(ROOT)))
def test_no_forbidden_hot_path_patterns(file: Path) -> None:
    src = file.read_text()
    lines = src.splitlines()
    tree = ast.parse(src)

    violations: list[str] = []

    for node, parents in _walk(tree):
        # ``.item()`` / ``.cpu()`` calls
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            if node.func.attr in {"item", "cpu"}:
                line = lines[node.lineno - 1] if node.lineno - 1 < len(lines) else ""
                if not _exempt(line):
                    violations.append(
                        f"{file.name}:{node.lineno}: .{node.func.attr}() — "
                        "forces device sync; mark with `# bench-ok: <reason>` to allow"
                    )

        # torch.<alloc>(...) inside a loop
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr in ALLOC_FNS
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "torch"
            and _is_in_loop(parents)
        ):
            line = lines[node.lineno - 1] if node.lineno - 1 < len(lines) else ""
            if not _exempt(line):
                violations.append(
                    f"{file.name}:{node.lineno}: torch.{node.func.attr}(...) inside loop — "
                    "hoist allocation; mark with `# bench-ok: <reason>` to allow"
                )

        # ``if x.dim() == N`` rank branching
        if isinstance(node, ast.If) and isinstance(node.test, ast.Compare):
            test = node.test
            if (
                isinstance(test.left, ast.Call)
                and isinstance(test.left.func, ast.Attribute)
                and test.left.func.attr == "dim"
            ):
                line = lines[node.lineno - 1] if node.lineno - 1 < len(lines) else ""
                if not _exempt(line):
                    violations.append(
                        f"{file.name}:{node.lineno}: branching on `tensor.dim()` — "
                        "kills `torch.compile`; mark with `# bench-ok: <reason>` to allow"
                    )

    if violations:
        pytest.fail("hot-path violations:\n  " + "\n  ".join(violations))
