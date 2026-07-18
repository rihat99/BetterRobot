"""Forbid hot-path patterns that break ``torch.compile`` and CUDA throughput.

AST-walks ``kinematics/``, ``dynamics/``, ``optim/lm.py``,
``optim/first_order.py``, ``residuals/``, and ``lie/`` and fails the test
if any forbidden idiom appears:

* ``.item()`` / ``.cpu()`` — force a CUDA-host sync.
* ``float(tensor)`` / ``bool(tensor)`` / ``int(tensor)`` — force a
  CUDA-host sync.
* ``tensor.new_tensor(...)`` — allocates a tensor on every call.
* ``torch.zeros`` / ``torch.ones`` / ``torch.empty`` / ``torch.eye`` inside a
  Python loop — should be allocated once outside.
* ``if x.dim() == N`` — branches on rank, which kills compile.

A line may exempt itself with ``# bench-ok: <reason>``.

See ``docs/conventions/performance.md §1`` and
``docs/conventions/testing.md §3``.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2] / "src" / "better_robot"
WATCHED_DIRS = ("kinematics", "dynamics", "residuals", "lie")
# Keep this list narrow: public ``Problem``/``VarSpec`` methods intentionally
# perform host-side boundary validation, while the prevalidated solver update
# must remain sync-free.  ``run`` and initialization may use a reasoned
# ``# bench-ok`` exemption at their documented eager/static boundaries.
WATCHED_FILES = ("optim/lm.py", "optim/first_order.py")
ALLOC_FNS = ("zeros", "ones", "empty", "full", "rand", "randn", "eye")

# Calls with these names are tensor evidence when reached through ``torch``.
# This intentionally stays conservative: Python scalar conversions are allowed
# unless the operand can be shown to derive from a tensor.
_TORCH_TENSOR_FNS = {
    "abs",
    "arange",
    "as_tensor",
    "atan2",
    "cat",
    "clamp",
    "cos",
    "diag",
    "einsum",
    "empty",
    "empty_like",
    "eye",
    "full",
    "full_like",
    "linspace",
    "matmul",
    "ones",
    "ones_like",
    "rand",
    "rand_like",
    "randn",
    "randn_like",
    "sin",
    "sqrt",
    "stack",
    "tensor",
    "vector_norm",
    "where",
    "zeros",
    "zeros_like",
}
_TENSOR_METHODS = {
    "abs",
    "all",
    "any",
    "clamp",
    "clone",
    "detach",
    "expand",
    "expand_as",
    "flatten",
    "max",
    "mean",
    "min",
    "norm",
    "reshape",
    "square",
    "squeeze",
    "sum",
    "to",
    "transpose",
    "unsqueeze",
}


def _walk(node: ast.AST, parents: list[ast.AST] | None = None):
    parents = parents or []
    yield node, parents
    for child in ast.iter_child_nodes(node):
        yield from _walk(child, parents + [node])


def _find_hot_path_files() -> list[Path]:
    out: list[Path] = []
    for sub in WATCHED_DIRS:
        for path in (ROOT / sub).rglob("*.py"):
            if "__pycache__" in path.parts:
                continue
            out.append(path)
    out.extend(ROOT / relative for relative in WATCHED_FILES)
    return sorted(out)


def _is_in_loop(parents: list[ast.AST]) -> bool:
    loop_nodes = (
        ast.For,
        ast.AsyncFor,
        ast.While,
        ast.ListComp,
        ast.SetComp,
        ast.DictComp,
        ast.GeneratorExp,
    )
    return any(isinstance(parent, loop_nodes) for parent in parents)


def _function_name(parents: list[ast.AST]) -> str | None:
    for parent in reversed(parents):
        if isinstance(parent, (ast.FunctionDef, ast.AsyncFunctionDef)):
            return parent.name
    return None


def _is_named_block_solver(file: Path) -> bool:
    return any(file.as_posix().endswith(f"/{relative}") for relative in WATCHED_FILES)


def _exempt(line: str) -> bool:
    return "# bench-ok" in line


def _annotation_is_tensor(  # noqa: PLR0911 - explicit annotation forms
    annotation: ast.expr | None,
) -> bool:
    if annotation is None:
        return False
    if isinstance(annotation, ast.Constant) and isinstance(annotation.value, str):
        return annotation.value in {"Tensor", "torch.Tensor"}
    if isinstance(annotation, ast.Name):
        return annotation.id == "Tensor"
    if isinstance(annotation, ast.Attribute):
        return annotation.attr == "Tensor"
    if isinstance(annotation, ast.BinOp) and isinstance(annotation.op, ast.BitOr):
        return _annotation_is_tensor(annotation.left) or _annotation_is_tensor(annotation.right)
    if isinstance(annotation, ast.Subscript) and isinstance(annotation.value, ast.Name):
        if annotation.value.id in {"Annotated", "Optional"}:
            return _annotation_is_tensor(annotation.slice)
    return False


def _torch_call_name(func: ast.expr) -> str | None:
    """Return the final name for ``torch.foo`` / ``torch.linalg.foo`` calls."""
    if not isinstance(func, ast.Attribute):
        return None
    name = func.attr
    value = func.value
    while isinstance(value, ast.Attribute):
        value = value.value
    if isinstance(value, ast.Name) and value.id == "torch":
        return name
    return None


def _is_tensor_expr(  # noqa: PLR0911, PLR0912 - explicit AST cases are auditable
    node: ast.AST | None,
    tensor_names: set[str],
    tensor_attributes: set[str],
    tensor_returns: set[str],
) -> bool:
    if node is None:
        return False
    if isinstance(node, ast.Name):
        return node.id in tensor_names
    if isinstance(node, ast.Attribute):
        if isinstance(node.value, ast.Name) and node.value.id == "self":
            if node.attr in tensor_attributes:
                return True
        return _is_tensor_expr(node.value, tensor_names, tensor_attributes, tensor_returns)
    if isinstance(node, ast.Subscript):
        return _is_tensor_expr(node.value, tensor_names, tensor_attributes, tensor_returns)
    if isinstance(node, ast.Call):
        torch_name = _torch_call_name(node.func)
        if torch_name in _TORCH_TENSOR_FNS:
            return True
        if isinstance(node.func, ast.Name) and node.func.id in tensor_returns:
            return True
        if isinstance(node.func, ast.Attribute):
            return node.func.attr in _TENSOR_METHODS and _is_tensor_expr(
                node.func.value,
                tensor_names,
                tensor_attributes,
                tensor_returns,
            )
        return False
    if isinstance(node, ast.BinOp):
        # ``@`` is strong array/tensor evidence even before either local has
        # been inferred (for example ``J.mT @ residual``).
        return (
            isinstance(node.op, ast.MatMult)
            or _is_tensor_expr(node.left, tensor_names, tensor_attributes, tensor_returns)
            or _is_tensor_expr(node.right, tensor_names, tensor_attributes, tensor_returns)
        )
    if isinstance(node, ast.UnaryOp):
        return _is_tensor_expr(node.operand, tensor_names, tensor_attributes, tensor_returns)
    if isinstance(node, ast.Compare):
        return _is_tensor_expr(node.left, tensor_names, tensor_attributes, tensor_returns) or any(
            _is_tensor_expr(item, tensor_names, tensor_attributes, tensor_returns) for item in node.comparators
        )
    if isinstance(node, ast.BoolOp):
        return any(_is_tensor_expr(item, tensor_names, tensor_attributes, tensor_returns) for item in node.values)
    if isinstance(node, (ast.List, ast.Tuple)):
        return any(_is_tensor_expr(item, tensor_names, tensor_attributes, tensor_returns) for item in node.elts)
    if isinstance(node, ast.IfExp):
        return _is_tensor_expr(node.body, tensor_names, tensor_attributes, tensor_returns) or _is_tensor_expr(
            node.orelse, tensor_names, tensor_attributes, tensor_returns
        )
    return False


def _scope_for(parents: list[ast.AST], tree: ast.Module) -> ast.AST:
    for parent in reversed(parents):
        if isinstance(parent, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
            return parent
    for parent in reversed(parents):
        if isinstance(parent, ast.ClassDef):
            return parent
    return tree


def _class_for(parents: list[ast.AST]) -> ast.ClassDef | None:
    for parent in reversed(parents):
        if isinstance(parent, ast.ClassDef):
            return parent
    return None


def _mark_tensor_target(
    target: ast.expr,
    tensor_names: set[str],
    tensor_attributes: set[str],
    *,
    class_body: bool = False,
) -> bool:
    changed = False
    if isinstance(target, ast.Name):
        destination = tensor_attributes if class_body else tensor_names
        if target.id in destination:
            return False
        destination.add(target.id)
        changed = True
    elif (
        isinstance(target, ast.Attribute)
        and isinstance(target.value, ast.Name)
        and target.value.id == "self"
        and target.attr not in tensor_attributes
    ):
        tensor_attributes.add(target.attr)
        changed = True
    elif isinstance(target, (ast.Tuple, ast.List)):
        for item in target.elts:
            changed |= _mark_tensor_target(
                item,
                tensor_names,
                tensor_attributes,
                class_body=class_body,
            )
    return changed


def _mark_tensor_assignment(
    target: ast.expr,
    value: ast.expr | None,
    tensor_names: set[str],
    tensor_attributes: set[str],
    tensor_returns: set[str],
    *,
    class_body: bool,
) -> bool:
    if (
        isinstance(target, (ast.Tuple, ast.List))
        and isinstance(value, (ast.Tuple, ast.List))
        and len(target.elts) == len(value.elts)
    ):
        return any(
            _mark_tensor_assignment(
                target_item,
                value_item,
                tensor_names,
                tensor_attributes,
                tensor_returns,
                class_body=class_body,
            )
            for target_item, value_item in zip(target.elts, value.elts, strict=True)
        )
    if not _is_tensor_expr(value, tensor_names, tensor_attributes, tensor_returns):
        return False
    return _mark_tensor_target(
        target,
        tensor_names,
        tensor_attributes,
        class_body=class_body,
    )


def _infer_tensor_evidence(  # noqa: PLR0912 - explicit evidence propagation
    tree: ast.Module,
) -> tuple[dict[ast.AST, set[str]], dict[ast.AST, set[str]], set[str]]:
    tensor_returns = {
        node.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and _annotation_is_tensor(node.returns)
    }
    scopes: dict[ast.AST, list[ast.AST]] = {tree: []}
    tensor_names: dict[ast.AST, set[str]] = {tree: set()}
    scope_classes: dict[ast.AST, ast.ClassDef | None] = {tree: None}

    for node, parents in _walk(tree):
        scope = _scope_for(parents, tree)
        scopes.setdefault(scope, []).append(node)
        tensor_names.setdefault(scope, set())
        scope_classes.setdefault(
            scope,
            scope if isinstance(scope, ast.ClassDef) else _class_for(parents),
        )
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            names = tensor_names.setdefault(node, set())
            scope_classes[node] = _class_for(parents)
            args = [*node.args.posonlyargs, *node.args.args, *node.args.kwonlyargs]
            names.update(arg.arg for arg in args if _annotation_is_tensor(arg.annotation))
            if node.args.vararg and _annotation_is_tensor(node.args.vararg.annotation):
                names.add(node.args.vararg.arg)
            if node.args.kwarg and _annotation_is_tensor(node.args.kwarg.annotation):
                names.add(node.args.kwarg.arg)

    tensor_attributes = {class_scope: set() for class_scope in {*scope_classes.values()}}
    changed = True
    while changed:
        changed = False
        for scope, nodes in scopes.items():
            names = tensor_names[scope]
            attributes = tensor_attributes[scope_classes[scope]]
            class_body = isinstance(scope, ast.ClassDef)
            for node in nodes:
                target: ast.expr | None = None
                value: ast.expr | None = None
                if isinstance(node, ast.Assign):
                    value = node.value
                    for item in node.targets:
                        changed |= _mark_tensor_assignment(
                            item,
                            value,
                            names,
                            attributes,
                            tensor_returns,
                            class_body=class_body,
                        )
                    continue
                if isinstance(node, ast.AnnAssign):
                    target, value = node.target, node.value
                    if _annotation_is_tensor(node.annotation):
                        changed |= _mark_tensor_target(
                            target,
                            names,
                            attributes,
                            class_body=class_body,
                        )
                        continue
                elif isinstance(node, ast.NamedExpr):
                    target, value = node.target, node.value
                if target is not None:
                    changed |= _mark_tensor_assignment(
                        target,
                        value,
                        names,
                        attributes,
                        tensor_returns,
                        class_body=class_body,
                    )

    scope_attributes = {scope: tensor_attributes[class_scope] for scope, class_scope in scope_classes.items()}
    return tensor_names, scope_attributes, tensor_returns


def _find_violations(file: Path, src: str) -> list[str]:  # noqa: PLR0912 - explicit AST checks
    lines = src.splitlines()
    tree = ast.parse(src)
    tensor_names, tensor_attributes, tensor_returns = _infer_tensor_evidence(tree)
    violations: list[str] = []

    for node, parents in _walk(tree):
        if not isinstance(node, ast.Call):
            continue
        line = lines[node.lineno - 1] if node.lineno - 1 < len(lines) else ""

        # ``.item()`` / ``.cpu()`` / ``.new_tensor(...)`` calls.
        if isinstance(node.func, ast.Attribute):
            if node.func.attr in {"item", "cpu"} and not _exempt(line):
                violations.append(
                    f"{file.name}:{node.lineno}: .{node.func.attr}() — "
                    "forces device sync; mark with `# bench-ok: <reason>` to allow"
                )
            if node.func.attr == "new_tensor" and not _exempt(line):
                violations.append(
                    f"{file.name}:{node.lineno}: .new_tensor(...) — "
                    "hoist tensor construction; mark with `# bench-ok: <reason>` to allow"
                )

        # ``float(tensor)`` / ``bool(tensor)`` / ``int(tensor)`` host syncs. Tensor provenance
        # is inferred per function, so ordinary ``float(dt)`` remains legal.
        conversion_names = {"float", "bool"}
        if _is_named_block_solver(file):
            conversion_names.add("int")
        if isinstance(node.func, ast.Name) and node.func.id in conversion_names and node.args:
            scope = _scope_for(parents, tree)
            if _is_tensor_expr(
                node.args[0],
                tensor_names[scope],
                tensor_attributes[scope],
                tensor_returns,
            ) and not _exempt(line):
                violations.append(
                    f"{file.name}:{node.lineno}: {node.func.id}(tensor) — "
                    "forces device sync; mark with `# bench-ok: <reason>` to allow"
                )

        # torch.<alloc>(...) inside an explicit Python loop.
        torch_name = _torch_call_name(node.func)
        init_boundary = _is_named_block_solver(file) and _function_name(parents) in {
            "_static_layout",
            "init_state",
        }
        if torch_name in ALLOC_FNS and _is_in_loop(parents) and not (_exempt(line) or init_boundary):
            violations.append(
                f"{file.name}:{node.lineno}: torch.{torch_name}(...) inside loop — "
                "hoist allocation; mark with `# bench-ok: <reason>` to allow"
            )

    # ``if x.dim() == N`` rank branching is an ``If``, not a call, so keep
    # this small second pass separate from the call-oriented checks above.
    for node, _parents in _walk(tree):
        if not isinstance(node, ast.If) or not isinstance(node.test, ast.Compare):
            continue
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

    return violations


@pytest.mark.parametrize("file", _find_hot_path_files(), ids=lambda path: str(path.relative_to(ROOT)))
def test_no_forbidden_hot_path_patterns(file: Path) -> None:
    violations = _find_violations(file, file.read_text())
    if violations:
        pytest.fail("hot-path violations:\n  " + "\n  ".join(violations))


@pytest.mark.parametrize(
    ("name", "src", "needle"),
    (
        (
            "float_tensor",
            "import torch\ndef hot(x: torch.Tensor):\n    return float(x.norm())\n",
            "float(tensor)",
        ),
        (
            "bool_tensor",
            "import torch\ndef hot(x: torch.Tensor):\n    return bool(x.any())\n",
            "bool(tensor)",
        ),
        (
            "int_tensor",
            "import torch\ndef hot(x: torch.Tensor):\n    return int(x.sum())\n",
            "int(tensor)",
        ),
        (
            "new_tensor",
            "import torch\ndef hot(x: torch.Tensor):\n    return x.new_tensor([1.0])\n",
            ".new_tensor(...)",
        ),
        (
            "eye_in_loop",
            "import torch\ndef hot(x: torch.Tensor):\n"
            "    for _ in range(2):\n        x = x + torch.eye(3)\n    return x\n",
            "torch.eye(...) inside loop",
        ),
    ),
)
def test_new_forbidden_patterns_are_detected(name: str, src: str, needle: str) -> None:
    path = ROOT / "optim/lm.py" if name == "int_tensor" else Path(f"{name}.py")
    violations = _find_violations(path, src)
    assert any(needle in violation for violation in violations), violations


def test_named_block_update_modules_are_watched() -> None:
    watched = {path.relative_to(ROOT).as_posix() for path in _find_hot_path_files()}
    assert "optim/lm.py" in watched
    assert "optim/first_order.py" in watched


def test_scalar_float_conversions_are_not_tensor_syncs() -> None:
    src = (
        "import torch\n"
        "class TensorJoint:\n"
        "    pitch: torch.Tensor\n"
        "class Joint:\n"
        "    pitch = 1.0\n"
        "    def hot(self, x: torch.Tensor):\n"
        "        dt, axis = 0.1, x\n"
        "        return float(self.pitch) + float(dt)\n"
    )
    assert _find_violations(Path("scalar.py"), src) == []


def test_tensor_attribute_provenance_is_detected() -> None:
    src = (
        "import torch\n"
        "class Holder:\n"
        "    def __init__(self, value: torch.Tensor):\n"
        "        self.value = value\n"
        "    def hot(self):\n"
        "        return bool(self.value.any())\n"
    )
    violations = _find_violations(Path("attribute.py"), src)
    assert any("bool(tensor)" in violation for violation in violations), violations


def test_bench_ok_and_hoisted_eye_are_allowed() -> None:
    src = (
        "import torch\n"
        "def hot(x: torch.Tensor):\n"
        "    identity = torch.eye(3)\n"
        "    for _ in range(2):\n"
        "        x = x + identity\n"
        "    return float(x.sum())  # bench-ok: public scalar result\n"
    )
    assert _find_violations(Path("allowed.py"), src) == []
