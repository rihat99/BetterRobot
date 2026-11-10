"""No hot-path pattern that breaks ``torch.compile`` or CUDA throughput (C-TENS-4): ``tools/hot_path_lint.py`` runs
on every watched file, and planted sources show what it catches and what it allows."""

from __future__ import annotations

import runpy
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
LINT = runpy.run_path(str(REPO / "tools" / "hot_path_lint.py"))  # tools/ is scripts, not a package
find_violations, hot_path_files = LINT["find_violations"], LINT["hot_path_files"]
SOLVERS = REPO / LINT["WATCHED_FILES"][0]


@pytest.mark.parametrize("file", hot_path_files(), ids=lambda path: path.relative_to(REPO).as_posix())
def test_no_forbidden_hot_path_patterns(file: Path) -> None:
    violations = find_violations(file, file.read_text())
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
    path = SOLVERS if name == "int_tensor" else Path(f"{name}.py")
    violations = find_violations(path, src)
    assert any(needle in violation for violation in violations), violations


def test_optimizer_update_modules_are_watched() -> None:
    assert SOLVERS in hot_path_files()


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
    assert find_violations(Path("scalar.py"), src) == []


def test_tensor_attribute_provenance_is_detected() -> None:
    src = (
        "import torch\n"
        "class Holder:\n"
        "    def __init__(self, value: torch.Tensor):\n"
        "        self.value = value\n"
        "    def hot(self):\n"
        "        return bool(self.value.any())\n"
    )
    violations = find_violations(Path("attribute.py"), src)
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
    assert find_violations(Path("allowed.py"), src) == []
