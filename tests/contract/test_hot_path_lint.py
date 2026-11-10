"""No hot-path pattern that breaks ``torch.compile`` or CUDA throughput (C-TENS-4): ``tools/hot_path_lint.py`` runs
on every watched file, and planted sources show what it catches and what it allows."""

from __future__ import annotations

import re
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
        (
            "implicit_truth_of_a_reduction",
            "def hot(state):\n    if state.q.abs().max() > 1e6:\n        raise ValueError\n",
            "implicit bool(tensor) in `if` test",
        ),
        (
            "implicit_truth_of_a_tensor_name",
            "import torch\ndef hot(mask: torch.Tensor):\n    while mask:\n        mask = mask[1:]\n",
            "implicit bool(tensor) in `while` test",
        ),
        (
            "tolist",
            "import torch\ndef hot(x: torch.Tensor):\n    if x.tolist()[0] > 0:\n        return x\n",
            ".tolist()",
        ),
        (
            "numpy",
            "import torch\ndef hot(x: torch.Tensor):\n    return x.numpy()\n",
            ".numpy()",
        ),
        (
            "torch_equal",
            "import torch\ndef hot(a, b):\n    if not torch.equal(a, b):\n        raise ValueError\n",
            "torch.equal(...)",
        ),
    ),
)
def test_new_forbidden_patterns_are_detected(name: str, src: str, needle: str) -> None:
    path = SOLVERS if name == "int_tensor" else Path(f"{name}.py")
    violations = find_violations(path, src)
    assert any(needle in violation for violation in violations), violations


def test_optimizer_update_modules_are_watched() -> None:
    """The solver step and the linearization and linear algebra it runs every iteration."""
    optim = REPO / "src" / "better_robot" / "optim"
    assert {SOLVERS, optim / "linear.py", optim / "linearization.py"} <= set(hot_path_files())


def test_contract_09_names_the_watched_list() -> None:
    """C-TENS-4 and contract 09 name one scope: the hot paths are the lint's watched directories and files."""
    page = (REPO / "docs" / "contracts" / "09-performance.md").read_text()
    sentence = re.search(r"Hot paths are (.+?);", page, re.DOTALL)
    assert sentence is not None, "contract 09 no longer defines its hot paths in one 'Hot paths are ...;' sentence"
    assert set(re.findall(r"`([^`]+)`", sentence[1])) == {*LINT["WATCHED_DIRS"], *LINT["WATCHED_FILES"]}


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


def test_structure_and_python_scalar_tests_are_not_tensor_truth() -> None:
    src = (
        "import math\nimport numpy as np\nimport torch\n"
        "def hot(x: torch.Tensor, lower: torch.Tensor | None, c: float, points):\n"
        "    if x.shape[0] > 3 or x.ndim != 2 or lower is None or not math.isfinite(c):\n"
        "        raise ValueError\n"
        "    if np.abs(points).max() > 1.0:\n"
        "        raise ValueError\n"
        "    batch, blocks = x.shape[:2]\n"
        "    if blocks > batch:\n"
        "        raise ValueError\n"
    )
    assert find_violations(Path("structure.py"), src) == []


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
