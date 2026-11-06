"""``imported_packages`` on small package trees written to ``tmp_path``."""

from __future__ import annotations

from pathlib import Path

from better_robot.testing import imported_packages


def _write(root: Path, files: dict[str, str]) -> Path:
    for name, source in files.items():
        path = root / "pkg" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(source)
    return root


def test_every_kind_of_import_is_an_edge(tmp_path: Path) -> None:
    root = _write(
        tmp_path,
        {
            "__init__.py": "from . import low, mid\nfrom .top import run\n",
            "low.py": "import math\n",
            "mid/__init__.py": "from ..low import x\n",
            "mid/inner.py": "def f():\n    import pkg.side\n",
            "side.py": "from typing import TYPE_CHECKING\nif TYPE_CHECKING:\n    from pkg.low import y\n",
            "top/__init__.py": "from pkg import mid\nfrom pkg import helper\n",
            "top/deep/leaf.py": "from ...side import z\nfrom .. import run\n",
        },
    )
    assert imported_packages(root, "pkg") == {
        "pkg": {"low", "mid", "top"},
        "low": set(),
        "mid": {"low", "side"},
        "side": {"low"},
        "top": {"mid", "pkg", "side"},
    }


def test_imports_of_itself_and_of_other_packages_are_dropped(tmp_path: Path) -> None:
    root = _write(
        tmp_path,
        {
            "__init__.py": "from pkg import name_in_root\n",
            "a/__init__.py": "from . import b\nfrom pkg.a.b import c\nimport torch\nfrom other.pkg import d\n",
            "a/b.py": "",
        },
    )
    assert imported_packages(root, "pkg") == {"pkg": set(), "a": set()}
