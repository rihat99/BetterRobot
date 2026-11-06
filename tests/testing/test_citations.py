"""``dead_citations`` on a small git repository written to ``tmp_path``."""

from __future__ import annotations

import subprocess
from pathlib import Path

from better_robot.testing import dead_citations


def _repo(root: Path, files: dict[str, str]) -> Path:
    for name, text in files.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    subprocess.run(["git", "init", "-q"], cwd=root, check=True)
    subprocess.run(["git", "add", "."], cwd=root, check=True)
    return root


def test_reports_untracked_paths_undefined_symbols_and_line_numbers(tmp_path: Path) -> None:
    root = _repo(
        tmp_path,
        {
            "src/m.py": "LIMIT = 3\n\n\nclass Box:\n    def open(self):\n        pass\n\n\nasync def fetch():\n    pass\n",
            "docs/page.md": (
                "`src/m.py::Box.open`, `src/m.py::LIMIT`, `src/m.py::fetch` and `docs/page.md` resolve.\n"
                "`src/m.py::missing`, `src/gone.py` and `tests/x.py::f` do not; `src/m.py:4` cites a line.\n"
                "`BetterHuman/src/m.py`, `../src/gone.py` and `notes/src/gone.py` belong to someone else.\n"
            ),
            "notes.txt": "src/gone.py is not read: .txt is not a cited text kind.\n",
            "tests/contract/ratchets/citations.json": '{"docs/page.md: src/gone.py": "a ratchet lists dead citations"}\n',
        },
    )
    assert dead_citations(root) == [
        "docs/page.md: src/gone.py",
        "docs/page.md: src/m.py:4",
        "docs/page.md: src/m.py::missing",
        "docs/page.md: tests/x.py::f",
    ]
