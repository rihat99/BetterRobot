"""``dead_citations`` on a small git repository written to ``tmp_path``."""

from __future__ import annotations

import subprocess
from pathlib import Path

from better_robot.testing import dead_citations

SOURCE = """LIMIT = 3
level: int = 1


class Box:
    size = 2

    def open(self):
        pass


async def fetch():
    pass


try:
    import fast
except ImportError:

    def fallback():
        pass
"""


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
            "src/m.py": SOURCE,
            "docs/figure.svg": "<svg/>\n",
            "docs/page.md": (
                "`src/m.py::Box.open`, `src/m.py::Box.size`, `src/m.py::LIMIT`, `src/m.py::level`, `src/m.py::fetch`,\n"
                "`src/m.py::fallback` and `docs/figure.svg` resolve. `src/m.py::missing`, `src/m.py::open` (no class),\n"
                "`src/m.py::Box.close`, `docs/figure.svg::x`, `src/gone.py` and `tests/x.py::f` do not; `src/m.py:4`\n"
                "and `docs/figure.svg:1` cite lines. `.github/workflows/ci.yml`, `plan/README.md`, `scripts/run.sh`\n"
                "and `notes/src/gone.py` are not tracked.\n"
            ),
            "notes.txt": "src/gone.py is not read: .txt is not a cited text kind.\n",
            "tests/contract/ratchets/citations.json": '{"docs/page.md: src/gone.py": "a ratchet lists dead citations"}\n',
        },
    )
    assert dead_citations(root) == [
        "docs/page.md: .github/workflows/ci.yml",
        "docs/page.md: docs/figure.svg:1",
        "docs/page.md: docs/figure.svg::x",
        "docs/page.md: notes/src/gone.py",
        "docs/page.md: plan/README.md",
        "docs/page.md: scripts/run.sh",
        "docs/page.md: src/gone.py",
        "docs/page.md: src/m.py:4",
        "docs/page.md: src/m.py::Box.close",
        "docs/page.md: src/m.py::missing",
        "docs/page.md: src/m.py::open",
        "docs/page.md: tests/x.py::f",
    ]


def test_links_resolve_from_their_page_and_paths_outside_the_repository_are_left(tmp_path: Path) -> None:
    root = _repo(
        tmp_path,
        {
            "docs/guides/run.md": "",
            "docs/page.md": (
                "[Run](guides/run.md) resolves from the page, [gone](guides/gone.md) does not.\n"
                "`BetterHuman/src/m.py`, `../src/gone.py`, `./src/gone.py`, `/abs/src/gone.py`, `$TASK/wp/gone.md`,\n"
                "https://example.com/src/gone.py, `mathjax@3/es5/gone.js` and `gone.md` are not citations.\n"
            ),
            "mkdocs.yml": "nav:\n  - guides/run.md\n  - guides/gone.md\n",
        },
    )
    assert dead_citations(root) == ["docs/page.md: guides/gone.md", "mkdocs.yml: guides/gone.md"]
