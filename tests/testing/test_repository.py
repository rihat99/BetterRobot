"""``testing.repository`` on small git repositories written to ``tmp_path``: a clean one has no problem, and each
planted break of a layout or agent-file rule is a line that names its path and its principle.

A toy repository's paths start with ``./``, its root, so that ``tests/contract/test_citations.py`` does not take them
for citations of this repository.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

from better_robot.testing import repository

CLEAN = {
    ".gitignore": ".claude/*\n!.claude/settings.json\n!.claude/rules/\n!.claude/skills/\n",
    "./.claude/settings.json": "{}\n",
    "AGENTS.md": "Read CLAUDE.md.\n",
    "CLAUDE.md": "# Toy\n",
    "./src/toy/__init__.py": "",
    "./src/toy/core/__init__.py": "",
    "./tests/core/test_core.py": "",
    "./tests/contract/test_rules.py": "",
}
TOP_LEVEL = {".claude", ".gitignore", "AGENTS.md", "CLAUDE.md", "src", "tests"}


def _repo(root: Path, files: dict[str, str]) -> Path:
    for name, text in files.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    subprocess.run(["git", "init", "-q"], cwd=root, check=True)
    subprocess.run(["git", "add", "."], cwd=root, check=True)
    return root


def _paths(problems: list[str]) -> list[str]:
    """The path each problem names, a nested one written from the root as the fixtures write it."""
    paths = [line.split(": ", 1)[0] for line in problems]
    return [f"./{path}" if "/" in path else path for path in paths]


def test_a_clean_repository_has_no_problem(tmp_path: Path) -> None:
    root = _repo(tmp_path, CLEAN)
    assert repository.layout_problems(root, package="toy", top_level=TOP_LEVEL, suites={"contract"}) == []
    assert repository.agent_file_problems(root) == []


def test_each_layout_break_is_a_line(tmp_path: Path) -> None:
    files = {**CLEAN, "notes.txt": "", "./src/toy/io/__init__.py": "", "./tests/misc/test_x.py": ""}
    root = _repo(tmp_path, {**files, "big.bin": "x" * (1024 * 1024 + 1)})
    (root / "CLAUDE.md").rename(root / "AGENTS.txt")
    (root / "link").symlink_to("AGENTS.txt")
    (root / "empty" / "__pycache__").mkdir(parents=True)
    subprocess.run(["git", "add", "-A"], cwd=root, check=True)

    problems = repository.layout_problems(root, package="toy", top_level=TOP_LEVEL, suites={"contract"})

    assert _paths(problems) == [
        "AGENTS.txt", "big.bin", "link", "notes.txt", "CLAUDE.md", "./src/toy/io", "./tests/misc", "big.bin", "link",
        "empty", "./empty/__pycache__",
    ]  # fmt: skip
    assert [line[line.index("(principle") :] for line in problems] == ["(principle 9.1)"] * 5 + [
        "(principle 9.2)"
    ] * 2 + ["(principle 9.3)"] * 2 + ["(principle 9.6)"] * 2
    allowed = repository.layout_problems(
        root, package="toy", top_level=(TOP_LEVEL | {"AGENTS.txt", "big.bin", "link", "notes.txt"}) - {"CLAUDE.md"},
        suites={"contract", "misc"}, large_on_purpose={"big.bin"},
    )  # fmt: skip
    assert _paths(allowed) == ["./src/toy/io", "link", "empty", "./empty/__pycache__"]  # a symlink is never allowed


def test_each_agent_file_break_is_a_line(tmp_path: Path) -> None:
    files = {
        **CLEAN,
        "CLAUDE.md": "line\n" * 201,
        "./src/toy/core/CLAUDE.md": "You MUST read this.\n" + "line\n" * 60,
        "AGENTS.md": "Read CLAUDE.md.\nThen README.md.\n",
        "./docs/GEMINI.md": "",
        "./.cursor/rules/style.mdc": "",
        "./.claude/skills/release/SKILL.md": "NEVER push.\n",
    }
    root = _repo(tmp_path, files)
    (root / ".gitignore").write_text(".claude/*\n!.claude/skills/\n")
    (root / ".claude" / "rules").mkdir()
    (root / ".claude" / "rules" / "new.md").write_text("")
    (root / ".cursorrules").write_text("")  # ignored or not, it steers the agent of whoever has it
    (root / ".git" / "info" / "exclude").write_text(".cursorrules\n")

    problems = repository.agent_file_problems(root)

    assert _paths(problems) == [
        "./.claude/skills/release/SKILL.md", "CLAUDE.md", "./src/toy/core/CLAUDE.md", "./src/toy/core/CLAUDE.md",
        "AGENTS.md", "./.cursor/rules/style.mdc", ".cursorrules", "./docs/GEMINI.md", ".gitignore",
    ]  # fmt: skip
    assert "NEVER" in problems[0] and "201 lines, limit 200" in problems[1] and "61 lines, limit 60" in problems[2]
    assert "MUST" in problems[3] and "settings.json" in problems[-1]
