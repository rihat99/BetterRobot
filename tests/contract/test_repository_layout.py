"""The repository's tree: its top-level entries, a tests folder per src package, no tracked symlink or large file, no
ghost directory (principles 9.1, 9.2, 9.3, 9.6).

Each kind of file has one home, so a second home fails here. ``better_robot.testing.repository.layout_problems``
holds the rules, shared with BetterHuman; this module holds BetterRobot's data: its top-level entries, the folders of
``tests/`` that are not a package's, and the files allowed over 1 MiB, each with its reason.
"""

from __future__ import annotations

from pathlib import Path

from better_robot.testing import repository

REPO = Path(__file__).resolve().parents[2]
LARGE_ON_PURPOSE: dict[str, str] = {}
# Principle 9.1's set without `.github/`: this major has no CI. `agent_notes/` is ignored whole.
TOP_LEVEL = {
    ".claude",
    ".gitignore",
    ".pre-commit-config.yaml",
    ".python-version",
    "AGENTS.md",
    "CLAUDE.md",
    "LICENSE",
    "README.md",
    "benchmarks",
    "docs",
    "examples",
    "mkdocs.yml",
    "pyproject.toml",
    "src",
    "tests",
    "tools",
    "uv.lock",
}
# The folders of tests/ that are not a src package: one per cross-cutting suite.
SUITES = {
    "contract": "rules as tests",
    "parity": "comparisons against an oracle, one folder per oracle",
    "docs": "every snippet of README.md and docs/, with its output",
    "examples": "every example, headless",
    "benchmarks": "the smoke test of benchmarks/speed.py",
    "support": "fixtures and helpers that only these tests import, never collected",
    "fixtures": "the tracked test robots with their licences",
}


def test_the_tree_has_the_standard_layout() -> None:
    """The top-level entries are ``TOP_LEVEL``; each src package has its tests folder and each tests folder is a
    package's or a suite's; no tracked symlink, no file over 1 MiB outside ``LARGE_ON_PURPOSE``, no directory without
    a tracked file."""
    problems = repository.layout_problems(
        REPO, package="better_robot", top_level=TOP_LEVEL, suites=SUITES, large_on_purpose=LARGE_ON_PURPOSE
    )
    assert not problems, "\n".join(problems)
