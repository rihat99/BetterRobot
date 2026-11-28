"""A repository's layout and agent files against the standard of the principles repository, chapters 9 and 10.

Both libraries run these checks from ``tests/contract/test_repository_layout.py`` and
``tests/contract/test_agent_files.py`` with their own package and folders, so a rule changes in one place. Each
function returns one line per problem, naming the path, the rule and the fix, and an empty list for a repository
that follows the standard. They read the working tree and ``git``, so ``root`` is a git checkout.

The layout (``layout_problems``): the tracked top-level entries are exactly the standard ones the caller lists (9.1),
every package under ``src/<package>/`` has a ``tests/`` folder of its name and every ``tests/`` folder is a
package's or a named suite's (9.2), no tracked path is a symlink, which would replace an ignored directory on
checkout, or a file over 1 MiB, which weighs on every clone forever (9.3), and every directory on disk outside the
ignored ones holds a tracked file (9.6).

The agent files (``agent_file_problems``): the root ``CLAUDE.md`` has at most 200 lines and a nested one at most 60
(10.2); no agent file or skill writes a word in capitals for emphasis (6.1, 10.2); ``AGENTS.md`` at the root is one
line naming ``CLAUDE.md`` and no other tool's instruction file exists (10.1); ``.gitignore`` admits exactly the
shared parts of ``.claude/`` and every shared file there is tracked (10.3).
"""

from __future__ import annotations

import os
import re
import subprocess
from collections.abc import Collection
from pathlib import Path, PurePosixPath

__all__ = ["agent_file_problems", "layout_problems"]

_SIZE_LIMIT = 1024 * 1024
_ROOT_LIMIT = 200
_NESTED_LIMIT = 60
_EMPHASIS = re.compile(r"\b(?:ALWAYS|NEVER|IMPORTANT|CRITICAL|MUST|SHOULD|REQUIRED|NOTE|WARNING|DO|NOT|NO|ONLY)\b")
# Other tools' instruction files, by file name or by a folder on their path; AGENTS.md is allowed at the root only.
_OTHER_NAMES = ("AGENTS.md", "GEMINI.md", ".cursorrules", ".windsurfrules", ".clinerules", "copilot-instructions.md")
_OTHER_FOLDERS = (".cursor", ".windsurf", ".clinerules")
_SHARED = ("settings.json", "rules/example", "skills/example")  # under .claude/
_PER_MACHINE = ("settings.local.json", "autoresearch")  # under .claude/


def _git(root: Path, *arguments: str) -> list[str]:
    return subprocess.run(["git", *arguments], cwd=root, capture_output=True, text=True, check=True).stdout.splitlines()


def layout_problems(
    root: Path,
    *,
    package: str,
    top_level: Collection[str],
    suites: Collection[str],
    large_on_purpose: Collection[str] = (),
) -> list[str]:
    """One line per break of the standard layout in the git checkout at ``root`` (module docstring).

    ``package`` names ``src/<package>/``; ``top_level`` lists the allowed top-level entries, ``suites`` the
    ``tests/`` folders that are not a package's (contract, parity, docs), and ``large_on_purpose`` the tracked files
    allowed over 1 MiB, each kept by the caller beside its reason.
    """
    tracked = [(line.split()[0], line.split("\t", 1)[1]) for line in _git(root, "ls-files", "-s")]
    paths = [path for _, path in tracked]
    entries, allowed = {path.split("/", 1)[0] for path in paths}, set(top_level)
    problems = [
        f"{entry}: move it into one of the standard homes (principle 9.1)" for entry in sorted(entries - allowed)
    ]
    problems += [
        f"{entry}: this standard top-level entry is missing (principle 9.1)" for entry in sorted(allowed - entries)
    ]
    packages = {path.split("/")[2] for path in paths if PurePosixPath(path).match(f"src/{package}/*/__init__.py")}
    folders = {path.split("/")[1] for path in paths if path.startswith("tests/") and path.count("/") >= 2}
    problems += [
        f"src/{package}/{name}: give this package its tests/ folder (principle 9.2)"
        for name in sorted(packages - folders)
    ]
    problems += [
        f"tests/{name}: neither a src package nor a suite; move its tests into the folder of the package or the suite "
        "they test (principle 9.2)"
        for name in sorted(folders - packages - set(suites))
    ]
    for mode, path in tracked:
        if mode == "120000":
            problems.append(f"{path}: replace this tracked symlink by a file or an ignore line (principle 9.3)")
        elif path not in large_on_purpose and (size := (root / path).stat().st_size) > _SIZE_LIMIT:
            problems.append(f"{path}: {size} B; shrink it under 1 MiB, or allow it with its reason (principle 9.3)")
    problems += [
        f"{folder}: holds no tracked file; move it to the trash (principle 9.6)" for folder in _ghosts(root, paths)
    ]
    return problems


def _ghosts(root: Path, paths: list[str]) -> list[str]:
    """Directories on disk, ignored ones and ``.git`` aside, that hold no tracked file."""
    holding = {str(PurePosixPath(path).parent) for path in paths}
    holding |= {str(parent) for folder in list(holding) for parent in PurePosixPath(folder).parents}
    listing = _git(root, "ls-files", "--others", "--ignored", "--exclude-standard", "--directory")
    folders = "\n".join(path for path in listing if path.endswith("/"))
    # Directories an ignore rule names itself (.venv/, site/), not those that only hold ignored files; check-ignore
    # exits 1 when it ignores none of them.
    named = subprocess.run(
        ["git", "check-ignore", "--stdin"], cwd=root, input=folders, capture_output=True, text=True, check=False
    ).stdout
    skipped = {path.rstrip("/") for path in named.splitlines()} | {".git"}
    ghosts = []
    for folder, subfolders, _ in os.walk(root):
        relative = Path(folder).relative_to(root).as_posix()
        subfolders[:] = [name for name in subfolders if Path(relative, name).as_posix() not in skipped]
        if relative not in holding:
            ghosts.append(relative)
    return ghosts


def agent_file_problems(root: Path) -> list[str]:
    """One line per break of the agent-file rules in the git checkout at ``root`` (module docstring)."""
    listed = _git(root, "ls-files", "--cached", "--others", "--exclude-standard")
    problems = []
    for path in sorted(p for p in set(listed) if p.endswith("CLAUDE.md") or p.startswith(".claude/")):
        if not (root / path).is_file():
            continue
        text = (root / path).read_text()
        limit, count = _ROOT_LIMIT if path == "CLAUDE.md" else _NESTED_LIMIT, len(text.splitlines())
        if path.endswith("CLAUDE.md") and count > limit:
            problems.append(
                f"{path}: {count} lines, limit {limit}; move procedures into .claude/skills/ and design notes into "
                "module docstrings, and drop what the code shows (principle 10.2)"
            )
        if path.endswith(".md"):
            problems += [
                f"{path}: {match[0]}; write the word in lower case, and make a rule that must hold a contract test "
                "(principles 6.1, 10.2)"
                for match in _EMPHASIS.finditer(text)
            ]
    lines = (root / "AGENTS.md").read_text().splitlines()
    if len(lines) != 1 or set(re.findall(r"[\w./-]+\.md\b", "\n".join(lines))) != {"CLAUDE.md"}:
        problems.append(f"AGENTS.md: {lines}; make it one line that names CLAUDE.md only (principle 10.1)")
    others = {
        path
        for path in listed
        if path != "AGENTS.md"
        and (path.rsplit("/", 1)[-1] in _OTHER_NAMES or any(part in _OTHER_FOLDERS for part in path.split("/")[:-1]))
    }
    # An ignored one at the root counts too: it still steers the agent of whoever has it.
    others |= {
        name
        for name in (*_OTHER_NAMES[1:], *_OTHER_FOLDERS, str(Path(".github", "copilot-instructions.md")))
        if (root / name).exists() and not any(path.startswith(f"{name}/") for path in others)
    }
    problems += [f"{path}: fold it into CLAUDE.md and remove it (principle 10.1)" for path in sorted(others)]
    probe = subprocess.run(
        ["git", "check-ignore", "--no-index", *_SHARED, *_PER_MACHINE],
        cwd=root / ".claude",
        capture_output=True,
        text=True,
        check=False,
    )
    if probe.returncode not in (0, 1):  # 1: none of the paths is ignored
        problems.append(f".claude: git check-ignore failed, so the shared files are unchecked: {probe.stderr.strip()}")
    elif (ignored := set(probe.stdout.splitlines())) != set(_PER_MACHINE):
        problems.append(
            f".gitignore: ignores {sorted(ignored)} under .claude/; admit {list(_SHARED)} and ignore the rest "
            "(principle 10.3)"
        )
    problems += [
        f"{path}: commit this shared agent file, or move it out of .claude/ (principle 10.3)"
        for path in _git(root, "ls-files", "--others", "--exclude-standard", "--", ".claude")
    ]
    return problems
