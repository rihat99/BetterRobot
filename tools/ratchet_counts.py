"""Check that no ratchet under ``tests/contract/ratchets/`` ever grew or took a new key (principle 6.1).

A ratchet maps a key to its reason, or to ``{"count": n, "reason": ...}`` when one key holds several open sites (a
symbol with three pyright errors). Walks the mainline history of each ratchet (``git log --first-parent --follow``,
oldest first; a ``git mv`` in a commit of its own keeps the history), then the working tree when it differs from
``HEAD``, and fails when:

- a ratchet's total count, or a key's count, rose from one step to the next (counts compare only between two steps
  that both store counts; the step that first records counts may only keep or drop keys);
- the latest step holds a key absent at an earlier step, or a count above that key's count at an earlier step;
- the commit that created a ratchet also removed another one (a rename git could not follow because the content
  changed too: a rename goes in a commit of its own).

Prints one line per ratchet (creating commit, count at creation, latest count) and every violation.
``tests/contract/test_ratchet_history.py`` runs it in the full tier.

    uv run python tools/ratchet_counts.py [--repo PATH]
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

RATCHETS = "tests/contract/ratchets"
WORKING_TREE = "working tree"


def git(repo: Path, *args: str) -> str:
    """Stdout of ``git -C repo *args``."""
    return subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True, check=True).stdout


def counts(text: str) -> tuple[dict[str, int], bool]:
    """Key to count of one ratchet file's text, and whether the file stores counts (a plain entry counts 1)."""
    data = json.loads(text)
    entries = data if isinstance(data, dict) else {json.dumps(item, sort_keys=True): 1 for item in data}
    counted = bool(entries) and all(isinstance(value, dict) for value in entries.values())
    return {key: value["count"] if counted else 1 for key, value in entries.items()}, counted


def steps(repo: Path, path: str) -> list[tuple[str, dict[str, int], bool]]:
    """``(commit or "working tree", counts, counted)`` of every mainline commit that changed ``path``, oldest first,
    then the working tree when it differs from ``HEAD``; deletions are skipped (the file has no keys there)."""
    found = []
    commit = ""
    log = git(repo, "log", "--first-parent", "--follow", "--format=commit %H", "--name-status", "--", path)
    for line in log.splitlines():
        if line.startswith("commit "):
            commit = line.split()[1]
        elif line.strip():
            status, *names = line.split("\t")
            if not status.startswith("D"):
                found.append((commit, *counts(git(repo, "show", f"{commit}:{names[-1]}"))))
    found.reverse()
    on_disk = repo / path
    if on_disk.exists() and (not found or on_disk.read_text() != git(repo, "show", f"HEAD:{path}")):
        found.append((WORKING_TREE, *counts(on_disk.read_text())))
    return found


def check(repo: Path, path: str) -> tuple[str, int, int, list[str]]:
    """Creating commit, count at creation, latest count and the violations of one ratchet."""
    history = steps(repo, path)
    created, first, _ = history[0]
    *_, (_, latest, latest_counted) = history
    problems = []
    if created != WORKING_TREE:
        removed = [
            line.split("\t")[-1]
            for line in git(repo, "show", "--format=", "--name-status", created, "--", f"{RATCHETS}/").splitlines()
            if line.startswith("D")
        ]
        if removed:
            problems.append(f"created in {created[:7]}, which also removed {removed} (rename with edits)")
    for (previous, earlier, earlier_counted), (commit, later, later_counted) in zip(history, history[1:], strict=False):
        both_counted = earlier_counted and later_counted
        total_before, total_after = (
            (sum(earlier.values()), sum(later.values())) if both_counted else (len(earlier), len(later))
        )
        if total_after > total_before:
            problems.append(f"grew {total_before} -> {total_after} in {commit[:12]} (after {previous[:7]})")
        if both_counted:
            grown = sorted(key for key in later.keys() & earlier.keys() if later[key] > earlier[key])
            if grown:
                problems.append(f"counts rose in {commit[:12]} (after {previous[:7]}): {grown}")
    for commit, earlier, earlier_counted in history[:-1]:
        added = sorted(latest.keys() - earlier.keys())
        if added:
            problems.append(f"latest keys absent at {commit[:7]}: {added}")
        if earlier_counted and latest_counted:
            above = sorted(key for key in latest.keys() & earlier.keys() if latest[key] > earlier[key])
            if above:
                problems.append(f"latest counts above those at {commit[:7]}: {above}")
    return created[:12], sum(first.values()), sum(latest.values()), problems


def main() -> int:
    """Check every ratchet tracked at ``HEAD`` or present in the working tree; 1 on any violation."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    repo = parser.parse_args().repo.resolve()
    tracked = set(git(repo, "ls-tree", "-r", "--name-only", "HEAD", f"{RATCHETS}/").split())
    on_disk = {path.relative_to(repo).as_posix() for path in (repo / RATCHETS).glob("**/*.json")}
    lines = [f"{'ratchet':50} {'created':>12} {'at create':>9} {'latest':>7}"]
    failed = False
    for path in sorted(path for path in tracked | on_disk if path.endswith(".json")):
        created, at_create, latest, problems = check(repo, path)
        failed |= bool(problems)
        lines.append(f"{path.removeprefix(RATCHETS + '/'):50} {created:>12} {at_create:>9} {latest:>7}")
        lines.extend(f"  FAIL {problem}" for problem in problems)
    sys.stdout.write("".join(f"{line}\n" for line in lines))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
