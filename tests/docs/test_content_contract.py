"""Reader-facing documentation contracts."""

from __future__ import annotations

import ast
from pathlib import Path
import re

import pytest


ROOT = Path(__file__).parents[2]
DOCS = ROOT / "docs"
MARKED_EXAMPLES = {
    DOCS / "index.md": ("front-page-example",),
    DOCS / "guides" / "custom_residual.md": ("custom-residual-example",),
}
PROCESS_PATTERNS = (
    r"named[- ]block",
    r"\blegacy\b",
    r"pre[- ]validated",
    r"work order",
    r"vertical-slice",
    r"migration ledger",
    r"\bM\d+(?:\.\d+|[a-z])?\b",
    r"\bmilestone\b",
    r"roadmap M",
    r"GraphExecutor",
    r"owner approved",
    r"2026-07-1",
)
PYTHON_FENCE = re.compile(
    r"^```(?:python|\{testcode\})[ \t]*\n(.*?)^```[ \t]*$",
    re.MULTILINE | re.DOTALL,
)


def _handwritten_pages() -> tuple[Path, ...]:
    return tuple(
        path for path in DOCS.rglob("*.md") if "_build" not in path.parts and "reference/api" not in path.as_posix()
    )


def _without_marked_examples(path: Path, text: str) -> str:
    for name in MARKED_EXAMPLES.get(path, ()):
        start = f"<!-- {name}:start -->"
        end = f"<!-- {name}:end -->"
        assert text.count(start) == text.count(end) == 1
        prefix, remainder = text.split(start, 1)
        _example, suffix = remainder.split(end, 1)
        text = prefix + suffix
    return text


def test_python_fences_are_executable_or_marked() -> None:
    """Ordinary Python fences are reserved for examples executed by pytest."""
    offenders = []
    for path in _handwritten_pages():
        text = _without_marked_examples(path, path.read_text(encoding="utf-8"))
        if "```python" in text:
            offenders.append(path.relative_to(ROOT).as_posix())
    assert not offenders, (
        f"use {{testcode}} for runnable examples or text for schematics; unclassified Python fences in {offenders}"
    )


def test_reader_snippets_show_results_instead_of_asserting() -> None:
    """Published snippets demonstrate outcomes; tests own assertions."""
    offenders = []
    for path in _handwritten_pages():
        for index, source in enumerate(PYTHON_FENCE.findall(path.read_text(encoding="utf-8")), start=1):
            tree = ast.parse(source)
            has_assertion = any(
                isinstance(node, ast.Assert)
                or (isinstance(node, ast.Call) and ast.unparse(node.func) == "torch.testing.assert_close")
                for node in ast.walk(tree)
            )
            if has_assertion:
                offenders.append(f"{path.relative_to(ROOT).as_posix()}::snippet-{index}")
    assert not offenders, f"replace snippet assertions with printed results in {offenders}"


@pytest.mark.parametrize("pattern", PROCESS_PATTERNS)
def test_reader_docs_do_not_contain_process_jargon(pattern: str) -> None:
    offenders = []
    regex = re.compile(pattern, flags=re.IGNORECASE)
    for path in _handwritten_pages():
        text = path.read_text(encoding="utf-8")
        if regex.search(text):
            offenders.append(path.relative_to(ROOT).as_posix())
    assert not offenders, f"{pattern!r} appears in {offenders}"


def test_ledger_is_only_used_for_source_provenance() -> None:
    offenders = []
    allowed = DOCS / "conventions" / "source_and_license.md"
    for path in _handwritten_pages():
        if path != allowed and re.search(r"\bledger\b", path.read_text(encoding="utf-8"), re.IGNORECASE):
            offenders.append(path.relative_to(ROOT).as_posix())
    assert not offenders, f"process-facing ledger language appears in {offenders}"
