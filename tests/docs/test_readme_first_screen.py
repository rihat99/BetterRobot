"""The README's first screen runs something and shows its output (principle 7.2).

The first ```` ```python ```` fence of ``README.md`` starts within its first 40 lines and has an ```` ```output ````
fence docked under it; ``tests/docs/test_snippets.py`` runs that fence and compares the output.
"""

from __future__ import annotations

import re
from pathlib import Path

README = Path(__file__).parents[2] / "README.md"
FIRST_SCREEN_LINES = 40


def test_the_first_python_fence_and_its_output_are_on_the_first_screen() -> None:
    """The first Python fence opens by line 40 and an output fence follows it directly."""
    text = README.read_text()
    fence = re.search(r"^```python[^\n]*\n.*?^```\n(?P<next>[^\n]*)", text, re.MULTILINE | re.DOTALL)
    assert fence, "README.md has no Python fence"
    first_line = text.count("\n", 0, fence.start()) + 1
    assert first_line <= FIRST_SCREEN_LINES, f"README.md's first Python fence opens on line {first_line}"
    assert fence["next"].startswith("```output"), "README.md's first Python fence has no output fence under it"
