"""Executable documentation checks for the project front page."""

from __future__ import annotations

import re
from pathlib import Path


def test_front_page_example_executes_exactly_as_published() -> None:
    """Extract and execute the front-page example instead of duplicating it."""
    index = Path(__file__).parents[2] / "docs" / "index.md"
    page = index.read_text(encoding="utf-8")
    marked = page.split("<!-- front-page-example:start -->", 1)[1].split("<!-- front-page-example:end -->", 1)[0]
    matches = re.findall(r"```python\n(.*?)```", marked, flags=re.DOTALL)

    assert len(matches) == 1
    namespace: dict[str, object] = {}
    exec(compile(matches[0], str(index), "exec"), namespace)

    model = namespace["model"]
    q = namespace["q"]
    info = namespace["info"]
    solution_pose = namespace["solution_pose"]
    assert info.converged
    assert q.tensor.shape == (model.nq,)
    assert solution_pose.shape == (7,)
