"""Execute the Python snippets in the handwritten tutorials and guides."""

from __future__ import annotations

import re
from pathlib import Path

import pytest


DOCS = Path(__file__).parents[2] / "docs"
PYTHON_FENCE = re.compile(
    r"^```(?:python|\{testcode\})[ \t]*\n(.*?)^```[ \t]*$",
    re.MULTILINE | re.DOTALL,
)

PAGES = (
    ("getting_started/installation.md", 1),
    ("getting_started/01_robot_model.md", 1),
    ("getting_started/02_forward_kinematics.md", 1),
    ("getting_started/03_inverse_kinematics.md", 1),
    ("getting_started/04_floating_base.md", 1),
    ("getting_started/05_batched_gpu.md", 1),
    ("guides/custom_residual.md", 1),
    ("guides/load_a_robot.md", 2),
    ("guides/visualize.md", 1),
    ("guides/differentiate_through_kinematics.md", 1),
    ("guides/own_your_optimization_loop.md", 1),
)


@pytest.mark.parametrize(("relative_path", "expected_count"), PAGES)
def test_handwritten_snippets_execute(
    relative_path: str,
    expected_count: int,
) -> None:
    """Run every Python fence exactly as it appears on its documentation page."""
    page = DOCS / relative_path
    sources = PYTHON_FENCE.findall(page.read_text())

    assert len(sources) == expected_count, f"expected {expected_count} Python snippets in {page}, found {len(sources)}"

    namespace: dict[str, object] = {"__name__": "__better_robot_docs__"}
    for index, source in enumerate(sources, start=1):
        filename = f"{page}::snippet-{index}"
        exec(compile(source, filename, "exec"), namespace)
