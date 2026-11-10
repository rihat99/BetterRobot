"""The effects table of ``docs/reference/effects.md`` is the result of one run of the contract tests (principle 2.6).

Rows are the entry points of ``tests/contract/_public_ops.py``. Each column is one contract test parametrized over
those entry points: a cell reads ``yes`` when every case of that test passed for the row and is blank when the test
does not cover the row, so no cell exists without a test behind it. A failed, skipped or missing test stops the render.
``uv run python -m tests.docs.test_effects_table`` rewrites the table on the page; the test below fails while the page
differs from a fresh render.
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path

import pytest

from tests.contract._public_ops import CASE_NAMES

REPO = Path(__file__).resolve().parents[2]
PAGE = REPO / "docs" / "reference" / "effects.md"
BEGIN = "<!-- effects table: rendered by tests/docs/test_effects_table.py -->"
END = "<!-- end of effects table -->"
NOT_PASSED = {"failure", "error", "skipped"}
CONTRACT = "tests/contract"
COLUMNS: dict[str, str] = {
    "inputs untouched": "test_no_inplace.py::test_public_op_leaves_inputs_untouched",
    "dtype and device kept": "test_dtype_device.py::test_output_matches_input_dtype_and_device",
    "batch axes per element": "test_dtype_device.py::test_leading_batch_axes_give_each_element_its_own_result",
    "another dtype raises": "test_dtype_device.py::test_tensor_input_in_another_dtype_raises",
    "inputs validated once": "test_boundary_validation_count.py::test_each_public_call_validates_its_inputs_once",
    "another model's state raises": "test_state_attachment.py::test_state_of_another_model_is_rejected",
    "same operators at every batch size": (
        "test_performance_rules.py::test_a_batch_dispatches_the_same_operators_at_every_size"
    ),
}


def _outcomes() -> dict[tuple[str, str], list[bool]]:
    """``(column, entry point) -> [passed]`` for every case of the column tests, from one pytest run."""
    with tempfile.TemporaryDirectory() as scratch:
        report = Path(scratch) / "report.xml"
        node_ids = [f"{CONTRACT}/{node}" for node in COLUMNS.values()]
        command = [sys.executable, "-m", "pytest", *node_ids, "-q", "-p", "no:cacheprovider", f"--junitxml={report}"]
        run = subprocess.run(command, cwd=REPO, capture_output=True, text=True, check=False)
        assert run.returncode in (0, 1), f"pytest could not run the contract tests:\n{run.stdout[-2000:]}"
        cases = ET.parse(report).getroot().iter("testcase")
        column_of = {node.replace(".py::", "."): column for column, node in COLUMNS.items()}
        outcomes: dict[tuple[str, str], list[bool]] = defaultdict(list)
        for case in cases:
            function, _, parameters = case.attrib["name"].partition("[")
            column = column_of[f"{case.attrib['classname'].rsplit('.', 1)[-1]}.{function}"]
            entry_point = parameters.removesuffix("]").partition("-")[0]
            outcomes[column, entry_point].append(not any(child.tag in NOT_PASSED for child in case))
    return outcomes


def render() -> str:
    """The Markdown table, one row per entry point of ``_public_ops.CASE_NAMES``, in registry order."""
    outcomes = _outcomes()
    failed = sorted(f"{column} / {entry_point}" for (column, entry_point), runs in outcomes.items() if not all(runs))
    assert not failed, "contract tests did not pass, so the table cannot be rendered:\n  " + "\n  ".join(failed)
    empty = [column for column in COLUMNS if not any(key[0] == column for key in outcomes)]
    assert not empty, f"no test result for columns {empty}: the test was renamed or removed"
    unknown = sorted({entry_point for _, entry_point in outcomes} - set(CASE_NAMES))
    assert not unknown, f"results for entry points outside _public_ops.CASE_NAMES: {unknown}"
    lines = ["| Entry point | " + " | ".join(COLUMNS) + " |", "|---" * (len(COLUMNS) + 1) + "|"]
    for entry_point in CASE_NAMES:
        cells = ["yes" if (column, entry_point) in outcomes else "" for column in COLUMNS]
        lines.append(f"| `{entry_point}` | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _split_page(text: str) -> tuple[str, str, str]:
    head, rest = text.split(BEGIN, 1)
    table, tail = rest.split(END, 1)
    return head, table.strip("\n"), tail


@pytest.mark.slow  # runs the column tests in a child pytest (about 6 s)
def test_the_effects_page_equals_a_fresh_render() -> None:
    """Rerun ``uv run python -m tests.docs.test_effects_table`` after a change to the contract tests or the API."""
    assert _split_page(PAGE.read_text())[1] == render()


if __name__ == "__main__":
    head, _, tail = _split_page(PAGE.read_text())
    PAGE.write_text(f"{head}{BEGIN}\n{render()}\n{END}{tail}")
