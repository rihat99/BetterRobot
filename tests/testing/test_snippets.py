"""``outputs_match`` and ``check_page`` on synthetic pages: each case is a way a wrong page could pass."""

from __future__ import annotations

from pathlib import Path

import pytest

from better_robot.testing.snippets import check_page, outputs_match

F = "```"


@pytest.mark.parametrize(
    ("expected", "actual"),
    [
        ("tensor([0., 0., 0., 1.])", "tensor([1., 0., 0., 0.])"),  # a scalar-first quaternion
        ("tensor(0.)", "tensor(1.)"),
        ("tensor(0.)", "tensor(-1.)"),
        ("1e-06", "2e-06"),
        ("0.1235", "0.1236"),
    ],
)
def test_a_number_off_in_its_last_printed_digit_differs(expected: str, actual: str) -> None:
    """A float printed without fraction digits is exact; one with fraction digits gets half a unit of the last."""
    assert not outputs_match(expected, actual)


@pytest.mark.parametrize(
    ("expected", "actual"),
    [
        ("tensor([0., 0., 0., 1.])", "tensor([0., 0., 0., 1.])"),
        ("0.1235", "0.12346"),
        ("-0.", "0."),
        ("1e-06", "1e-06"),
    ],
)
def test_a_number_within_half_a_unit_matches(expected: str, actual: str) -> None:
    """Rounding the actual value to the printed digits gives the printed value."""
    assert outputs_match(expected, actual)


def _check(tmp_path: Path, text: str) -> None:
    page = tmp_path / "page.md"
    page.write_text(text)
    check_page(page, root=tmp_path)


@pytest.mark.parametrize(
    "text",
    [
        f"{F}py\nraise RuntimeError('boom')\n{F}\n",
        f"{F}pycon\n>>> raise RuntimeError('boom')\n{F}\n",
        "~~~python\nraise RuntimeError('boom')\n~~~\n",
        f'{F}python {{ .no-run data-reason=" " }}\nprint(1)\n{F}\n',
    ],
    ids=["py", "pycon", "tilde", "blank-reason"],
)
def test_a_fence_the_runner_would_skip_fails(tmp_path: Path, text: str) -> None:
    """A Python fence under another tag, or an opt-out with a blank reason, is a problem, not a silent skip."""
    with pytest.raises(AssertionError):
        _check(tmp_path, text)


@pytest.mark.slow  # starts an interpreter that imports torch
@pytest.mark.parametrize(
    "text",
    [
        f"{F}python\nprint('hello')\n{F}\n",
        f"{F}python\nimport sys\nprint('a')\nsys.exit(0)\n{F}\n{F}output\nb\n{F}\n",
    ],
    ids=["prints-without-output-fence", "sys-exit"],
)
def test_a_run_that_hides_its_output_fails(tmp_path: Path, text: str) -> None:
    """A fence that prints needs an output fence, and a fence that ends the interpreter fails cleanly."""
    with pytest.raises(AssertionError):
        _check(tmp_path, text)
