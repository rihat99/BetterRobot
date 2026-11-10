"""Run the Python fences of a Markdown page once and compare what each prints with its ``output`` fence (principle 7.3).

A page's ```` ```python ```` fences run in order in one fresh interpreter, on CPU, seeded (``torch``, ``random`` and
``numpy`` at 0), with viser's server replaced by `FakeServer`. Each fence's standard output is compared in full
with the ```` ```output ```` fence docked under it: lines match after trailing spaces are dropped, a line ``...``
stands for any number of lines, a number printed with fraction digits matches within half a unit of its last printed
digit (so ``0.1235`` accepts ``0.12346``, never ``0.1236``), a float printed without fraction digits (``0.``,
``1e-06``) must equal exactly, and other text matches exactly. A fence that prints needs an output fence. A fence
opts out of running with ``{ .no-run data-reason="..." }``; a page under a folder the caller names opts out with that
folder's reason. Only ```` ```python ```` fences run, so a ```` ```py ````, ```` ```pycon ```` or ``~~~`` Python fence is
a problem. ``BR_DOCS_CAPTURE=1`` also writes each page's per-fence output to
``$BR_DOCS_CAPTURE_DIR/<page>.outputs.json`` (default ``/tmp/br_docs_outputs``), from which a changed output fence
is copied.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from collections.abc import Mapping
from dataclasses import KW_ONLY, dataclass
from pathlib import Path
from typing import final

__all__ = ["check_page", "outputs_match"]

_FENCE = re.compile(
    r"^(?P<indent>[ \t]*)```[ \t]*python(?P<info>[^\n]*)\n(?P<body>.*?)^(?P=indent)```[ \t]*$", re.M | re.S
)
_OUTPUT = re.compile(r"(?P<indent>[ \t]*)```[ \t]*output[^\n]*\n(?P<body>.*?)^[ \t]*```[ \t]*$", re.M | re.S)
_REASON = re.compile(r'data-reason="(?P<reason>[^"]*[^"\s][^"]*)"')
_UNREAD_PYTHON = re.compile(
    r"^[ \t]*(?:```[ \t]*(?:py|py3|python3|pycon|ipython)|~~~[ \t]*(?:python|py|py3|python3|pycon|ipython))(?=[ \t{]|$)",
    re.M,
)
_NUMBER = re.compile(r"[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?")

_RUNNER = """
import contextlib, io, json, os, random, sys
import torch
torch.manual_seed(0)
random.seed(0)
try:
    import numpy
    numpy.random.seed(0)
except ImportError:
    pass
from better_robot.testing import FakeServer
try:
    import viser
    viser.ViserServer = lambda *args, **kwargs: FakeServer()
except ImportError:
    pass
namespace = {"__name__": "__docs__"}
outputs = []
for index, source in enumerate(sys.argv[1:]):
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        exec(compile(source, f"<fence {index}>", "exec"), namespace)
    outputs.append(buffer.getvalue())
sys.stderr.write("\\n__SNIPPET_OUTPUTS__" + json.dumps(outputs))
"""


@final
@dataclass(frozen=True)
class _Fence:
    """One ```` ```python ```` fence: its 1-based ``line``, ``body``, opt-out ``reason`` and docked ``output``."""

    line: int
    body: str
    _: KW_ONLY
    runs: bool
    reason: str | None
    output: str | None


def _page_fences(text: str) -> list[_Fence]:
    """Every Python fence of ``text`` with its opt-out and the ``output`` fence that starts on the next line."""
    fences = []
    for match in _FENCE.finditer(text):
        indent = match["indent"]
        body = "".join(
            line[len(indent) :] if line.startswith(indent) else line for line in match["body"].splitlines(True)
        )
        docked = _OUTPUT.match(text, match.end() + 1)
        output = None
        if docked:
            output = "".join(line[len(indent) :] for line in docked["body"].splitlines(True))
        reason = _REASON.search(match["info"])
        fences.append(
            _Fence(
                line=text.count("\n", 0, match.start()) + 1,
                body=body,
                runs="no-run" not in match["info"],
                reason=reason["reason"] if reason else None,
                output=output,
            )
        )
    return fences


def _numbers_match(expected: str, actual: str) -> bool:
    if not re.search(r"[.eE]", expected):
        return expected == actual
    mantissa, _, exponent = expected.lower().partition("e")
    decimals = len(mantissa.partition(".")[2])
    if decimals == 0:
        # torch prints integer-valued floats as ``1.``: a tolerance of one unit would let ``0.`` accept ``1.``.
        return float(expected) == float(actual)
    half_unit = 0.5 * 10.0 ** (int(exponent or 0) - decimals)
    return abs(float(expected) - float(actual)) <= half_unit * (1 + 1e-9)


def _line_matches(expected: str, actual: str) -> bool:
    expected_numbers, actual_numbers = _NUMBER.findall(expected), _NUMBER.findall(actual)
    return (
        _NUMBER.split(expected) == _NUMBER.split(actual)
        and len(expected_numbers) == len(actual_numbers)
        and all(map(_numbers_match, expected_numbers, actual_numbers))
    )


def outputs_match(expected: str, actual: str) -> bool:
    """Whether ``actual`` printed output matches the ``expected`` output fence (rules in the module docstring)."""
    want = [line.rstrip() for line in expected.rstrip("\n").splitlines()]
    have = [line.rstrip() for line in actual.rstrip("\n").splitlines()]

    def match(i: int, j: int) -> bool:
        if i == len(want):
            return j == len(have)
        if want[i] == "...":
            return any(match(i + 1, k) for k in range(j, len(have) + 1))
        return j < len(have) and _line_matches(want[i], have[j]) and match(i + 1, j + 1)

    return match(0, 0)


def check_page(
    page: Path, *, root: Path, folder_reasons: Mapping[str, str] | None = None, timeout_s: float = 120.0
) -> int:
    """Run ``page``'s fences once and raise ``AssertionError`` on any problem; return the number of compared outputs.

    A problem is a fence that raises or ends the interpreter, an output that differs, a fence that prints with no
    ``output`` fence, an ``output`` fence under a fence that does not run, a Python fence the runner does not read,
    or an opt-out with no reason (its own non-blank ``data-reason``, or the reason of the folder of
    ``folder_reasons``, keyed by path relative to ``root``, that holds the page). ``root`` is the working directory
    of the run.
    """
    name = page.relative_to(root).as_posix()
    folder_reason = next((r for folder, r in (folder_reasons or {}).items() if name.startswith(folder)), None)
    text = page.read_text()
    fences = _page_fences(text)
    problems = [
        f"{name}:{line} a Python fence the runner does not read: write ```python"
        for line in (text.count("\n", 0, m.start()) + 1 for m in _UNREAD_PYTHON.finditer(text))
    ]
    if folder_reason is None:
        problems += [f"{name}:{f.line} opts out without data-reason" for f in fences if not f.runs and not f.reason]
    problems += [f"{name}:{f.line} output under a fence that does not run" for f in fences if f.output and not f.runs]
    running = [fence for fence in fences if fence.runs]
    compared = 0
    if running:
        env = {**os.environ, "CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "2", "MPLBACKEND": "Agg"}
        completed = subprocess.run(
            [sys.executable, "-c", _RUNNER, *(fence.body for fence in running)],
            cwd=root,
            env=env,
            capture_output=True,
            text=True,
            timeout=timeout_s,
            check=False,
        )
        assert completed.returncode == 0, f"{name} failed:\n{completed.stderr[-4000:]}"
        assert "__SNIPPET_OUTPUTS__" in completed.stderr, f"{name}: a fence ended the interpreter before the last fence"
        outputs = json.loads(completed.stderr.rsplit("__SNIPPET_OUTPUTS__", 1)[1])
        if os.environ.get("BR_DOCS_CAPTURE") == "1":
            target = Path(os.environ.get("BR_DOCS_CAPTURE_DIR", "/tmp/br_docs_outputs")) / f"{name}.outputs.json"
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(json.dumps(outputs, indent=1))
        for fence, actual in zip(running, outputs, strict=True):
            if fence.output is None:
                if actual.strip():
                    problems.append(f"{name}:{fence.line} prints but has no output fence:\n{actual}")
                continue
            compared += 1
            if not outputs_match(fence.output, actual):
                problems.append(
                    f"{name}:{fence.line} prints\n{actual}which differs from its output fence\n{fence.output}"
                )
    assert not problems, "\n".join(problems)
    return compared
