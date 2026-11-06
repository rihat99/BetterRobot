"""Fixtures shared by every test folder, and the hooks that turn every skip or xfail into a failure."""

from __future__ import annotations

from collections.abc import Callable, Generator
from itertools import product
from typing import TypeAlias, cast

import pytest
import torch

BatchOutput: TypeAlias = torch.Tensor | tuple[torch.Tensor, ...]


def _stack_loop_outputs(
    outputs: list[BatchOutput],
    batch_shape: tuple[int, ...],
) -> BatchOutput:
    """Stack scalar-loop pass outputs back into their execution batch."""
    first = outputs[0]
    if isinstance(first, torch.Tensor):
        tensors = cast(list[torch.Tensor], outputs)
        return torch.stack(tensors).reshape(*batch_shape, *first.shape)

    tuple_outputs = cast(list[tuple[torch.Tensor, ...]], outputs)
    return tuple(
        torch.stack([output[i] for output in tuple_outputs]).reshape(
            *batch_shape,
            *first[i].shape,
        )
        for i in range(len(first))
    )


def assert_value_batched_matches_loop(
    batched_call: Callable[[], BatchOutput],
    loop_call: Callable[[tuple[int, ...]], BatchOutput],
    *,
    execution_batch_shape: tuple[int, ...],
    rtol: float,
    atol: float,
) -> None:
    """Compare one value-batched pass call with scalar calls at every index.

    T3.2 supplies closures that bind the batched values and index a scalar
    ``Model`` respectively. Outputs may be one tensor or a tuple of
    tensors (for example FK's world/local pair). ``execution_batch_shape``
    is the already-resolved q/value broadcast shape, keeping this test helper
    independent of the runtime broadcast implementation it is meant to test.
    """
    if not execution_batch_shape or any(size < 1 for size in execution_batch_shape):
        raise ValueError("execution_batch_shape must contain positive dimensions")

    indices = product(*(range(size) for size in execution_batch_shape))
    expected = _stack_loop_outputs(
        [loop_call(index) for index in indices],
        execution_batch_shape,
    )
    actual = batched_call()

    actual_parts = (actual,) if isinstance(actual, torch.Tensor) else actual
    expected_parts = (expected,) if isinstance(expected, torch.Tensor) else expected
    if len(actual_parts) != len(expected_parts):
        raise AssertionError("batched and loop calls returned different output tuple lengths")
    for actual_part, expected_part in zip(actual_parts, expected_parts, strict=True):
        torch.testing.assert_close(
            actual_part,
            expected_part,
            rtol=rtol,
            atol=atol,
        )


@pytest.fixture
def value_batch_loop_oracle() -> Callable[..., None]:
    """Expose the T3.2 value-batch oracle without coupling tests to conftest."""
    return assert_value_batched_matches_loop


def _fail_skip(report: pytest.TestReport | pytest.CollectReport) -> None:
    """Turn a skip or xfail into a failure that names it (principle 6.5, decision record 0033).

    A skipped or xfailed gate reports green while checking nothing. Reading the report catches every spelling:
    ``pytest.skip``, an aliased ``pytest`` or ``mark``, ``unittest.skip``, ``raise SkipTest`` and ``xfail`` with
    either strictness. A test that needs a GPU carries the ``cuda`` marker instead, which deselects it.
    """
    if hasattr(report, "wasxfail"):
        what = f"xfail ({report.wasxfail or 'no reason given'})"
        del report.wasxfail
    elif report.skipped:
        what = f"skip ({report.longrepr[2]})"
    else:
        return
    report.outcome = "failed"
    report.longrepr = (
        f"{report.nodeid}: {what}. No test skips or xfails (decision record 0033): mark a GPU test `cuda`; a missing "
        "oracle, fixture or browser is an error to fix."
    )


@pytest.hookimpl(wrapper=True, tryfirst=True)
def pytest_runtest_makereport() -> Generator[None, pytest.TestReport, pytest.TestReport]:
    """Fail a test whose setup, call or teardown skipped or xfailed."""
    report = yield
    _fail_skip(report)
    return report


@pytest.hookimpl(wrapper=True, tryfirst=True)
def pytest_make_collect_report() -> Generator[None, pytest.CollectReport, pytest.CollectReport]:
    """Fail a module that skips at collection (``importorskip`` or ``skip(allow_module_level=True)``)."""
    report = yield
    _fail_skip(report)
    return report
