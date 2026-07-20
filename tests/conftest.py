"""Ensure ``better_robot`` is importable when running tests directly.

If the package has been installed via ``uv pip install -e .`` this conftest
is a no-op; otherwise it prepends ``<repo>/src`` to ``sys.path`` so the
tests still work.
"""

from __future__ import annotations

from collections.abc import Callable
from itertools import product
import sys
from pathlib import Path
from typing import TypeAlias, cast

import pytest
import torch

_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))


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
    ``ModelValues`` respectively. Outputs may be one tensor or a tuple of
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
