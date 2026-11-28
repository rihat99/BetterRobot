"""The tensor validators of ``better_robot.lie.checks`` beyond what the boundary tests reach (C-ERR-1).

Every public function's boundary test (``tests/lie/test_boundary.py``, ``tests/spatial/test_algebra.py``,
``tests/collision/test_validation.py``) goes through ``check_tensor``; these pin the dtype rule, the shape
``check_broadcast`` returns and the batch-shape arithmetic, with literals worked by hand and ``torch.broadcast_shapes``
as the oracles.
"""

from __future__ import annotations

import re

import pytest
import torch

from better_robot.exceptions import DtypeMismatchError, ShapeError
from better_robot.lie.checks import broadcast_batch, check_broadcast, check_dtype


def test_the_dtype_rule_keeps_float32_and_float64() -> None:
    """``check_dtype`` is the rule ``io.build_model`` and ``Model.to`` apply to their ``dtype`` argument."""
    assert check_dtype(torch.float32) is torch.float32
    assert check_dtype(torch.float64) is torch.float64
    with pytest.raises(DtypeMismatchError, match="dtype=torch.int64 is not float32 or float64"):
        check_dtype(torch.int64)
    with pytest.raises(
        DtypeMismatchError, match="dtype='float64' is not float32 or float64; pass torch.float32 or torch.float64"
    ):
        check_dtype("float64")


def test_check_broadcast_returns_the_shape_or_names_two_arguments_that_do_not_broadcast() -> None:
    assert check_broadcast(q=torch.Size([2, 1]), identity=(3,), streams=()) == (2, 3)
    with pytest.raises(ShapeError, match=re.escape("q has batch axes (2,) and identity (3,), which do not broadcast")):
        check_broadcast(q=(2,), identity=(3,))


@pytest.mark.parametrize(
    "batches", [((2, 1), (3,)), ((), (4, 2)), ((0,), (1,)), ((1,), (0,)), ((2, 3), (3,), (1, 1, 3))]
)
def test_batch_shapes_broadcast_as_torch_broadcasts_them(batches: tuple[tuple[int, ...], ...]) -> None:
    """``broadcast_batch`` is ``torch.broadcast_shapes`` without the sympy import of its first call."""
    assert broadcast_batch(*batches) == tuple(torch.broadcast_shapes(*batches))


@pytest.mark.parametrize("batches", [((2,), (3,)), ((0,), (3,)), ((3, 2), (2, 1), (4, 1))])
def test_batch_shapes_that_do_not_broadcast_raise(batches: tuple[tuple[int, ...], ...]) -> None:
    with pytest.raises(ShapeError, match=re.escape(f"batch shapes {', '.join(map(str, batches))} do not broadcast")):
        broadcast_batch(*batches)
