"""The flat-E execution ABI avoids expanded shared-value storage."""

from __future__ import annotations

import pytest
import torch

from better_robot.data_model.execution_batch import flatten_execution_batch
from better_robot.exceptions import ShapeError


def test_batched_q_and_shared_values():
    q = torch.arange(20.0).reshape(4, 5)
    placements = torch.arange(21.0).reshape(3, 7)
    execution = flatten_execution_batch(
        q, (placements,), value_event_ndims=(2,)
    )
    assert execution.batch_shape == (4,)
    assert execution.size == 4
    assert execution.q.tensor.shape == (4, 5)
    assert execution.values[0].tensor.shape == (1, 3, 7)
    assert execution.values[0].batch_indices.tolist() == [0, 0, 0, 0]


def test_shared_q_and_batched_values():
    q = torch.arange(5.0)
    placements = torch.arange(84.0).reshape(4, 3, 7)
    execution = flatten_execution_batch(
        q, (placements,), value_event_ndims=(2,)
    )
    assert execution.batch_shape == (4,)
    assert execution.q.batch_indices.tolist() == [0, 0, 0, 0]
    assert execution.values[0].batch_indices.tolist() == [0, 1, 2, 3]


def test_multi_axis_broadcast_uses_index_maps():
    q = torch.zeros(2, 1, 5)
    placements = torch.zeros(1, 3, 4, 7)
    execution = flatten_execution_batch(
        q, (placements,), value_event_ndims=(2,)
    )
    assert execution.batch_shape == (2, 3)
    assert execution.size == 6
    assert execution.q.batch_indices.tolist() == [0, 0, 0, 1, 1, 1]
    assert execution.values[0].batch_indices.tolist() == [0, 1, 2, 0, 1, 2]
    assert execution.unflatten(torch.zeros(6, 7)).shape == (2, 3, 7)


def test_mismatched_batches_raise_clear_shape_error():
    with pytest.raises(ShapeError, match="do not broadcast"):
        flatten_execution_batch(
            torch.zeros(2, 5),
            (torch.zeros(3, 4, 7),),
            value_event_ndims=(2,),
        )


def test_shared_value_gradient_is_reduced_once():
    q = torch.zeros(4, 5)
    shared = torch.zeros(3, 7)
    execution = flatten_execution_batch(q, (shared,), value_event_ndims=(2,))
    per_execution = torch.arange(1.0, 5.0).reshape(4, 1, 1).expand(4, 3, 7)
    reduced = execution.values[0].reduce_gradient(per_execution)
    assert reduced.shape == (3, 7)
    torch.testing.assert_close(reduced, torch.full((3, 7), 10.0))
