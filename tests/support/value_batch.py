"""The loop oracle of value batching: a call on a model whose values carry a batch equals the unbatched call at each
index of that batch."""

from __future__ import annotations

from collections.abc import Callable
from itertools import product

import torch


def assert_value_batched_matches_loop(
    batched_call: Callable[[], tuple[torch.Tensor, ...]],
    loop_call: Callable[[tuple[int, ...]], tuple[torch.Tensor, ...]],
    *,
    execution_batch_shape: tuple[int, ...],
    rtol: float,
    atol: float,
) -> None:
    """Assert that each output of ``batched_call()``, of shape ``(*execution_batch_shape, ...)``, equals at every
    ``index`` of ``execution_batch_shape`` the same output of ``loop_call(index)``, which runs on unbatched values.

    The test writes ``execution_batch_shape``, the broadcast of the q and model-value batches, so the oracle does not
    depend on the broadcast code under test.
    """
    loop_outputs = [loop_call(index) for index in product(*(range(size) for size in execution_batch_shape))]
    actual = batched_call()
    assert len(actual) == len(loop_outputs[0]), "the batched and loop calls return different numbers of outputs"
    for position, actual_part in enumerate(actual):
        expected = torch.stack([outputs[position] for outputs in loop_outputs])
        torch.testing.assert_close(
            actual_part, expected.reshape(*execution_batch_shape, *expected.shape[1:]), rtol=rtol, atol=atol
        )
