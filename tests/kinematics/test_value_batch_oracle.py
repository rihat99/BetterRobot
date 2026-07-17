"""Self-test for the value-batch loop oracle scaffold introduced by T3.1."""

from __future__ import annotations

import torch


def test_value_batch_loop_oracle_scaffold(value_batch_loop_oracle):
    batch_shape = (2, 3)
    first = torch.arange(6, dtype=torch.float64).reshape(2, 3, 1)
    second = torch.arange(12, dtype=torch.float64).reshape(2, 3, 2)

    def batched_call():
        return first, second

    def loop_call(index):
        return first[index], second[index]

    value_batch_loop_oracle(
        batched_call,
        loop_call,
        execution_batch_shape=batch_shape,
        rtol=0.0,
        atol=0.0,
    )
