"""C-STATE-2/3: public kinematics, dynamics, Lie and spatial ops never mutate their tensor inputs.

The version counter of every tensor reachable from the arguments (``q``, ``v``, ``a``, the kinematic state, the
model's values) must be unchanged after the call.
"""

from __future__ import annotations

from typing import Any

import pytest
import torch

from better_robot.data_model import Model

from ._public_ops import CASE_NAMES, build_case, tensors_of


def _input_tensors(arg: Any) -> list[torch.Tensor]:
    if isinstance(arg, Model):
        return tensors_of(arg.values)
    return tensors_of(arg)


@pytest.mark.parametrize("name", CASE_NAMES)
def test_public_op_leaves_inputs_untouched(name: str) -> None:
    fn, args, kwargs = build_case(name, torch.float64, torch.device("cpu"))
    inputs = [tensor for arg in (*args, *kwargs.values()) for tensor in _input_tensors(arg)]
    assert inputs, f"{name}: no tensor inputs found"
    versions = [tensor._version for tensor in inputs]
    fn(*args, **kwargs)
    changed = [index for index, (tensor, before) in enumerate(zip(inputs, versions)) if tensor._version != before]
    assert not changed, f"{name} mutated input tensors at positions {changed}"
