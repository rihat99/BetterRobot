"""C-TENS-1/2: public kinematics, dynamics, Lie and spatial ops preserve the input dtype and device, and the
robot passes refuse a tensor input in another dtype than the model's (record 0021: no silent promotion)."""

from __future__ import annotations

import pytest
import torch

from better_robot.exceptions import DtypeMismatchError

from ._public_ops import CASE_NAMES, build_case, tensors_of


def _tensor_inputs(args: tuple, kwargs: dict) -> list[tuple[str | int, torch.Tensor]]:
    """Plain tensor arguments after the model (the state is a record, not an input tensor)."""
    positional = [(index, value) for index, value in enumerate(args) if isinstance(value, torch.Tensor)]
    return positional + [(name, value) for name, value in kwargs.items() if isinstance(value, torch.Tensor)]


ROBOT_CASES_WITH_TENSOR_INPUTS = tuple(
    name
    for name in CASE_NAMES
    if "." not in name and _tensor_inputs(*build_case(name, torch.float64, torch.device("cpu"))[1:])
)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("name", CASE_NAMES)
def test_output_matches_input_dtype_and_device(name: str, dtype: torch.dtype) -> None:
    device = torch.device("cpu")
    fn, args, kwargs = build_case(name, dtype, device)
    outputs = tensors_of(fn(*args, **kwargs))
    assert outputs, f"{name} returned no tensors"
    for output in outputs:
        assert output.dtype == dtype, f"{name}: {output.dtype} != {dtype}"
        assert output.device == device, f"{name}: {output.device} != {device}"


@pytest.mark.parametrize("name", ROBOT_CASES_WITH_TENSOR_INPUTS)
def test_tensor_input_in_another_dtype_raises(name: str) -> None:
    fn, args, kwargs = build_case(name, torch.float64, torch.device("cpu"))
    for key, value in _tensor_inputs(args, kwargs):
        cast_args, cast_kwargs = list(args), dict(kwargs)
        if isinstance(key, int):
            cast_args[key] = value.float()
        else:
            cast_kwargs[key] = value.float()
        with pytest.raises(DtypeMismatchError):
            fn(*cast_args, **cast_kwargs)
