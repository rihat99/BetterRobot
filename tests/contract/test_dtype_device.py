"""C-TENS-1/2: public kinematics, dynamics, Lie and spatial ops preserve the input dtype and device, the robot
passes refuse a tensor input in another dtype than the model's (record 0021: no silent promotion), and leading batch
axes ``(2,)`` and ``(3, 1)`` give each element the result it has alone. The entry points are every public function
of those packages (``_public_ops``), less the reasoned ``NOT_DRIVEN`` ones."""

from __future__ import annotations

import math
from typing import Any

import pytest
import torch

from better_robot.exceptions import DtypeMismatchError
from better_robot.kinematics import KinematicState, forward_kinematics
from better_robot.model import Model

from ._public_ops import CASE_NAMES, NOT_DRIVEN, PUBLIC_FUNCTIONS, build_case, tensors_of

BATCHES = ((2,), (3, 1))


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


def test_the_cases_cover_every_public_function() -> None:
    """Each public kinematics, dynamics, lie and spatial function has a case or a reason it has none."""
    assert sorted(CASE_NAMES) == sorted(PUBLIC_FUNCTIONS.keys() - NOT_DRIVEN.keys())
    assert NOT_DRIVEN.keys() <= PUBLIC_FUNCTIONS.keys()


def _stacked(values: list[Any], batch: tuple[int, ...], model: Model | None) -> Any:
    """One argument for a batch of variants: floating tensors stacked to ``(*batch, *event)``, a state recomputed
    from its stacked configuration; anything else (the model, ids, an integer table) is the first variant's."""
    first = values[0]
    if isinstance(first, torch.Tensor) and first.is_floating_point():
        return torch.stack(values).reshape(*batch, *first.shape)
    if isinstance(first, KinematicState):
        return forward_kinematics(model, torch.stack([value.q for value in values]).reshape(*batch, *first.q.shape))
    return first


@pytest.mark.parametrize("batch", BATCHES, ids=str)
@pytest.mark.parametrize("name", CASE_NAMES)
def test_leading_batch_axes_give_each_element_its_own_result(name: str, batch: tuple[int, ...]) -> None:
    """C-TENS-2: inputs ``(*batch, n)`` give results ``(*batch, ...)`` whose every element equals the ``(n,)`` call."""
    variants = [build_case(name, torch.float64, torch.device("cpu"), variant=k) for k in range(math.prod(batch))]
    fn, args, kwargs = variants[0]
    model = args[0] if isinstance(args[0], Model) else None
    batched_args = [_stacked([case[1][i] for case in variants], batch, model) for i in range(len(args))]
    batched_kwargs = {key: _stacked([case[2][key] for case in variants], batch, model) for key in kwargs}
    results = tensors_of(fn(*batched_args, **batched_kwargs))
    for k, (_, element_args, element_kwargs) in enumerate(variants):
        index = tuple(int(i) for i in torch.unravel_index(torch.tensor(k), batch))
        expected = tensors_of(fn(*element_args, **element_kwargs))
        assert len(results) == len(expected), name
        for result, alone in zip(results, expected, strict=True):
            assert result.shape == (*batch, *alone.shape), f"{name}: {tuple(result.shape)} for {batch} + {alone.shape}"
            torch.testing.assert_close(result[index], alone, msg=f"{name}: element {index} differs from its call alone")
