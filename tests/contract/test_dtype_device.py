"""C-TENS-1/2: public kinematics, dynamics, Lie and spatial ops preserve the input dtype and device."""

from __future__ import annotations

import pytest
import torch

from ._public_ops import CASE_NAMES, build_case, tensors_of


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
