"""The one boundary checker of the tensor functions: the rule of C-ERR-1 as `better_robot.lie.checks` applies it."""

from __future__ import annotations

from collections.abc import Callable, Mapping

import pytest
import torch

from better_robot.exceptions import DeviceMismatchError, DtypeMismatchError, ShapeError


def check_boundary(function: Callable[..., object], shapes: Mapping[str, tuple[int, ...]], **inputs: object) -> None:
    """Assert that ``function`` rejects each bad version of every tensor parameter in ``shapes`` with the typed error
    that names it.

    ``shapes`` maps each floating tensor parameter, in the order the function checks them, to its trailing shape
    (``()`` takes any); the first is the one the others must match in dtype, device and batch axes. The valid call
    passes ``torch.zeros(2, *shape)`` (float32, batch ``(2,)``) for each, ``inputs`` for the other parameters or in
    place of one, every argument by keyword. One at a time, each tensor becomes an int64 tensor; with a trailing shape
    also a list and a tensor with one entry too many on its last axis; after the first also a float64 tensor, a meta
    tensor and a tensor with batch ``(3,)``.
    """
    valid = {name: torch.zeros(2, *shape) for name, shape in shapes.items()} | inputs
    function(**valid)
    reference = next(iter(shapes))
    for index, (name, shape) in enumerate(shapes.items()):
        good = valid[name]
        assert isinstance(good, torch.Tensor)
        bad_inputs = [(good.long(), DtypeMismatchError, rf"{name}\.dtype=torch\.int64 is not float32 or float64")]
        if shape:
            sizes = ", ".join(map(str, shape))
            bad_inputs += [
                (good.tolist(), TypeError, rf"{name} must be a torch\.Tensor, got list"),
                (
                    torch.zeros(*good.shape[:-1], good.shape[-1] + 1),
                    ShapeError,
                    rf"{name} must have shape \(\.\.\., {sizes}\), got",
                ),
            ]
        if index:
            bad_inputs += [
                (
                    good.double(),
                    DtypeMismatchError,
                    rf"{name}\.dtype=torch\.float64 != {reference}\.dtype=torch\.float32",
                ),
                (
                    good.to("meta"),
                    DeviceMismatchError,
                    rf"{name}\.device=meta != {reference}\.device=cpu; pass {name}\.to\(cpu\)",
                ),
                (
                    torch.zeros(3, *shape),
                    ShapeError,
                    rf"{reference} has batch axes \(2,\) and {name} \(3,\), which do not broadcast",
                ),
            ]
        for bad, error, match in bad_inputs:
            with pytest.raises(error, match=match):
                function(**{**valid, name: bad})
