"""Shared query/value execution-batch validation for dynamics passes."""

from __future__ import annotations

from collections.abc import Mapping

import torch

from ..data_model.execution_batch import broadcast_to_execution_batch
from ..data_model.model_structure import ModelStructure
from ..data_model.model_values import ModelValues
from ..exceptions import DeviceMismatchError, DtypeMismatchError
from ..kinematics.forward import _validate_q


def prepare_dynamics_inputs(
    structure: ModelStructure,
    values: ModelValues,
    q: torch.Tensor,
    inputs: Mapping[str, tuple[torch.Tensor, tuple[int, ...]]],
) -> tuple[torch.Tensor, dict[str, torch.Tensor], tuple[int, ...]]:
    """Validate and broadcast a dynamics query to the model execution batch."""

    _validate_q(structure, values, q)
    batch_shape = values.execution_batch_shape(structure, q)
    q_exec = broadcast_to_execution_batch(
        q,
        batch_shape,
        (structure.nq,),
        name="q",
    )
    prepared: dict[str, torch.Tensor] = {}
    for name, (tensor, event_shape) in inputs.items():
        if tensor.device != q.device:
            raise DeviceMismatchError(f"{name}.device={tensor.device} != q.device={q.device}")
        if tensor.dtype != q.dtype:
            raise DtypeMismatchError(f"{name}.dtype={tensor.dtype} != q.dtype={q.dtype}")
        prepared[name] = broadcast_to_execution_batch(
            tensor,
            batch_shape,
            event_shape,
            name=name,
        )
    return q_exec, prepared, batch_shape


__all__ = ["prepare_dynamics_inputs"]
