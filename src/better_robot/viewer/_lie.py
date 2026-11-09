"""Rotations for the layers, computed by ``lie`` (records 0014 and 0022): numpy arrays in, numpy arrays out.

Layers keep numpy ``(T, ...)`` arrays because viser takes numpy; every quaternion operation runs through
``better_robot.lie`` on torch views of them. Quaternions are ``xyzw`` (scalar last) unless a name says ``wxyz``,
viser's order.
"""

from __future__ import annotations

import math

import numpy as np
import torch

from ..lie import se3, so3


def _tensors(*arrays: np.ndarray) -> list[torch.Tensor]:
    """Torch views of ``arrays`` in one floating dtype, the widest given (lie checks that the dtypes agree)."""
    dtype = np.result_type(*arrays, np.float32)
    return [torch.from_numpy(np.ascontiguousarray(array, dtype=dtype)) for array in arrays]


def act(quaternion: np.ndarray, vector: np.ndarray) -> np.ndarray:
    """``vector (..., 3)`` rotated by ``quaternion (..., 4)``, broadcast (``so3.act``)."""
    return so3.act(*_tensors(quaternion, vector)).numpy()


def inverse(quaternion: np.ndarray) -> np.ndarray:
    """Inverse rotations ``(..., 4)`` (``so3.inverse``)."""
    return so3.inverse(*_tensors(quaternion)).numpy()


def compose_rotations(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Rotations ``a * b`` ``(..., 4)``, broadcast: ``b`` applied first (``so3.compose``)."""
    return so3.compose(*_tensors(a, b)).numpy()


def compose(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Poses ``a @ b`` ``(..., 7)``, broadcast: ``b`` placed in the frame of ``a`` (``se3.compose``)."""
    return se3.compose(*_tensors(a, b)).numpy()


def from_z(direction: np.ndarray) -> np.ndarray:
    """Shortest-arc rotations ``(..., 4)`` taking ``+z`` onto each ``direction (..., 3)`` (``so3.exp``).

    A zero direction gives the identity, one opposite ``+z`` a half turn about ``x``.
    """
    (target,) = _tensors(direction)
    axis = torch.stack([-target[..., 1], target[..., 0], torch.zeros_like(target[..., 0])], dim=-1)
    sine = torch.linalg.vector_norm(axis, dim=-1, keepdim=True)
    angle = torch.atan2(sine, target[..., 2:])
    flip = torch.tensor([1.0, 0.0, 0.0], dtype=target.dtype)  # the axis of a half turn, where z x d vanishes
    unit = torch.where(sine > 1e-9, axis / sine.clamp_min(1e-30), flip)
    return so3.exp(unit * torch.where(sine > 1e-9, angle, torch.where(target[..., 2:] < 0, math.pi, 0.0))).numpy()


def to_wxyz(quaternion: np.ndarray) -> np.ndarray:
    """``xyzw`` quaternions ``(..., 4)`` in viser's scalar-first order."""
    return np.concatenate([quaternion[..., 3:4], quaternion[..., :3]], axis=-1)
