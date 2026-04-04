"""Lie group operations wrapping PyPose SE3/SO3.

All Lie group operations in BetterRobot go through this module.
To switch to a pure-PyTorch backend, only this file needs changing.
"""

from __future__ import annotations

import torch

# Convention: SE3 tangent vectors follow PyPose ordering: [rotation (3,), translation (3,)]


def se3_exp(tangent: torch.Tensor) -> torch.Tensor:
    """Map se(3) tangent vector to SE3 transform.

    Args:
        tangent: Shape (..., 6). Lie algebra element following PyPose convention:
            [rx, ry, rz, tx, ty, tz] (rotation first, then translation).

    Returns:
        Shape (..., 7). SE3 transform as wxyz+xyz quaternion+translation.
    """
    raise NotImplementedError


def se3_compose(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Compose two SE3 transforms: result = a @ b.

    Args:
        a: Shape (..., 7). Left SE3 transform.
        b: Shape (..., 7). Right SE3 transform.

    Returns:
        Shape (..., 7). Composed SE3 transform.
    """
    raise NotImplementedError


def se3_inverse(t: torch.Tensor) -> torch.Tensor:
    """Invert an SE3 transform.

    Args:
        t: Shape (..., 7). SE3 transform.

    Returns:
        Shape (..., 7). Inverse SE3 transform.
    """
    raise NotImplementedError


def se3_log(t: torch.Tensor) -> torch.Tensor:
    """Map SE3 transform to se(3) tangent vector.

    Args:
        t: Shape (..., 7). SE3 transform.

    Returns:
        Shape (..., 6). Lie algebra element.
    """
    raise NotImplementedError


def se3_identity() -> torch.Tensor:
    """Return the SE3 identity transform.

    Returns:
        Shape (7,). Identity: wxyz=[1,0,0,0], xyz=[0,0,0].
    """
    raise NotImplementedError
