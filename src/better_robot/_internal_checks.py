"""One switch for the internal invariants that read tensor values (decision record 0021, principle 6.8).

``ENABLED`` reads the environment variable ``BETTER_ROBOT_INTERNAL_CHECKS`` once, at import; ``"1"`` turns the checks
on. A check here verifies the library's own result, so it fails only on a bug in the library, never on a caller's
input (a caller checks their own input with an opt-in ``check_*`` argument). It reads tensor values, a host sync per
call, so every call sits under ``if _internal_checks.ENABLED:`` and the performance promises are void while the switch
is on. A failure raises ``AssertionError``, explicitly, because ``python -O`` strips ``assert`` statements.
"""

from __future__ import annotations

import os

import torch

ENABLED: bool = os.environ.get("BETTER_ROBOT_INTERNAL_CHECKS") == "1"


def check_unit_quaternions(poses: torch.Tensor, *, name: str) -> None:
    """Raise ``AssertionError`` unless the quaternion of every finite pose has unit norm.

    Parameters
    ----------
    poses
        ``(..., 7)`` poses ``[tx, ty, tz, qx, qy, qz, qw]`` the library normalised.
    name
        The field the poses fill, for the message.

    The tolerance is ``sqrt(eps)`` of the dtype, never tighter than float32's (``3.5e-4``): a model built in float32
    and moved to float64 keeps float32-unit quaternions. A non-finite pose passes: it comes from a non-finite input,
    which only a check of that input may reject.
    """
    tolerance = max(torch.finfo(poses.dtype).eps, torch.finfo(torch.float32).eps) ** 0.5
    error = (torch.linalg.vector_norm(poses[..., 3:], dim=-1) - 1.0).abs()
    if bool((error > tolerance).any()):
        largest = float(error.nan_to_num(0.0).max())
        raise AssertionError(f"{name}: a quaternion is {largest:.3g} from unit norm; a library bug")
