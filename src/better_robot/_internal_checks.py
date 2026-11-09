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


def check_rotations(transforms: torch.Tensor, *, name: str) -> None:
    """Raise ``AssertionError`` unless the rotation block of every finite transform is orthonormal.

    Parameters
    ----------
    transforms
        ``(..., 3, 4)`` rigid transforms ``[R | p]`` the library computed.
    name
        The field the transforms fill, for the message.

    The tolerance is ``sqrt(eps)`` of the dtype (``3.5e-4`` in float32, ``1.5e-8`` in float64) on every entry of
    ``R @ R.T - I``. A non-finite transform passes: it comes from a non-finite input, which only an opt-in check of
    that input may reject. Under ``torch.compile`` the check returns at once: a host read would break the graph, and
    the eager run of the same code checks it.
    """
    if torch.compiler.is_compiling():
        return
    rotations = transforms[..., :3]
    identity = torch.eye(3, dtype=transforms.dtype, device=transforms.device)
    error = (rotations @ rotations.transpose(-1, -2) - identity).abs().amax(dim=(-2, -1))
    tolerance = torch.finfo(transforms.dtype).eps ** 0.5
    if bool((error > tolerance).any()):
        largest = float(error.nan_to_num(0.0).max())
        raise AssertionError(f"{name}: a rotation block is {largest:.3g} from orthonormal; a library bug")
