"""The contract of a joint kind, checked on one instance (record 0013): shapes, the identity at ``neutral``, the
manifold round trip and the motion subspace against finite differences of the placement."""

from __future__ import annotations

import torch

from ..model.joints import JointModel
from ..model.manifolds import RIGID

_BATCHES = ((), (5,), (2, 3))
_IDENTITY = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0)


def _step(joint: JointModel, q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """``q ⊕ v`` on the joint's own slices, block by block on its manifold: ``(..., nq), (..., nv) -> (..., nq)``."""
    manifold = joint.manifold
    blocks = manifold.integrate(q.unflatten(-1, (-1, manifold.nq)), v.unflatten(-1, (-1, manifold.nv)))
    return blocks.flatten(start_dim=-2)


def _difference(joint: JointModel, q0: torch.Tensor, q1: torch.Tensor) -> torch.Tensor:
    """``q1 ⊖ q0`` on the joint's own slices: ``(..., nq), (..., nq) -> (..., nv)``."""
    manifold = joint.manifold
    blocks = manifold.difference(q0.unflatten(-1, (-1, manifold.nq)), q1.unflatten(-1, (-1, manifold.nq)))
    return blocks.flatten(start_dim=-2)


def _close(actual: torch.Tensor, expected: torch.Tensor, *, tolerance: float, what: str) -> None:
    error = (actual - expected).abs().max().item() if actual.numel() else 0.0
    if error > tolerance:
        raise AssertionError(f"{what}: max error {error:.3e} > {tolerance:.1e}")


def check_joint_model(joint: JointModel, *, dtype: torch.dtype, generator: torch.Generator) -> None:
    """Raise ``AssertionError`` unless ``joint`` honours the `JointModel` contract in ``dtype`` on the CPU.

    Checks, on configurations drawn by ``joint.random`` with limits ``[-0.9, 0.9]`` and tangents drawn from
    ``generator``: ``neutral()`` is ``(nq,)`` and places the child at the identity; ``placement`` and
    ``motion_subspace`` keep the dtype and return ``(B..., 7)`` and ``(B..., 6, nv)`` for the batch shapes ``()``,
    ``(5,)`` and ``(2, 3)``; the placement of every drawn ``q`` carries a unit quaternion (a non-unit axis would
    scale it); the manifold round trip ``difference(q, integrate(q, v)) == v`` and
    ``integrate(q, 0) == q``; and the derivative of ``placement`` along ``integrate(q, t v)`` at ``t = 0``, measured
    by a forward difference of the relative pose, equals the child-frame twist ``motion_subspace(q) @ v``. The
    finite-difference step is ``sqrt(eps(dtype))`` and the tolerance 50 times that.
    """
    nq, nv = joint.nq, joint.nv
    neutral = joint.neutral().to(dtype)
    if tuple(neutral.shape) != (nq,):
        raise AssertionError(f"neutral() has shape {tuple(neutral.shape)}, expected ({nq},)")
    limits = torch.full((nq,), 0.9, dtype=dtype)
    samples = [joint.random(generator, lower=-limits, upper=limits).to(dtype) for _ in range(6)]
    if any(tuple(sample.shape) != (nq,) for sample in samples):
        raise AssertionError(f"random() must return ({nq},)")

    identity = torch.tensor(_IDENTITY, dtype=dtype)
    _close(joint.placement(neutral), identity, tolerance=1e-6, what="placement(neutral()) is the identity pose")
    for batch in _BATCHES:
        q = neutral.expand(*batch, nq)
        pose, subspace = joint.placement(q), joint.motion_subspace(q)
        if pose.dtype != dtype or subspace.dtype != dtype:
            raise AssertionError(f"placement/motion_subspace changed the dtype {dtype}")
        if tuple(pose.shape) != (*batch, 7) or tuple(subspace.shape) != (*batch, 6, nv):
            raise AssertionError(f"placement{batch} is {tuple(pose.shape)}, motion_subspace {tuple(subspace.shape)}")
    if nv == 0:
        return

    q = torch.stack(samples)  # (6, nq)
    eps = torch.finfo(dtype).eps
    quaternion_norm = joint.placement(q)[..., 3:].norm(dim=-1)
    _close(quaternion_norm, torch.ones_like(quaternion_norm), tolerance=100 * eps, what="placement(q) quaternion is unit")
    v = 0.3 * torch.randn(q.shape[0], nv, generator=generator, dtype=dtype)
    _close(_step(joint, q, torch.zeros_like(v)), q, tolerance=1e-6, what="integrate(q, 0) == q")
    tolerance = 1e-5 if dtype == torch.float32 else 1e-10
    _close(
        _difference(joint, q, _step(joint, q, v)), v, tolerance=tolerance, what="difference(q, integrate(q, v)) == v"
    )

    step = eps**0.5
    moved = joint.placement(_step(joint, q, step * v))
    twist = RIGID.difference(joint.placement(q), moved) / step
    expected = (joint.motion_subspace(q) @ v.unsqueeze(-1)).squeeze(-1)
    _close(twist, expected, tolerance=50 * step, what="motion_subspace(q) @ v against the placement's derivative")


__all__ = ["check_joint_model"]
