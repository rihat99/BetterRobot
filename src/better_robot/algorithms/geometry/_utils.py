"""Geometric utility functions for collision distance computation."""
from __future__ import annotations
import torch

_SAFE_EPS = 1e-6


def normalize_with_norm(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Normalize vector, return (normalized, norm). Safe against zero vectors."""
    # Add epsilon inside norm for gradient safety
    norm = (x + _SAFE_EPS).norm(dim=-1, keepdim=True)
    safe_norm = torch.where(norm == 0.0, torch.ones_like(norm), norm)
    normalized = x / safe_norm
    result_vec = torch.where(norm == 0.0, torch.zeros_like(x), normalized)
    result_norm = norm.squeeze(-1)
    return result_vec, result_norm


def closest_segment_point(
    a: torch.Tensor,
    b: torch.Tensor,
    pt: torch.Tensor,
) -> torch.Tensor:
    """Closest point on segment [a, b] to point pt."""
    ab = b - a
    t = (pt - a).mul(ab).sum(dim=-1) / (ab.mul(ab).sum(dim=-1) + _SAFE_EPS)
    t_clamped = t.clamp(0.0, 1.0)
    return a + ab * t_clamped.unsqueeze(-1)


def closest_segment_to_segment_points(
    a1: torch.Tensor,
    b1: torch.Tensor,
    a2: torch.Tensor,
    b2: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Closest points between two line segments [a1,b1] and [a2,b2]."""
    d1 = b1 - a1
    d2 = b2 - a2
    r = a1 - a2

    a = d1.mul(d1).sum(dim=-1)  # squared length of d1
    e = d2.mul(d2).sum(dim=-1)  # squared length of d2
    f = d2.mul(r).sum(dim=-1)
    c = d1.mul(r).sum(dim=-1)
    b = d1.mul(d2).sum(dim=-1)
    denom = a * e - b * b

    s_num = b * f - c * e
    t_num = a * f - b * c

    s_parallel = -c / (a + _SAFE_EPS)
    t_parallel = f / (e + _SAFE_EPS)

    s = torch.where(denom < _SAFE_EPS, s_parallel, s_num / (denom + _SAFE_EPS))
    t = torch.where(denom < _SAFE_EPS, t_parallel, t_num / (denom + _SAFE_EPS))

    s_clamped = s.clamp(0.0, 1.0)
    t_clamped = t.clamp(0.0, 1.0)

    t_recomp = d2.mul((a1 + d1 * s_clamped.unsqueeze(-1)) - a2).sum(dim=-1) / (e + _SAFE_EPS)
    t_final = torch.where(
        (s - s_clamped).abs() > _SAFE_EPS,
        t_recomp.clamp(0.0, 1.0),
        t_clamped
    )

    s_recomp = d1.mul((a2 + d2 * t_final.unsqueeze(-1)) - a1).sum(dim=-1) / (a + _SAFE_EPS)
    s_final = torch.where(
        (t - t_final).abs() > _SAFE_EPS,
        s_recomp.clamp(0.0, 1.0),
        s_clamped
    )

    c1 = a1 + d1 * s_final.unsqueeze(-1)
    c2 = a2 + d2 * t_final.unsqueeze(-1)
    return c1, c2
