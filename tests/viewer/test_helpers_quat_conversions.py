"""Tests for quaternion conversion helpers.

See ``docs/concepts/viewer.md §13``.
"""

from __future__ import annotations

import torch
import pytest

from better_robot.viewer.helpers import quat_xyzw_to_wxyz, quat_wxyz_to_xyzw


def test_roundtrip_xyzw_to_wxyz():
    q = torch.tensor([0.1, 0.2, 0.3, 0.9272])
    q = q / q.norm()
    assert torch.allclose(quat_wxyz_to_xyzw(quat_xyzw_to_wxyz(q)), q, atol=1e-6)


def test_roundtrip_wxyz_to_xyzw():
    q = torch.tensor([0.9272, 0.1, 0.2, 0.3])
    q = q / q.norm()
    assert torch.allclose(quat_xyzw_to_wxyz(quat_wxyz_to_xyzw(q)), q, atol=1e-6)


def test_xyzw_to_wxyz_known():
    # [qx, qy, qz, qw] = [0, 0, 0, 1] → [qw, qx, qy, qz] = [1, 0, 0, 0]
    q = torch.tensor([0.0, 0.0, 0.0, 1.0])
    out = quat_xyzw_to_wxyz(q)
    assert torch.allclose(out, torch.tensor([1.0, 0.0, 0.0, 0.0]))


def test_batch_roundtrip():
    q = torch.randn(8, 4)
    q = q / q.norm(dim=-1, keepdim=True)
    assert torch.allclose(quat_wxyz_to_xyzw(quat_xyzw_to_wxyz(q)), q, atol=1e-6)
