"""Tests for cost residuals (stubs — fail until implemented)."""

import pytest
import torch


def test_pose_residual_placeholder() -> None:
    """Placeholder: verify pose_residual raises NotImplementedError."""
    from better_robot.costs import pose_residual

    with pytest.raises(NotImplementedError):
        pose_residual(
            cfg=torch.zeros(7),
            robot=None,  # type: ignore[arg-type]
            target_link_index=0,
            target_pose=torch.tensor([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        )
