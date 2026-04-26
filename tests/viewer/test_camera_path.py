"""Tests for the V1 ``Camera`` dataclass.

``CameraPath`` (orbit / follow_frame / static) is future work — see
``docs/design/12_VIEWER.md §10.7``. Instantiating it today raises
``NotImplementedError``; we cover only that contract here.
"""

from __future__ import annotations

import pytest
import torch

from better_robot.viewer.camera import Camera, CameraPath


def test_camera_dataclass_defaults():
    cam = Camera(
        position=torch.tensor([1.0, 2.0, 3.0]),
        look_at=torch.tensor([0.0, 0.0, 0.0]),
    )
    assert cam.fov_deg == 50.0
    assert cam.up == (0.0, 0.0, 1.0)
    assert cam.near == pytest.approx(0.01)
    assert cam.far == pytest.approx(100.0)


def test_camera_dataclass_carries_tensors():
    cam = Camera(
        position=torch.tensor([2.0, 0.0, 1.0]),
        look_at=torch.tensor([0.0, 0.0, 0.5]),
    )
    assert torch.allclose(cam.position, torch.tensor([2.0, 0.0, 1.0]))
    assert torch.allclose(cam.look_at, torch.tensor([0.0, 0.0, 0.5]))


def test_camera_path_orbit_is_future_work():
    with pytest.raises(NotImplementedError, match="§10.7"):
        CameraPath.orbit(center=torch.zeros(3), radius=1.0, n_frames=4)


def test_camera_path_static_is_future_work():
    cam = Camera(
        position=torch.tensor([2.0, 0.0, 1.0]),
        look_at=torch.tensor([0.0, 0.0, 0.5]),
    )
    with pytest.raises(NotImplementedError, match="§10.7"):
        CameraPath.static(cam, n_frames=5)
