"""Tests for ``TargetsOverlay``.

Uses ``MockBackend`` — no viser/pyrender required. The interactive
``add_transform_control`` path is only exercised in its static form
(frame triad + recorded call) because MockBackend.is_interactive is
False.
"""

from __future__ import annotations

import pytest
import torch

import better_robot as br
from better_robot.viewer.overlays.targets import TargetsOverlay
from better_robot.viewer.render_modes.base import RenderContext
from better_robot.viewer.renderers.testing import MockBackend
from robot_descriptions import panda_description


@pytest.fixture(scope="module")
def panda():
    return br.load(panda_description.URDF_PATH)


@pytest.fixture(scope="module")
def panda_data(panda):
    return br.forward_kinematics(panda, panda.q_neutral)


def _identity_pose() -> torch.Tensor:
    return torch.tensor([0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 1.0])


def test_attach_adds_frame_per_target(panda, panda_data):
    targets = {
        "panda_hand": _identity_pose(),
        "panda_link4": _identity_pose().clone()
            + torch.tensor([0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
    }
    backend = MockBackend()
    ctx = RenderContext(backend=backend, namespace="/test/tgt")
    overlay = TargetsOverlay(targets)
    overlay.attach(ctx, panda, panda_data)

    frame_calls = backend.calls_for("add_frame")
    assert len(frame_calls) == 2


def test_attach_sets_transform_per_target(panda, panda_data):
    targets = {"hand": _identity_pose()}
    backend = MockBackend()
    ctx = RenderContext(backend=backend, namespace="/test/tgt2")
    overlay = TargetsOverlay(targets, scale=0.1)
    overlay.attach(ctx, panda, panda_data)

    transform_calls = backend.calls_for("set_transform")
    assert len(transform_calls) == 1


def test_detach_removes_frames(panda, panda_data):
    targets = {"a": _identity_pose(), "b": _identity_pose()}
    backend = MockBackend()
    ctx = RenderContext(backend=backend, namespace="/test/tgt3")
    overlay = TargetsOverlay(targets)
    overlay.attach(ctx, panda, panda_data)
    n_before = len(backend.nodes)
    overlay.detach()
    assert len(backend.nodes) < n_before


def test_targets_property_returns_copy(panda, panda_data):
    targets = {"hand": _identity_pose()}
    backend = MockBackend()
    ctx = RenderContext(backend=backend, namespace="/test/tgt4")
    overlay = TargetsOverlay(targets)
    overlay.attach(ctx, panda, panda_data)
    copy = overlay.targets
    assert "hand" in copy
    # Modifying the copy must not mutate internal state
    copy["hand"] = torch.zeros(7)
    assert not torch.allclose(overlay.targets["hand"], torch.zeros(7))


def test_set_visible_false(panda, panda_data):
    targets = {"hand": _identity_pose()}
    backend = MockBackend()
    ctx = RenderContext(backend=backend, namespace="/test/tgt5")
    overlay = TargetsOverlay(targets)
    overlay.attach(ctx, panda, panda_data)
    overlay.set_visible(False)
    vis_calls = backend.calls_for("set_visible")
    assert any(not c.args[1] for c in vis_calls)


def test_no_transform_control_on_non_interactive(panda, panda_data):
    """``MockBackend.is_interactive`` is False → no gizmo is attached."""
    targets = {"hand": _identity_pose()}
    backend = MockBackend()
    assert not backend.is_interactive
    ctx = RenderContext(backend=backend, namespace="/test/tgt6")
    overlay = TargetsOverlay(targets)
    overlay.attach(ctx, panda, panda_data)
    assert len(backend.calls_for("add_transform_control")) == 0


def test_on_change_not_called_without_interaction(panda, panda_data):
    """Without a drag, ``on_change`` is never invoked."""
    calls = []
    targets = {"hand": _identity_pose()}
    backend = MockBackend()
    ctx = RenderContext(backend=backend, namespace="/test/tgt7")
    overlay = TargetsOverlay(targets, on_change=calls.append)
    overlay.attach(ctx, panda, panda_data)
    overlay.update(panda_data)
    assert len(calls) == 0


def test_callback_closure_updates_targets(panda, panda_data):
    """Directly fire the private callback; it should update the
    internal targets dict and call the user hook with the new dict.
    """
    received: list[dict] = []
    targets = {"hand": _identity_pose()}
    overlay = TargetsOverlay(targets, on_change=received.append)

    cb = overlay._make_callback("hand")
    new_pose = torch.tensor([1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 1.0])
    cb(new_pose)

    assert len(received) == 1
    assert torch.allclose(received[0]["hand"], new_pose)
    assert torch.allclose(overlay.targets["hand"], new_pose)


def test_live_targets_falls_back_on_non_interactive(panda, panda_data):
    """``MockBackend`` has ``is_interactive=False`` → no handles are
    stored → ``live_targets`` returns the cached ``_targets`` copy."""
    pose = _identity_pose()
    targets = {"hand": pose}
    backend = MockBackend()
    ctx = RenderContext(backend=backend, namespace="/test/live1")
    overlay = TargetsOverlay(targets)
    overlay.attach(ctx, panda, panda_data)

    live = overlay.live_targets()
    assert "hand" in live
    assert torch.allclose(live["hand"], pose)


def test_live_targets_reads_handle(panda, panda_data):
    """With a stub handle that exposes ``position`` / ``wxyz``, the
    overlay reads pose straight off the handle and converts wxyz→xyzw."""
    from types import SimpleNamespace

    class _InteractiveBackend(MockBackend):
        is_interactive = True

        def add_transform_control(self, name, pose, *, scale=0.15,
                                  on_update=None):
            self._record("add_transform_control", name, pose=pose, scale=scale)
            self.nodes.add(name)
            # Return a live handle that the overlay can poll.
            return SimpleNamespace(
                position=(0.25, 0.50, 0.75),
                wxyz=(1.0, 0.0, 0.0, 0.0),
            )

    backend = _InteractiveBackend()
    ctx = RenderContext(backend=backend, namespace="/test/live2")
    overlay = TargetsOverlay({"hand": _identity_pose()})
    overlay.attach(ctx, panda, panda_data)

    live = overlay.live_targets()
    assert "hand" in live
    # position from handle, identity quaternion (wxyz=(1,0,0,0) → xyzw=(0,0,0,1))
    expected = torch.tensor([0.25, 0.50, 0.75, 0.0, 0.0, 0.0, 1.0])
    assert torch.allclose(live["hand"], expected, atol=1e-5)

    # ``targets`` property mirrors the last live read.
    cached = overlay.targets
    assert torch.allclose(cached["hand"], expected, atol=1e-5)
