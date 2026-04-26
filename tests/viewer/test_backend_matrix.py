"""Backend-matrix test: ``SkeletonMode`` through the V1 backends.

V1 ships ``MockBackend`` as a testable in-memory backend and
``ViserBackend`` as the real interactive backend. ``OffscreenBackend``
is future work — see ``docs/design/12_VIEWER.md §10.2`` — so the matrix
comparison returns when the offscreen path lands.
"""

from __future__ import annotations

import pytest
import torch

import better_robot as br
from better_robot.viewer.render_modes.skeleton import SkeletonMode
from better_robot.viewer.render_modes.base import RenderContext
from better_robot.viewer.renderers.testing import MockBackend
from better_robot.viewer.themes import DEFAULT_THEME
from robot_descriptions import panda_description


@pytest.fixture(scope="module")
def panda():
    return br.load(panda_description.URDF_PATH)


@pytest.fixture(scope="module")
def panda_data(panda):
    q0 = panda.q_neutral.clamp(panda.lower_pos_limit, panda.upper_pos_limit)
    return br.forward_kinematics(panda, q0)


def _run_skeleton_on_backend(model, data, backend):
    """Attach SkeletonMode to backend, return the mode."""
    ctx = RenderContext(backend=backend, namespace="/test", theme=DEFAULT_THEME)
    mode = SkeletonMode()
    mode.attach(ctx, model, data)
    return mode


def test_mock_backend_sphere_and_cylinder_counts(panda, panda_data):
    backend = MockBackend()
    mode = _run_skeleton_on_backend(panda, panda_data, backend)

    n_spheres = len(backend.calls_for("add_sphere"))
    n_cylinders = len(backend.calls_for("add_cylinder"))

    assert n_spheres == len(mode._sphere_joints)
    assert n_cylinders == len(mode._cylinder_joints)


def test_mock_backend_update_produces_transforms(panda, panda_data):
    backend = MockBackend()
    mode = _run_skeleton_on_backend(panda, panda_data, backend)
    backend.reset()
    mode.update(panda_data)
    n_transforms = len(backend.calls_for("set_transform"))
    assert n_transforms == len(mode._sphere_joints) + len(mode._cylinder_joints)


def test_mock_backend_detach_removes_all(panda, panda_data):
    backend = MockBackend()
    mode = _run_skeleton_on_backend(panda, panda_data, backend)
    assert len(backend.nodes) > 0
    mode.detach()
    assert len(backend.nodes) == 0


def test_offscreen_backend_is_future_work():
    """OffscreenBackend is a placeholder — see docs/design/12_VIEWER.md §10.2."""
    from better_robot.viewer.renderers.offscreen_backend import OffscreenBackend
    with pytest.raises(NotImplementedError, match="§10.2"):
        OffscreenBackend(width=64, height=64)
