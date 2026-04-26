"""Tests for CollisionMode stub behaviour.

See ``docs/design/12_VIEWER.md §13``.
"""

from __future__ import annotations

import pytest

import better_robot as br
from better_robot.viewer.render_modes.collision import CollisionMode
from better_robot.viewer.render_modes.base import RenderContext
from better_robot.viewer.renderers.testing import MockBackend
from better_robot.viewer.themes import DEFAULT_THEME
from robot_descriptions import panda_description


@pytest.fixture(scope="module")
def panda():
    return br.load(panda_description.URDF_PATH)


class _DummyRC:
    pass


def test_is_available_with_rc(panda):
    data = br.forward_kinematics(panda, panda.q_neutral)
    assert CollisionMode.is_available(panda, data, robot_collision=_DummyRC())


def test_is_available_without_rc(panda):
    data = br.forward_kinematics(panda, panda.q_neutral)
    assert not CollisionMode.is_available(panda, data, robot_collision=None)


def test_attach_raises_not_implemented(panda):
    data = br.forward_kinematics(panda, panda.q_neutral)
    backend = MockBackend()
    ctx = RenderContext(backend=backend, namespace="/test", theme=DEFAULT_THEME)
    mode = CollisionMode(_DummyRC())
    with pytest.raises(NotImplementedError):
        mode.attach(ctx, panda, data)


def test_update_raises_not_implemented(panda):
    data = br.forward_kinematics(panda, panda.q_neutral)
    mode = CollisionMode(_DummyRC())
    with pytest.raises(NotImplementedError):
        mode.update(data)
