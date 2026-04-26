"""Bench fixtures — Panda + G1 + a pre-FK'd Data."""

from __future__ import annotations

import pytest
import torch

import better_robot as br


@pytest.fixture(scope="session")
def panda():
    pytest.importorskip("robot_descriptions")
    from robot_descriptions import panda_description
    return br.load(panda_description.URDF_PATH, dtype=torch.float32)


@pytest.fixture(scope="session")
def g1():
    pytest.importorskip("robot_descriptions")
    from robot_descriptions import g1_description
    return br.load(g1_description.URDF_PATH, free_flyer=True, dtype=torch.float32)


@pytest.fixture
def panda_data(panda):
    q = panda.q_neutral.clone()
    return br.forward_kinematics(panda, q, compute_frames=True)
