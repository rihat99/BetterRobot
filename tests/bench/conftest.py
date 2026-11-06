"""Bench fixtures — Panda + G1 + a precomputed KinematicState."""

from __future__ import annotations

import pytest
import torch

import better_robot as br
from tests.support.robots import g1_urdf, panda_urdf


@pytest.fixture(scope="session")
def panda():
    return br.load(panda_urdf(), dtype=torch.float32)


@pytest.fixture(scope="session")
def g1():
    return br.load(g1_urdf(), free_flyer=True, dtype=torch.float32)


@pytest.fixture
def panda_state(panda):
    q = panda.q_neutral.clone()
    return br.forward_kinematics_state(panda, q)
