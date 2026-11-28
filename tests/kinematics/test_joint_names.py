"""The topology queries take a joint's id or its name.

The functions that take a state run with the id and with the name in ``tests/contract/test_identifiers.py``; an
unknown name raising ``UnknownNameError`` is in ``tests/contract/test_coupling_standard.py``.
"""

from __future__ import annotations

import torch

from better_robot.kinematics import subtree, support
from tests.support.robots import two_link_arm


def test_joint_names_match_ids() -> None:
    model = two_link_arm(torch.float64)
    assert subtree(model, joint="j1") == subtree(model, joint=model.joint_id("j1"))
    assert support(model, joint="j2") == support(model, joint=model.joint_id("j2"))
