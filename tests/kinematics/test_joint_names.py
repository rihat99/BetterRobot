"""The topology queries take a joint's id or its name; unknown names raise ``UnknownNameError``.

The functions that take a state run with the id and with the name in ``tests/contract/test_identifiers.py``.
"""

from __future__ import annotations

import pytest
import torch

from better_robot.exceptions import UnknownNameError
from better_robot.kinematics import subtree, support
from tests.support.robots import two_link_arm


def test_joint_names_match_ids() -> None:
    model = two_link_arm(torch.float64)
    assert subtree(model, joint="j1") == subtree(model, joint=model.joint_id("j1"))
    assert support(model, joint="j2") == support(model, joint=model.joint_id("j2"))


@pytest.mark.parametrize("resolver", ["joint_id", "frame_id", "body_id"])
def test_unknown_name_raises_typed_error(resolver: str) -> None:
    model = two_link_arm(torch.float64)
    with pytest.raises(UnknownNameError):
        getattr(model, resolver)("no_such_name")
