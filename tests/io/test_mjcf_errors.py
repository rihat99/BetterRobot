"""``parse_mjcf`` loads the passive rigid tree: what the tree cannot express raises ``ModelError``, what lies
outside it is ignored by rule (``docs/concepts/io.md``)."""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import torch

from better_robot.exceptions import ModelError
from better_robot.io import load

TWO_HINGES = """<mujoco><worldbody><body name="a"><joint name="h1" type="hinge" axis="0 0 1"/><geom size="0.1"/>
<body name="b" pos="1 0 0"><joint name="h2" type="hinge" axis="0 0 1"/><geom size="0.1"/></body></body>
</worldbody>{extra}</mujoco>"""


def _write(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "r.xml"
    path.write_text(text)
    return path


@pytest.mark.parametrize(
    ("text", "message"),
    [
        pytest.param(
            '<mujoco><worldbody><body name="a"><joint name="b" type="ball" range="0 0.5"/><geom size="0.1"/></body>'
            "</worldbody></mujoco>",
            "ball joint 'b' has a range (a cone limit)",
            id="limited_ball",
        ),
        pytest.param(
            TWO_HINGES.format(extra='<equality><weld name="w" body1="b"/></equality>'),
            "equality w is a mjEQ_WELD constraint",
            id="weld",
        ),
        pytest.param(
            TWO_HINGES.format(extra='<equality><connect body1="b" anchor="0 0 0"/></equality>'),
            "equality #0 is a mjEQ_CONNECT constraint",
            id="connect",
        ),
        pytest.param(
            TWO_HINGES.format(
                extra='<equality><joint name="q" joint1="h2" joint2="h1" polycoef="0 1 0.5 0 0"/></equality>'
            ),
            "joint equality q is not an active linear coupling of two joints",
            id="quadratic_joint_equality",
        ),
        pytest.param(
            TWO_HINGES.format(extra='<equality><joint name="one" joint1="h2"/></equality>'),
            "joint equality one is not an active linear coupling of two joints",
            id="one_joint_equality",
        ),
        pytest.param(
            TWO_HINGES.format(extra='<equality><joint name="off" joint1="h2" joint2="h1" active="false"/></equality>'),
            "joint equality off is not an active linear coupling of two joints",
            id="inactive_joint_equality",
        ),
    ],
)
def test_what_the_tree_cannot_express_raises(tmp_path: Path, text: str, message: str) -> None:
    path = _write(tmp_path, text)
    with pytest.raises(ModelError, match=re.escape(f"{path}: {message}")):
        load(path)


def test_out_of_scope_elements_are_ignored_by_rule(tmp_path: Path) -> None:
    """Actuators, tendons, sensors, contact settings, springs and frictionloss leave the model as without them."""
    plain = load(_write(tmp_path, TWO_HINGES.format(extra="")), dtype=torch.float64)
    extra = """<tendon><fixed name="t"><joint joint="h1" coef="1"/><joint joint="h2" coef="-1"/></fixed></tendon>
<actuator><motor joint="h1" gear="50"/><position joint="h2" kp="10"/></actuator>
<sensor><jointpos joint="h1"/></sensor><contact><exclude body1="a" body2="b"/></contact>"""
    text = TWO_HINGES.format(extra=extra).replace(
        'name="h1" type="hinge"', 'name="h1" type="hinge" stiffness="10" springref="0.5" frictionloss="0.2"'
    )
    text = text.replace('<geom size="0.1"/>', '<geom size="0.1" friction="0.3" condim="1" margin="0.01"/>')
    rich = load(_write(tmp_path, text), dtype=torch.float64)
    assert rich.structure_key == plain.structure_key
    for field in ("joint_placements", "body_inertias", "lower_position_limit", "armature", "damping", "q_neutral"):
        assert torch.equal(getattr(rich, field), getattr(plain, field)), field
