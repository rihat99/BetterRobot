"""``repr(model)`` is one short line (decision record 0009): the name, the sizes, the dtype and the device."""

from __future__ import annotations

from better_robot import load
from tests.support.robots import panda_urdf


def test_panda_repr_is_one_short_line() -> None:
    model = load(panda_urdf())
    text = repr(model)
    assert "\n" not in text and len(text) < 200
    assert text.startswith("Model(") and repr(model.name) in text
    assert f"nq={model.nq}" in text and f"nv={model.nv}" in text and f"njoints={model.njoints}" in text
    assert str(model.dtype).removeprefix("torch.") in text and str(model.device) in text
