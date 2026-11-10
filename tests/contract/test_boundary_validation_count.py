"""Public passes validate each tensor input once, at their boundary, and trust the attached model (record 0021).

Every entry point of ``_public_ops`` passes each floating tensor argument to ``check_tensor`` (alone or through
``check_tensors``) exactly once, under its parameter name, and nothing else: a Jacobian or dynamics call reads its
state unchecked (the state's own check is attachment, C-STATE-6), ``forward_dynamics``' bias call takes its zero
acceleration unchecked, and an integer table (``inertia_of_mesh``'s faces) is checked by its own rule. Every
``better_robot`` module that imports the validators counts, so a check moved into a helper still counts.
"""

from __future__ import annotations

import inspect
import sys
from collections import Counter

import pytest
import torch

from better_robot import dynamics, kinematics
from better_robot.lie import _checks

from ._public_ops import CASE_NAMES, build_case

_CPU = torch.device("cpu")


@pytest.fixture
def checked_names(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Names passed to ``check_tensor`` and ``check_tensors`` by any ``better_robot`` module during the test."""
    names: list[str] = []
    check_tensor, check_tensors = _checks.check_tensor, _checks.check_tensors

    def counted(name: str, value: object, **kwargs: object) -> torch.Tensor:
        names.append(name)
        return check_tensor(name, value, **kwargs)

    def counted_each(**tensors: tuple[object, tuple[int, ...]]) -> None:
        names.extend(tensors)
        check_tensors(**tensors)

    for name, module in list(sys.modules.items()):
        if name.startswith("better_robot.") and module is not _checks:
            if vars(module).get("check_tensor") is check_tensor:
                monkeypatch.setattr(module, "check_tensor", counted)
            if vars(module).get("check_tensors") is check_tensors:
                monkeypatch.setattr(module, "check_tensors", counted_each)
    return names


@pytest.mark.parametrize("name", CASE_NAMES)
def test_each_public_call_validates_its_inputs_once(name: str, checked_names: list[str]) -> None:
    fn, args, kwargs = build_case(name, torch.float64, _CPU)
    arguments = inspect.signature(fn).bind(*args, **kwargs).arguments
    inputs = [key for key, value in arguments.items() if isinstance(value, torch.Tensor) and value.is_floating_point()]
    checked_names.clear()  # building the case ran forward kinematics once
    fn(*args, **kwargs)
    assert Counter(checked_names) == Counter(inputs), f"{name} checked {checked_names}, its tensor inputs are {inputs}"


def test_dynamics_on_a_fresh_state_validates_q_once_and_its_own_inputs(checked_names: list[str]) -> None:
    _, (model, _), kwargs = build_case("inverse_dynamics", torch.float64, _CPU)
    q = torch.tensor([0.3, -0.4], dtype=torch.float64)
    checked_names.clear()
    dynamics.inverse_dynamics(model, kinematics.forward_kinematics(model, q), **kwargs)
    assert checked_names == ["q", "v", "a"]
