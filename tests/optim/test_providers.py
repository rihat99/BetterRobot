"""Provider memoization, validation, and evaluation-lifetime contracts."""

from __future__ import annotations

import gc
import math
import weakref
from collections.abc import Mapping
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import pytest
import torch

from better_robot.io.build_model import build_model
from better_robot.io.parsers.programmatic import ModelBuilder
from better_robot.optim import (
    Problem,
    ResidualItem,
    RobotStateProvider,
    VarSpec,
    detach_values,
)


@dataclass
class _CountingProvider:
    name: str = "shared"
    reads: tuple[str, ...] = ("x",)
    outputs: tuple[str, ...] = ("feature",)
    multiplier: float = 2.0
    calls: int = 0

    def __call__(self, ctx: Mapping[str, Any]) -> dict[str, Any]:
        self.calls += 1
        return {self.outputs[0]: ctx[self.reads[0]] * self.multiplier}


@dataclass(frozen=True)
class _ScaleProvider:
    name: str
    reads: tuple[str, ...]
    outputs: tuple[str, ...]
    multiplier: float = 1.0

    def __call__(self, ctx: Mapping[str, Any]) -> dict[str, Any]:
        return {self.outputs[0]: ctx[self.reads[0]] * self.multiplier}


@dataclass(frozen=True)
class _UndeclaredProviderRead:
    name: str = "bad_provider"
    reads: tuple[str, ...] = ("x",)
    outputs: tuple[str, ...] = ("feature",)

    def __call__(self, ctx: Mapping[str, Any]) -> dict[str, Any]:
        return {"feature": ctx["unrelated"]}


@dataclass(frozen=True)
class _TensorResidual:
    name: str
    reads: tuple[str, ...]
    dim: int = 1

    def __call__(self, ctx: Mapping[str, Any]) -> torch.Tensor:
        return ctx[self.reads[0]][..., : self.dim]


@dataclass(frozen=True)
class _DataResidual:
    name: str
    reads: tuple[str, ...] = ("data",)
    dim: int = 1

    def __call__(self, ctx: Mapping[str, Any]) -> torch.Tensor:
        return ctx["data"].q[..., :1]


@dataclass(frozen=True)
class _CapturingResidual:
    holder: dict[str, Any]
    name: str = "capture"
    reads: tuple[str, ...] = ("x",)
    dim: int = 1

    def __call__(self, ctx: Mapping[str, Any]) -> torch.Tensor:
        self.holder["context"] = weakref.ref(ctx)
        return ctx["x"]


def _shared_problem(provider: _CountingProvider) -> Problem:
    residuals = tuple(
        ResidualItem(
            name=f"consumer_{index}",
            residual=_TensorResidual(f"consumer_{index}", ("feature",)),
        )
        for index in range(3)
    )
    return Problem(
        vars=(VarSpec("x", (1,)),),
        residuals=residuals,
        providers=(provider,),
    )


def test_provider_runs_once_per_evaluation_across_three_residuals() -> None:
    provider = _CountingProvider()
    problem = _shared_problem(provider)
    values = {"x": torch.tensor([1.0])}

    torch.testing.assert_close(problem.residual(values), torch.tensor([2.0, 2.0, 2.0]))
    assert provider.calls == 1

    # A new public evaluation gets a fresh context rather than reusing a
    # values-version or object-identity cache.
    problem.residual(values)
    assert provider.calls == 2

    gradient = problem.gradient(values)
    assert provider.calls == 3
    torch.testing.assert_close(gradient["x"], torch.tensor([12.0]))


def test_python_zero_weight_keeps_lazy_provider_inactive() -> None:
    provider = _CountingProvider()
    problem = Problem(
        vars=(VarSpec("x", (1,)),),
        residuals=(ResidualItem("consumer", _TensorResidual("consumer", ("feature",))),),
        providers=(provider,),
    )
    values = {"x": torch.tensor([1.0])}

    residual = problem.residual(values, weights={"consumer": 0.0})
    gradient = problem.gradient(values, weights={"consumer": 0.0})
    torch.testing.assert_close(residual, torch.zeros(1))
    torch.testing.assert_close(gradient["x"], torch.zeros(1))
    assert provider.calls == 0

    # A tensor zero stays graph-visible and therefore is not an inactive-set
    # declaration.
    problem.residual(values, weights={"consumer": torch.tensor(0.0)})
    assert provider.calls == 1


def test_provider_cycle_error_is_deterministic_and_exact() -> None:
    providers = (
        _ScaleProvider("nn_pass", ("sdf",), ("nn",)),
        _ScaleProvider("sdf", ("nn",), ("sdf",)),
    )
    for _ in range(2):
        with pytest.raises(ValueError, match="provider dependency cycle involving 'nn_pass'"):
            Problem(vars=(VarSpec("x", (1,)),), providers=providers)


def test_problem_rejects_unknown_provider_reads_and_item_reads() -> None:
    with pytest.raises(
        ValueError,
        match=r"Provider 'producer' declares unknown reads \['missing'\]",
    ):
        Problem(
            vars=(VarSpec("x", (1,)),),
            providers=(_ScaleProvider("producer", ("missing",), ("feature",)),),
        )

    with pytest.raises(
        ValueError,
        match=r"Item 'consumer' declares unknown reads \['missing'\]",
    ):
        Problem(
            vars=(VarSpec("x", (1,)),),
            residuals=(ResidualItem("consumer", _TensorResidual("consumer", ("missing",))),),
        )


def test_provider_may_read_an_undeclared_dependency() -> None:
    problem = Problem(
        vars=(VarSpec("x", (1,)), VarSpec("unrelated", (1,))),
        residuals=(ResidualItem("consumer", _TensorResidual("consumer", ("feature",))),),
        providers=(_UndeclaredProviderRead(),),
    )

    residual = problem.residual({"x": torch.ones(1), "unrelated": torch.tensor([3.0])})

    torch.testing.assert_close(residual, torch.tensor([3.0]))


def test_problem_rejects_context_and_item_name_collisions() -> None:
    spec = VarSpec("x", (1,))

    with pytest.raises(ValueError, match=r"variable/parameter names collide: \['x'\]"):
        Problem(vars=(spec,), parameters={"x": torch.tensor(1.0)})

    with pytest.raises(ValueError, match="provider output 'x' collides"):
        Problem(
            vars=(spec,),
            providers=(_ScaleProvider("producer", ("x",), ("x",)),),
        )

    with pytest.raises(ValueError, match="provider output 'feature' collides"):
        Problem(
            vars=(spec,),
            providers=(
                _ScaleProvider("first", ("x",), ("feature",)),
                _ScaleProvider("second", ("x",), ("feature",)),
            ),
        )

    with pytest.raises(ValueError, match="duplicate provider name 'producer'"):
        Problem(
            vars=(spec,),
            providers=(
                _ScaleProvider("producer", ("x",), ("first",)),
                _ScaleProvider("producer", ("x",), ("second",)),
            ),
        )


def _one_joint_model():
    builder = ModelBuilder("provider_arm")
    builder.add_body("base", mass=0.5)
    builder.add_body("link", mass=1.0)
    builder.add_revolute_z(
        "joint",
        parent="base",
        child="link",
        origin=torch.tensor([0.0, 0.0, 0.1, 0.0, 0.0, 0.0, 1.0]),
        lower=-math.pi,
        upper=math.pi,
    )
    return build_model(builder.finalize())


def test_robot_state_provider_runs_fk_once_per_evaluation(monkeypatch: pytest.MonkeyPatch) -> None:
    model = _one_joint_model()
    calls: list[torch.Tensor] = []

    def fake_forward_kinematics(model_arg, q: torch.Tensor, *, compute_frames: bool):
        assert model_arg is model
        assert compute_frames is True
        calls.append(q)
        return SimpleNamespace(q=q)

    monkeypatch.setattr(
        "better_robot.optim.providers.forward_kinematics",
        fake_forward_kinematics,
    )
    problem = Problem(
        vars=(VarSpec("q", (model.nq,)),),
        residuals=tuple(ResidualItem(f"consumer_{index}", _DataResidual(f"consumer_{index}")) for index in range(3)),
        providers=(RobotStateProvider(model),),
    )
    values = {"q": model.q_neutral.clone()}

    problem.residual(values)
    assert len(calls) == 1
    problem.gradient(values)
    assert len(calls) == 2


def test_evaluation_context_does_not_escape_problem_or_residual() -> None:
    holder: dict[str, Any] = {}

    def evaluate_once():
        residual = _CapturingResidual(holder)
        problem = Problem(
            vars=(VarSpec("x", (1,)),),
            residuals=(ResidualItem("capture", residual),),
        )
        problem_ref = weakref.ref(problem)
        residual_ref = weakref.ref(residual)
        torch.testing.assert_close(problem.residual({"x": torch.tensor([1.0])}), torch.ones(1))
        return problem_ref, residual_ref

    problem_ref, residual_ref = evaluate_once()
    gc.collect()

    assert holder["context"]() is None
    assert problem_ref() is None
    assert residual_ref() is None


def test_detach_values_returns_graph_free_artifacts() -> None:
    leaf = torch.tensor([2.0], requires_grad=True)
    values = {"x": leaf.square()}

    accepted = detach_values(values)

    assert accepted is not values
    assert accepted["x"] is not values["x"]
    assert accepted["x"].grad_fn is None
    assert accepted["x"].requires_grad is False
    assert values["x"].grad_fn is not None
    torch.testing.assert_close(accepted["x"], torch.tensor([4.0]))


def test_transitive_provider_dependencies_drive_jacobian_structure() -> None:
    problem = Problem(
        vars=(VarSpec("x", (2,)), VarSpec("unrelated", (1,))),
        residuals=(ResidualItem("consumer", _TensorResidual("consumer", ("feature",), dim=2)),),
        providers=(
            _ScaleProvider("encode", ("x",), ("encoded",), multiplier=2.0),
            _ScaleProvider("features", ("encoded",), ("feature",), multiplier=3.0),
        ),
    )
    values = {
        "x": torch.tensor([1.0, -2.0]),
        "unrelated": torch.tensor([4.0]),
    }

    blocks = problem.jacobian_blocks(values, strategy="jacrev")

    assert set(blocks) == {("consumer", "x")}
    torch.testing.assert_close(blocks[("consumer", "x")], 6.0 * torch.eye(2))
