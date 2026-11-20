"""`kinematics.compiled`: results equal to eager, one graph family per key, the library's own errors, kept results."""

from __future__ import annotations

import dataclasses
import threading
from typing import NamedTuple

import pytest
import torch
from torch._dynamo.utils import counters

from better_robot import JointFreeFlyer, load
from better_robot.dynamics import center_of_mass, forward_dynamics, inverse_dynamics, mass_matrix
from better_robot.exceptions import ArgumentError
from better_robot.io import build_model
from better_robot.io.builder import ModelBuilder
from better_robot.kinematics import KinematicState, compiled, forward_kinematics, frame_jacobian
from better_robot.kinematics.compilation import _map_tensors
from tests.support.robots import panda_urdf


def _free_flyer():
    builder = ModelBuilder("free_flyer_entry_point")
    builder.add_body("base", mass=1.0)
    builder.add_body("arm", mass=0.5)
    builder.add_free_flyer_root("floating", child="base", origin=torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]))
    builder.add_revolute("elbow", parent="base", child="arm", axis=(0.0, 0.0, 1.0), lower=-2.0, upper=2.0)
    return build_model(builder.finalize())


def _configurations(model, *batch: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(0)
    count = int(torch.Size(batch).numel())
    return torch.stack([model.random_configuration(generator=generator) for _ in range(count)]).view(*batch, -1)


@dataclasses.dataclass(frozen=True)
class _Pair:
    state: KinematicState
    tau: torch.Tensor


class _Named(NamedTuple):
    states: tuple
    extra: dict


def test_a_cold_model_compiles_one_graph_equal_to_eager():
    """The first call runs eagerly and builds the plans and memos, so the second traces them as inputs: one graph,
    no recompile (a plain ``torch.compile`` of a cold model traces the plan building, then recompiles)."""
    model = _free_flyer()
    q = _configurations(model, 3)
    fk = compiled(lambda q: forward_kinematics(model, q).joint_transforms)
    torch._dynamo.reset()
    counters.clear()
    with torch.no_grad():
        results = [fk(q) for _ in range(3)]
        expected = forward_kinematics(model, q).joint_transforms
    assert counters["stats"]["unique_graphs"] == 1
    for result in results:
        torch.testing.assert_close(result, expected)


def _panda_fixed_base(q: torch.Tensor, v: torch.Tensor, tau: torch.Tensor, model) -> dict[str, torch.Tensor]:
    state = forward_kinematics(model, q)
    return {
        "frame_jacobian": frame_jacobian(model, state, frame=model.frame_names[-1]),
        "jacobian_columns": state.jacobian_columns,
        "center_of_mass": center_of_mass(model, state),
        "forward_dynamics": forward_dynamics(model, state, v=v, tau=tau),
    }


def _floating_base_dynamics(q: torch.Tensor, v: torch.Tensor, tau: torch.Tensor, model) -> dict[str, torch.Tensor]:
    state = forward_kinematics(model, q)
    return {
        "inverse_dynamics": inverse_dynamics(model, state, v=v, a=tau),
        "mass_matrix": mass_matrix(model, state),
        "forward_dynamics": forward_dynamics(model, state, v=v, tau=tau),
        "frame_jacobian": frame_jacobian(model, state, frame=model.frame_names[-1], reference="local"),
    }


def _batched_model(q: torch.Tensor, v: torch.Tensor, tau: torch.Tensor, model) -> dict[str, torch.Tensor]:
    state = forward_kinematics(model, q)
    return {
        "joint_transforms": state.joint_transforms,
        "inverse_dynamics": inverse_dynamics(model, state, v=v, a=tau),
        "mass_matrix": mass_matrix(model, state),
    }


def _panda(case: str):
    if case == "panda float64":
        return load(panda_urdf(), dtype=torch.float64), _panda_fixed_base, (5,)
    if case == "floating base float32":
        return load(panda_urdf(), dtype=torch.float32, root_joint=JointFreeFlyer()), _floating_base_dynamics, (2, 3)
    model = load(panda_urdf(), dtype=torch.float64)
    scales = torch.linspace(0.5, 2.0, 4, dtype=torch.float64)[:, None, None]
    return model.with_values(body_inertias=model.body_inertias * scales), _batched_model, (4,)


@pytest.mark.parametrize("case", ["panda float64", "floating base float32", "batched model float64"])
def test_the_supported_calls_compile_equal_to_eager(case: str):
    """Every call the docs say `compiled` supports, compiled against eager: the frame Jacobian, the Jacobian table, the
    centre of mass, inverse and forward dynamics and the mass matrix, in float32 and float64, on a floating base and
    on a model whose values carry a batch. The kernels take their multiply-and-sum forms only when compiled."""
    model, function, batch = _panda(case)
    q = _configurations(model, *batch)
    generator = torch.Generator().manual_seed(1)
    v, tau = (torch.randn(*batch, model.nv, generator=generator, dtype=model.dtype) for _ in range(2))
    step = compiled(lambda q, v, tau: function(q, v, tau, model))
    torch._dynamo.reset()
    counters.clear()
    with torch.no_grad():
        results = [step(q, v, tau) for _ in range(3)]  # eager, then compiled, then the graph again
        expected = function(q, v, tau, model)
    assert counters["stats"]["unique_graphs"] == 1
    for result in results[1:]:
        for name, reference in expected.items():
            if model.dtype == torch.float32:
                # Fused kernels round in another order: float32 results differ from eager by about 1e-7 of their
                # largest entry (a 177 N base weight: 3e-5 N), forward dynamics by up to 1e-4.
                scale = 1e-3 if name == "forward_dynamics" else 1e-5
                tolerance = {"rtol": 0.0, "atol": scale * float(reference.abs().max())}
            else:
                tolerance = {}
            torch.testing.assert_close(
                result[name], reference, **tolerance, msg=lambda text, name=name: f"{name}: {text}"
            )


def test_a_warmed_call_raises_the_library_error_classes():
    """A library error raised inside the graph reaches the caller as its own class, as eagerly, on every call."""
    model = _free_flyer()
    fk = compiled(lambda q: forward_kinematics(model, q).joint_transforms)
    with torch.no_grad():
        fk(_configurations(model, 4))
        fk(_configurations(model, 4))
        for bad in (torch.zeros(4, model.nq + 1), _configurations(model, 4).double(), [0.0] * model.nq):
            with pytest.raises(Exception) as eager:
                forward_kinematics(model, bad)
            for _ in range(2):
                with pytest.raises(type(eager.value)):
                    fk(bad)


def test_each_key_has_its_own_budget_of_graphs():
    """Torch keeps 8 graphs per compiled function and ``fullgraph=True`` makes a ninth an error; nine non-tensor
    values, three grad modes by two dtypes by three batch sizes, all compile without one."""
    by_name = compiled(lambda x, name: x * len(name))
    sine = compiled(lambda x: x.sin())
    x = torch.linspace(0.0, 1.0, 3)
    with torch.no_grad():
        for name in ("a", "bb", "ccc", "dddd", "eeeee", "ffffff", "ggggggg", "hhhhhhhh", "iiiiiiiii"):
            for _ in range(2):
                torch.testing.assert_close(by_name(x, name), x * len(name))
    for dtype in (torch.float32, torch.float64):
        for mode in (torch.no_grad, torch.enable_grad, torch.inference_mode):
            for size in (2, 3, 1):
                with mode():
                    values = torch.linspace(0.0, 1.0, size, dtype=dtype)
                    for _ in range(2):
                        torch.testing.assert_close(sine(values), values.sin())


def test_threads_making_the_first_call_at_once_compile_one_graph():
    """The calls that may compile run one at a time: threads calling a new key at once used to compile a graph each
    and run out of graphs."""
    double = compiled(lambda x: x * 2.0)
    x = torch.linspace(0.0, 1.0, 5)
    barrier, results = threading.Barrier(12), []

    def work() -> None:
        barrier.wait()
        with torch.no_grad():
            results.extend(double(x) for _ in range(3))

    torch._dynamo.reset()
    counters.clear()
    threads = [threading.Thread(target=work) for _ in range(12)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert len(results) == 36
    assert counters["stats"]["unique_graphs"] == 1
    for result in results:
        torch.testing.assert_close(result, x * 2.0)


def test_a_side_effect_runs_once_per_call():
    calls = []

    def counted(x: torch.Tensor) -> torch.Tensor:
        calls.append(None)
        return x + 1.0

    step = compiled(counted)
    for _ in range(3):
        step(torch.zeros(2))
    assert len(calls) == 3


def test_the_result_traversal_reaches_every_tensor_and_keeps_the_model():
    """The clone of a CUDA-graph result (`_owned`) visits every tensor at any depth of tuples, lists, dicts and
    dataclasses, and keeps a model as the caller's object."""
    model = _free_flyer()
    state = forward_kinematics(model, _configurations(model, 2))
    result = _Named(
        states=(state, [_Pair(state=state, tau=torch.ones(2))]), extra={"pair": _Pair(state, torch.ones(3))}
    )
    mapped = _map_tensors(lambda tensor: tensor + 1.0, result)
    nested = mapped.states[1][0]
    for kept, original in ((mapped.states[0], state), (nested.state, state), (mapped.extra["pair"].state, state)):
        assert kept.model is model
        torch.testing.assert_close(kept.joint_transforms, original.joint_transforms + 1.0)
        torch.testing.assert_close(kept.q, original.q + 1.0)
    torch.testing.assert_close(nested.tau, torch.full((2,), 2.0))
    torch.testing.assert_close(mapped.extra["pair"].tau, torch.full((3,), 2.0))


def test_an_unknown_mode_raises():
    with pytest.raises(ArgumentError, match="mode"):
        compiled(lambda q: q, mode="max-autotune")


@pytest.mark.cuda
# Inductor advises TF32 matmuls, a process-wide setting the library leaves to the caller; its CUDA-graph tree
# captures an empty graph on purpose to keep its memory pool alive, a warning it records but "error" raises.
@pytest.mark.filterwarnings("ignore:TensorFloat32 tensor cores:UserWarning")
@pytest.mark.filterwarnings("ignore:The CUDA Graph is empty:UserWarning")
def test_reduce_overhead_captures_forward_dynamics_and_keeps_each_result():
    """CUDA graphs capture forward dynamics (no host sync), and a result stays valid after the next replay, which
    overwrites the graph's own outputs without an error: plain tensors, and the tensors of a dataclass holding a
    state and of a tuple of states, whose model stays the caller's."""
    model = load(panda_urdf(), dtype=torch.float32)
    q = _configurations(model, 64).cuda()
    model = model.to(device=torch.device("cuda"))
    generator = torch.Generator().manual_seed(0)
    v, tau = (torch.randn(64, model.nv, generator=generator).cuda() for _ in range(2))
    aba = compiled(
        lambda q, v, tau: forward_dynamics(model, forward_kinematics(model, q), v=v, tau=tau), mode="reduce-overhead"
    )
    pair = compiled(
        lambda q, v: _Pair(
            forward_kinematics(model, q), inverse_dynamics(model, forward_kinematics(model, q), v=v, a=v)
        ),
        mode="reduce-overhead",
    )
    states = compiled(
        lambda q: (forward_kinematics(model, q), forward_kinematics(model, 2.0 * q)), mode="reduce-overhead"
    )
    torch._dynamo.reset()
    counters.clear()
    scales = (1.0, 2.0, 3.0, 4.0)
    with torch.no_grad():
        accelerations = [aba(q, v, scale * tau) for scale in scales]
        pairs = [pair(scale * q, v) for scale in scales]
        tuples = [states(scale * q) for scale in scales]
        expected = [forward_dynamics(model, forward_kinematics(model, q), v=v, tau=scale * tau) for scale in scales]
        expected_pairs = [inverse_dynamics(model, forward_kinematics(model, scale * q), v=v, a=v) for scale in scales]
    assert counters["inductor"]["cudagraph_skips"] == 0
    for result, reference in zip(accelerations, expected, strict=True):
        # float32 ABA: compiled and eager differ by up to 1e-4 of the largest acceleration; an overwritten result by 1.
        torch.testing.assert_close(result, reference, rtol=0.0, atol=1e-3 * float(reference.abs().max()))
    for result, reference, scale in zip(pairs, expected_pairs, scales, strict=True):
        assert result.state.model is model
        reference_transforms = forward_kinematics(model, scale * q).joint_transforms
        torch.testing.assert_close(result.state.joint_transforms, reference_transforms, rtol=0.0, atol=1e-5)
        torch.testing.assert_close(result.tau, reference, rtol=0.0, atol=1e-4 * float(reference.abs().max()))
    for (first, second), scale in zip(tuples, scales, strict=True):
        assert first.model is model and second.model is model
        reference_transforms = forward_kinematics(model, scale * q).joint_transforms
        torch.testing.assert_close(first.joint_transforms, reference_transforms, rtol=0.0, atol=1e-5)
        torch.testing.assert_close(
            second.joint_transforms, forward_kinematics(model, 2.0 * scale * q).joint_transforms, rtol=0.0, atol=1e-5
        )
