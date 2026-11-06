"""One error tree under ``BetterRobotError`` (decision record 0020, C-ERR-2).

Every class in ``exceptions.__all__`` derives from ``BetterRobotError`` and from the builtin the record names, is
raised somewhere in ``src`` (a class nobody raises is deleted), the boundary guards raise the typed class, and
``src`` raises a builtin ``ValueError``, ``TypeError`` or ``RuntimeError`` only where ``PLAIN_RAISES`` (permanent,
reasoned) or the ratchet ``tests/contract/ratchets/plain_raises.json`` (shrinks as the io, optim and viewer rewrites land) allows.
"""

from __future__ import annotations

import ast
import json
import math
import sys
from pathlib import Path

import pytest
import torch

from better_robot import exceptions as exc
from better_robot.io import ModelBuilder, build_model, load
from better_robot.io.build_model import _check_topology_invariants
from better_robot.kinematics import forward_kinematics

REPO = Path(__file__).resolve().parents[2]
SRC = REPO / "src" / "better_robot"
PLAIN_RAISES_RATCHET = REPO / "tests" / "contract" / "ratchets" / "plain_raises.json"
BUILTIN_RAISES = {"ValueError", "TypeError", "RuntimeError"}
_NOT_A_TENSOR = "a value that is not a torch.Tensor is a Python TypeError; the tree has no node for it"
PLAIN_RAISES: dict[str, str] = {
    "src/better_robot/lie/_checks.py::check_tensor": _NOT_A_TENSOR,
    "src/better_robot/lie/umeyama.py::_validate_point_sets": _NOT_A_TENSOR,
    "src/better_robot/lie/umeyama.py::_normalized_fit_weights": _NOT_A_TENSOR,
    "src/better_robot/model/_checks.py::check_tensor": _NOT_A_TENSOR,
    "src/better_robot/spatial/_checks.py::check_tensors": _NOT_A_TENSOR,
    "src/better_robot/spatial/inertia.py::_check_mesh_structure": _NOT_A_TENSOR,
    "src/better_robot/spatial/inertia.py::_density_tensor": _NOT_A_TENSOR,
    "src/better_robot/spatial/mesh.py::_check_face_table": _NOT_A_TENSOR,
}
BASES: dict[type, type] = {
    exc.ShapeError: ValueError,
    exc.DtypeMismatchError: ValueError,
    exc.DeviceMismatchError: ValueError,
    exc.ArgumentError: ValueError,
    exc.ModelError: ValueError,
    exc.QuaternionNormError: ValueError,
    exc.UnknownNameError: KeyError,
    exc.OptionalDependencyError: ImportError,
}


def _raised_names() -> set[str]:
    """Class names in ``raise Name(...)`` and ``raise module.Name(...)`` statements under ``src``."""
    names: set[str] = set()
    for file in SRC.rglob("*.py"):
        for node in ast.walk(ast.parse(file.read_text())):
            if isinstance(node, ast.Raise) and isinstance(node.exc, ast.Call):
                callee = node.exc.func
                if isinstance(callee, ast.Name):
                    names.add(callee.id)
                elif isinstance(callee, ast.Attribute):
                    names.add(callee.attr)
    return names


def _builtin_raise_sites() -> set[str]:
    """``path::function`` of every ``raise ValueError/TypeError/RuntimeError(...)`` under ``src``."""
    sites: set[str] = set()

    def visit(node: ast.AST, file: Path, owner: str) -> None:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            owner = node.name
        if isinstance(node, ast.Raise) and isinstance(node.exc, ast.Call):
            callee = node.exc.func
            name = callee.id if isinstance(callee, ast.Name) else getattr(callee, "attr", None)
            if name in BUILTIN_RAISES:
                sites.add(f"{file.relative_to(REPO).as_posix()}::{owner}")
        for child in ast.iter_child_nodes(node):
            visit(child, file, owner)

    for file in SRC.rglob("*.py"):
        visit(ast.parse(file.read_text()), file, "<module>")
    return sites


def test_every_class_has_its_bases() -> None:
    """``__all__`` is exactly the root plus the eight classes, each under the root and its builtin."""
    assert set(exc.__all__) == {"BetterRobotError", *(cls.__name__ for cls in BASES)}
    for cls, builtin in BASES.items():
        assert issubclass(cls, exc.BetterRobotError) and issubclass(cls, builtin), cls.__name__


def test_every_class_is_raised_in_src() -> None:
    """The root is the catch-all a caller uses; every other class has a raise site."""
    missing = sorted(cls.__name__ for cls in BASES if cls.__name__ not in _raised_names())
    assert not missing, f"delete or raise: {missing}"


def test_src_raises_the_error_tree_only() -> None:
    """A builtin raise is allowed only by the permanent list or the ratchet, and every entry still has its raise."""
    sites = _builtin_raise_sites()
    ratchet = json.loads(PLAIN_RAISES_RATCHET.read_text())
    assert not set(PLAIN_RAISES) & set(ratchet), "an entry is either permanent or ratcheted"
    unexpected = sorted(sites - set(PLAIN_RAISES) - set(ratchet))
    assert unexpected == [], f"raise the error tree (exceptions.py) instead of a builtin: {unexpected}"
    stale = sorted((set(PLAIN_RAISES) | set(ratchet)) - sites)
    assert stale == [], f"drop these entries with the fix that removed the raise: {stale}"


@pytest.fixture(scope="module")
def arm_model():
    builder = ModelBuilder("arm")
    builder.add_body("base", mass=0.5)
    builder.add_body("link1", mass=1.0)
    builder.add_revolute(
        "j1", axis=(0.0, 0.0, 1.0),
        parent="base",
        child="link1",
        origin=torch.tensor([0.0, 0.0, 0.1, 0.0, 0.0, 0.0, 1.0]),
        lower=-math.pi,
        upper=math.pi,
    )
    return build_model(builder.finalize())


@pytest.fixture(scope="module")
def free_flyer_model():
    builder = ModelBuilder("floating")
    builder.add_body("base", mass=1.0)
    builder.add_body("link1", mass=0.5)
    builder.add_free_flyer_root("floating", child="base", origin=torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]))
    builder.add_revolute(
        "j1", axis=(0.0, 0.0, 1.0),
        parent="base",
        child="link1",
        origin=torch.tensor([0.0, 0.0, 0.1, 0.0, 0.0, 0.0, 1.0]),
        lower=-math.pi,
        upper=math.pi,
    )
    return build_model(builder.finalize())


@pytest.mark.parametrize("batch_shape, extra", [((), 3), ((4,), 1)], ids=["unbatched", "batched"])
def test_wrong_nq_raises_shape_error(arm_model, batch_shape, extra) -> None:
    with pytest.raises(exc.ShapeError):
        forward_kinematics(arm_model, torch.zeros(*batch_shape, arm_model.nq + extra))


def test_unknown_name_lists_the_valid_names(arm_model) -> None:
    """The message names the kind, the name and, with at most 20 candidates, every valid name."""
    with pytest.raises(
        exc.UnknownNameError, match=r"unknown joint 'nope'; valid joints: \['universe', 'root_joint', 'j1'\]"
    ):
        arm_model.joint_id("nope")
    assert str(exc.UnknownNameError("frame", "x", [str(i) for i in range(21)])) == "unknown frame 'x'; 21 frames"


@pytest.mark.cuda
def test_device_mismatch_between_a_cpu_model_and_a_cuda_configuration(arm_model) -> None:
    with pytest.raises(exc.DeviceMismatchError, match="device"):
        forward_kinematics(arm_model, torch.zeros(arm_model.nq, device="cuda"))


def test_id_out_of_range_raises_argument_error(arm_model) -> None:
    with pytest.raises(exc.ArgumentError, match="out of range"):
        arm_model.frame_id(arm_model.nframes)


def test_quaternion_norm_is_checked_only_on_request(free_flyer_model, arm_model) -> None:
    q = free_flyer_model.q_neutral.clone()
    q[3:7] = 0.0
    with pytest.raises(exc.QuaternionNormError, match="quaternion norm"):
        forward_kinematics(free_flyer_model, q, check_quaternion_norm=True)
    forward_kinematics(free_flyer_model, q)
    drifted = free_flyer_model.q_neutral.clone()
    drifted[3:7] *= 1.02
    forward_kinematics(free_flyer_model, drifted, check_quaternion_norm=True)
    forward_kinematics(arm_model, arm_model.q_neutral, check_quaternion_norm=True)


@pytest.mark.parametrize(
    ("invariants", "match"),
    [
        pytest.param(
            dict(
                parent_joint_ids=(0, 0, 1),
                nqs=(0, 0, 1),
                nvs=(0, 0, 1),
                idx_qs=(0, 0, 0),
                idx_vs=(0, 0, 0),
                nq_total=1,
                nv_total=1,
            ),
            r"parent_joint_ids\[0\]",
            id="bad_parents",
        ),
        pytest.param(
            dict(
                parent_joint_ids=(-1, 2, 0),
                nqs=(0, 1, 1),
                nvs=(0, 1, 1),
                idx_qs=(0, 0, 1),
                idx_vs=(0, 0, 1),
                nq_total=2,
                nv_total=2,
            ),
            "topological sort",
            id="out_of_order_parents",
        ),
        pytest.param(
            dict(
                parent_joint_ids=(-1, 0, 1),
                nqs=(0, 1, 1),
                nvs=(0, 1, 1),
                idx_qs=(0, 0, 2),
                idx_vs=(0, 0, 1),
                nq_total=2,
                nv_total=2,
            ),
            "q-slicing",
            id="q_slice_gap",
        ),
        pytest.param(
            dict(parent_joint_ids=(-1, 0), nqs=(0, 1), nvs=(0, 1), idx_qs=(0, 0), idx_vs=(0, 0), nq_total=5, nv_total=1),
            "nq",
            id="nq_mismatch",
        ),
    ],
)
def test_topology_invariants_raise_model_error(invariants, match) -> None:
    with pytest.raises(exc.ModelError, match=match):
        _check_topology_invariants(**invariants)


def test_missing_optional_package_names_the_install_line(monkeypatch, tmp_path) -> None:
    monkeypatch.setitem(sys.modules, "mujoco", None)
    with pytest.raises(exc.OptionalDependencyError, match=r"mujoco is not installed; run: uv sync --extra mjcf"):
        load(tmp_path / "scene.xml")
