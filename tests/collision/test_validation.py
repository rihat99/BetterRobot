"""Every public collision function validates its inputs' types, shapes, dtypes and devices (C-ERR-1, C-ERR-2): a
typed error naming the fix, never a silent promotion or a raw ``RuntimeError``."""

from __future__ import annotations

import pytest
import torch

import better_robot as br
from better_robot import collision, kinematics
from better_robot.exceptions import ArgumentError, DeviceMismatchError, DtypeMismatchError, ShapeError
from tests.support.boundary import check_boundary
from tests.support.cases import failures
from tests.support.robots import panda_urdf

#: Each distance function with the trailing shape of each tensor parameter, in call order, and its other arguments
#: (``sphere_sdf``'s grid has no batch axes and its origin is one point, so neither takes the default batch).
DISTANCES = {
    "sphere_sphere": ({"center_a": (3,), "radius_a": (), "center_b": (3,), "radius_b": ()}, {}),
    "sphere_capsule": ({"center": (3,), "radius": (), "start": (3,), "end": (3,), "capsule_radius": ()}, {}),
    "capsule_capsule": (
        {"start_a": (3,), "end_a": (3,), "radius_a": (), "start_b": (3,), "end_b": (3,), "radius_b": ()},
        {},
    ),
    "sphere_halfspace": ({"center": (3,), "radius": (), "normal": (3,)}, {"offset": 0.0}),
    "sphere_box": (
        {"center": (3,), "radius": (), "box_center": (3,), "box_rotation": (3, 3), "half_extents": (3,)},
        {},
    ),
    "sphere_sdf": (
        {"center": (3,), "radius": (), "origin": (3,)},
        {"origin": torch.zeros(3), "grid": torch.zeros(3, 3, 3), "voxel_size": 0.1},
    ),
}


def test_distance_functions_reject_bad_inputs() -> None:
    def check(name: str) -> None:
        shapes, others = DISTANCES[name]
        check_boundary(getattr(collision, name), shapes, **others)

    failed = failures(check, DISTANCES)
    assert not failed, "\n".join(failed)


def test_sphere_sphere_rejects_a_four_vector() -> None:
    """A ``(4,)`` centre is a shape error, never a distance."""
    with pytest.raises(ShapeError, match=r"center_a must have shape \(\.\.\., 3\), got \(4,\)"):
        collision.sphere_sphere(torch.zeros(4), torch.tensor(1.0), center_b=torch.ones(4), radius_b=torch.tensor(1.0))


def test_sphere_sdf_needs_a_three_axis_grid() -> None:
    with pytest.raises(ShapeError, match=r"grid must have shape \(X, Y, Z\)"):
        collision.sphere_sdf(
            torch.zeros(3), torch.zeros(()), grid=torch.zeros(3, 3), origin=torch.zeros(3), voxel_size=0.1
        )


@pytest.fixture(scope="module")
def panda() -> br.model.Model:
    return br.load(panda_urdf(), dtype=torch.float64)


def _robots(model: br.model.Model) -> tuple[collision.RobotSpheres, collision.RobotCapsules]:
    spheres = collision.RobotSpheres.from_frames(
        model, {"panda_link3": [((0.0, 0.0, 0.0), 0.08)], "panda_hand": [((0.0, 0.0, 0.05), 0.06)]}
    )
    capsules = collision.RobotCapsules.from_frames(
        model, {"panda_link3": [((0.0, 0.0, 0.0), (0.0, 0.0, 0.1), 0.08)], "panda_hand": [((0, 0, 0), (0, 0, 0), 0.06)]}
    )
    return spheres, capsules


def test_robot_geometry_must_match_the_state_dtype_and_device(panda: br.model.Model) -> None:
    """Spheres from a float64 model with a float32 state raise instead of computing in float64."""
    spheres, capsules = _robots(panda)
    state32 = kinematics.forward_kinematics(panda.to(dtype=torch.float32), panda.q_neutral.float())
    pairs = torch.tensor([[0, 1]])
    with pytest.raises(DtypeMismatchError, match=r"spheres\.dtype=torch\.float64 != state\.dtype=torch\.float32"):
        collision.self_distances(spheres, state32, pairs=pairs)
    with pytest.raises(DtypeMismatchError, match=r"capsules\.dtype=torch\.float64 != state\.dtype=torch\.float32"):
        collision.self_capsule_distances(capsules, state32, pairs=pairs)
    state = kinematics.forward_kinematics(panda, panda.q_neutral)
    with pytest.raises(DeviceMismatchError, match=r"spheres\.device=meta != state\.device=cpu"):
        spheres.to(device="meta").centers(state)
    with pytest.raises(TypeError, match="state must be a KinematicState"):
        spheres.centers(panda.q_neutral)


def test_pairs_must_be_a_long_p_by_2_tensor(panda: br.model.Model) -> None:
    spheres, capsules = _robots(panda)
    state = kinematics.forward_kinematics(panda, panda.q_neutral)
    for function, geometry in ((collision.self_distances, spheres), (collision.self_capsule_distances, capsules)):
        assert function(geometry, state, pairs=torch.tensor([[0, 1]])).shape == (1,)
        with pytest.raises(DtypeMismatchError, match=r"pairs\.dtype=torch\.float32 is not torch\.int64"):
            function(geometry, state, pairs=torch.tensor([[0.0, 1.0]]))
        with pytest.raises(ShapeError, match=r"pairs must have shape \(P, 2\), got \(2,\)"):
            function(geometry, state, pairs=torch.tensor([0, 1]))
        with pytest.raises(TypeError, match="pairs must be a torch.Tensor, got list"):
            function(geometry, state, pairs=[[0, 1]])


def test_pairs_reject_spheres_of_another_model(panda: br.model.Model) -> None:
    spheres = collision.RobotSpheres(
        frame_ids=torch.tensor([0, panda.nframes]),
        offsets=torch.zeros(2, 3, dtype=torch.float64),
        radii=torch.ones(2, dtype=torch.float64),
    )
    with pytest.raises(ArgumentError, match=rf"frame_ids must lie in \[0, {panda.nframes}\)"):
        collision.self_collision_pairs(panda, spheres)


def test_robot_spheres_check_their_tables_at_construction() -> None:
    """A bad table raises at construction, not later with a raw error: float or batched ids, (2, 2) offsets, 3 radii
    for 2 ids, mixed dtypes or devices."""
    ids, offsets, radii = torch.tensor([1, 2]), torch.zeros(2, 3), torch.ones(2)
    bad_tables = [
        ({"frame_ids": ids.float()}, DtypeMismatchError, r"frame_ids\.dtype=torch\.float32 is not torch\.int64"),
        ({"frame_ids": ids[None]}, ShapeError, r"frame_ids must have shape \(S,\)"),
        ({"frame_ids": [1, 2]}, TypeError, "frame_ids must be a torch.Tensor, got list"),
        ({"offsets": torch.zeros(2, 2)}, ShapeError, r"offsets must have shape \(\.\.\., 2, 3\)"),
        ({"radii": torch.ones(3)}, ShapeError, r"radii must have shape \(\.\.\., 2\)"),
        ({"radii": torch.ones(4, 2)}, ShapeError, r"radii must have shape \(2,\)"),
        ({"radii": radii.double()}, DtypeMismatchError, r"radii\.dtype=torch\.float64 != offsets\.dtype"),
        ({"offsets": torch.zeros(2, 3, device="meta")}, DeviceMismatchError, r"offsets\.device=meta != frame_ids"),
    ]
    for change, error, match in bad_tables:
        with pytest.raises(error, match=match):
            collision.RobotSpheres(**{"frame_ids": ids, "offsets": offsets, "radii": radii, **change})


def test_self_distances_name_indices_outside_their_tables(panda: br.model.Model) -> None:
    """Out-of-range pairs or frame ids raise ``ArgumentError`` with the fix, not torch's ``IndexError``."""
    spheres, _ = _robots(panda)
    state = kinematics.forward_kinematics(panda, panda.q_neutral)
    with pytest.raises(ArgumentError, match=r"pairs holds a sphere index outside \[0, 2\)"):
        collision.self_distances(spheres, state, pairs=torch.tensor([[0, 7]]))
    foreign = collision.RobotSpheres(
        frame_ids=torch.tensor([0, panda.nframes + 40]), offsets=spheres.offsets, radii=spheres.radii
    )
    with pytest.raises(ArgumentError, match=rf"frame id outside \[0, {panda.nframes}\) of the state's model"):
        collision.self_distances(foreign, state, pairs=torch.tensor([[0, 1]]))


def test_full_configuration_validates_q() -> None:
    """A float64 ``q`` on a float32 model, a ``(5,)`` ``q`` and a list raise typed errors, never a float64 result, a
    raw ``RuntimeError`` or an ``AttributeError``."""
    model = br.load(panda_urdf())
    with pytest.raises(DtypeMismatchError, match=r"q\.dtype=torch\.float64 != model\.dtype=torch\.float32"):
        kinematics.full_configuration(model, model.q_neutral.double())
    with pytest.raises(ShapeError, match=rf"q must have shape \(\.\.\., {model.nq}\), got \(5,\)"):
        kinematics.full_configuration(model, torch.zeros(5))
    with pytest.raises(TypeError, match="q must be a torch.Tensor, got list"):
        kinematics.full_configuration(model, model.q_neutral.tolist())
