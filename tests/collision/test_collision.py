"""Sphere distances, robot spheres and collision terms."""

import pytest
import torch

import better_robot as br
from better_robot import collision, optim, terms
from better_robot.kinematics import forward_kinematics_state

DT = torch.float64


@pytest.fixture(scope="module")
def panda():
    pytest.importorskip("robot_descriptions")
    from robot_descriptions import panda_description  # noqa: PLC0415

    return br.load(panda_description.URDF_PATH, dtype=DT)


def _randn(*shape, seed=0):
    return torch.randn(*shape, generator=torch.Generator().manual_seed(seed), dtype=DT)


def test_closed_form_cases():
    zero, one = torch.zeros(3, dtype=DT), torch.tensor([1.0, 0.0, 0.0], dtype=DT)
    r = torch.tensor(0.3, dtype=DT)
    assert float(collision.sphere_sphere(zero, r, center_b=2 * one, radius_b=r)) == pytest.approx(1.4)
    assert float(collision.sphere_sphere(zero, r, center_b=0.5 * one, radius_b=r)) == pytest.approx(-0.1)  # penetration
    up = torch.tensor([0.0, 0.0, 1.0], dtype=DT)
    assert float(collision.sphere_halfspace(up, r, normal=up, offset=0.2)) == pytest.approx(0.5)
    assert float(collision.sphere_capsule(up, r, start=-one, end=one, capsule_radius=r)) == pytest.approx(0.4)
    assert float(collision.sphere_capsule(up, r, start=one, end=one, capsule_radius=r)) == pytest.approx(2**0.5 - 0.6)  # zero length
    box = collision.sphere_box(2 * up, r, box_center=zero, box_rotation=torch.eye(3, dtype=DT), half_extents=torch.ones(3, dtype=DT))
    assert float(box) == pytest.approx(0.7)


def test_capsule_matches_brute_force():
    center, start, end = _randn(64, 3, seed=1), _randn(64, 3, seed=2), _randn(64, 3, seed=3)
    end[:8] = start[:8]  # degenerate capsules
    radius, capsule_radius = 0.5 * _randn(64, seed=4).abs(), 0.5 * _randn(64, seed=5).abs()
    t = torch.linspace(0, 1, 20001, dtype=DT)[:, None, None]
    brute = (center - (start + t * (end - start))).norm(dim=-1).amin(0) - radius - capsule_radius
    assert (brute < 0).any()
    assert torch.allclose(collision.sphere_capsule(center, radius, start=start, end=end, capsule_radius=capsule_radius), brute, atol=1e-6)


def test_box_matches_closest_point():
    rotation = torch.linalg.qr(_randn(32, 3, 3, seed=6))[0]
    box_center, half = _randn(32, 3, seed=7), 0.2 + _randn(32, 3, seed=8).abs()
    center = box_center + 1.5 * _randn(32, 3, seed=9)
    local = ((center - box_center)[:, None, :] @ rotation)[:, 0]
    outside = (local - torch.maximum(torch.minimum(local, half), -half)).norm(dim=-1)
    inside = -(half - local.abs()).amin(-1)
    expected = torch.where((local.abs() <= half).all(-1), inside, outside) - 0.1
    assert (expected < -0.1).any() and (expected > 0).any()
    radius = torch.full((32,), 0.1, dtype=DT)
    assert torch.allclose(collision.sphere_box(center, radius, box_center=box_center, box_rotation=rotation, half_extents=half), expected, atol=1e-10)


def test_gradients_finite_at_contact():
    r = torch.tensor(0.5, dtype=DT)
    center = torch.tensor([1.0, 0.0, 0.0], dtype=DT, requires_grad=True)
    origin = torch.zeros(3, dtype=DT)
    distances = [
        collision.sphere_sphere(center, r, center_b=origin, radius_b=r),
        collision.sphere_capsule(center, r, start=center.detach(), end=center.detach(), capsule_radius=r),  # centre on the segment
        collision.sphere_box(center, r, box_center=origin, box_rotation=torch.eye(3, dtype=DT), half_extents=torch.full((3,), 0.5, dtype=DT)),
    ]
    for distance in distances:
        (grad,) = torch.autograd.grad(distance, center)
        assert torch.isfinite(grad).all()


def test_sdf_lookup_matches_analytic_sphere():
    voxel, origin, ball = 0.02, torch.full((3,), -1.0, dtype=DT), torch.tensor([0.1, -0.2, 0.05], dtype=DT)
    axis = origin[0] + voxel * torch.arange(101, dtype=DT)
    points = torch.stack(torch.meshgrid(axis, axis, axis, indexing="ij"), dim=-1)
    grid = (points - ball).norm(dim=-1) - 0.3
    center, radius = 0.8 * torch.tanh(_randn(400, 3, seed=11)), torch.full((400,), 0.05, dtype=DT)
    center = center[(center - ball).norm(dim=-1) > 0.1]  # the field has a cone tip at the ball centre
    expected = collision.sphere_sphere(center, radius, center_b=ball, radius_b=torch.tensor(0.3, dtype=DT))
    assert torch.allclose(collision.sphere_sdf(center, radius, grid=grid, origin=origin, voxel_size=voxel), expected, atol=2e-3)
    far = torch.tensor([5.0, 0.0, 0.0], dtype=DT)  # clamped to the face x = 1
    clamped = collision.sphere_sdf(far, torch.tensor(0.0, dtype=DT), grid=grid, origin=origin, voxel_size=voxel)
    assert float(clamped) == pytest.approx(float((torch.tensor([1.0, 0, 0], dtype=DT) - ball).norm() - 0.3), abs=1e-3)


def _spheres(model):
    return collision.RobotSpheres.from_frames(
        model,
        {"body_panda_link3": [((0.0, 0.0, 0.0), 0.08)], "body_panda_link5": [((0.0, 0.05, -0.1), 0.07)],
         "body_panda_hand": [((0.0, 0.0, 0.05), 0.06)], "body_panda_link4": [((0.05, 0.0, 0.0), 0.07)]},
    )


def test_centers_match_manual_fk_and_pairs(panda):
    spheres = _spheres(panda)
    q = panda.lower_pos_limit + torch.rand(5, panda.nq, dtype=DT) * (panda.upper_pos_limit - panda.lower_pos_limit)
    state = forward_kinematics_state(panda, q)
    centers = spheres.centers(state)
    for s, frame in enumerate(spheres.frame_ids.tolist()):
        transform = state.frame_transform[:, frame]
        manual = transform[..., :3] @ spheres.offsets[s] + transform[..., 3]
        assert torch.allclose(centers[:, s], manual)
    pairs = collision.self_collision_pairs(panda, spheres)
    # spheres: 0 link3, 1 link5, 2 hand, 3 link4; link3-link4 and link4-link5 are parent-child
    assert pairs.tolist() == [[0, 1], [0, 2], [1, 2], [2, 3]]
    expected = collision.sphere_sphere(centers[:, 0], spheres.radii[0], center_b=centers[:, 2], radius_b=spheres.radii[2])
    assert torch.allclose(collision.self_distances(spheres, state, pairs=pairs)[:, 1], expected)


def test_pairs_ignore_fixed_joints(panda):
    # link7 -> link8 -> hand are joined by fixed joints (one rigid body); hand -> finger is one moving joint.
    spheres = collision.RobotSpheres.from_frames(
        panda,
        {"body_panda_link7": [((0.0, 0.0, 0.0), 0.05)], "body_panda_hand": [((0.0, 0.0, 0.05), 0.05)],
         "body_panda_leftfinger": [((0.0, 0.0, 0.02), 0.02)], "body_panda_link5": [((0.0, 0.0, 0.0), 0.05)]},
    )
    assert collision.self_collision_pairs(panda, spheres).tolist() == [[0, 3], [1, 3], [2, 3]]


def test_terms_pass_check_jacobian(panda):
    spheres = _spheres(panda)
    obstacle, size = torch.tensor([0.3, 0.0, 0.5], dtype=DT), torch.tensor(0.1, dtype=DT)

    def to_obstacle(centers, radii):
        return collision.sphere_sphere(centers, radii, center_b=obstacle, radius_b=size)

    q = 0.3 * _randn(1, 6, panda.nq, seed=10)
    problem = optim.Problem({"q": q}, manifolds={"q": terms.RobotManifold(panda)}, time=("q",))
    problem.cached(terms.kinematic_state(panda), name="state")
    world = terms.sphere_clearance(spheres, to_obstacle, margin=2.0)
    assert world.fn(forward_kinematics_state(panda, q)).shape == (1, 6, 4)
    problem.residual("world", world, window=1)
    pairs = collision.self_collision_pairs(panda, spheres)
    problem.residual("self", terms.self_clearance(spheres, pairs=pairs, margin=2.0), window=1)
    errors = optim.check_jacobian(problem)
    assert errors["world"] < 1e-9 and errors["self"] < 1e-9


def test_pairs_ignore_frame_pairs(panda):
    spheres = _spheres(panda)  # 0 link3, 1 link5, 2 hand, 3 link4
    ignore = [("body_panda_hand", "body_panda_link3"), (panda.frame_id("body_panda_link5"), "body_panda_hand")]
    assert collision.self_collision_pairs(panda, spheres, ignore=ignore).tolist() == [[0, 1], [2, 3]]
    with pytest.raises(br.exceptions.UnknownNameError):
        collision.self_collision_pairs(panda, spheres, ignore=[("no_such_frame", "body_panda_hand")])


def test_spheres_to_and_batched_offsets(panda):
    spheres = _spheres(panda)
    moved = spheres.to(dtype=torch.float32)
    assert moved.offsets.dtype == moved.radii.dtype == torch.float32 and moved.frame_ids.dtype == torch.long
    assert spheres.to(device="cpu", dtype=DT).offsets.dtype == DT
    # Per-sample offsets (B, S, 3) broadcast against a (B,) state: sample b uses its own offsets.
    q = 0.3 * _randn(2, panda.nq, seed=3)
    state = forward_kinematics_state(panda, q)
    offsets = torch.stack([spheres.offsets, spheres.offsets + 0.1])
    batched = collision.RobotSpheres(frame_ids=spheres.frame_ids, offsets=offsets, radii=spheres.radii)
    centers = batched.centers(state)
    assert centers.shape == (2, 4, 3)
    for b in range(2):
        single = collision.RobotSpheres(frame_ids=spheres.frame_ids, offsets=offsets[b], radii=spheres.radii)
        assert torch.allclose(centers[b], single.centers(state)[b])
    pairs = collision.self_collision_pairs(panda, spheres)
    assert collision.self_distances(batched, state, pairs=pairs).shape == (2, pairs.shape[0])
