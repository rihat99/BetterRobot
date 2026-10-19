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


def _segment_distance_brute(start_a, end_a, start_b, end_b):
    # Sample segment a densely and project every sample onto segment b exactly.
    points = start_a + torch.linspace(0, 1, 20001, dtype=DT)[:, None, None] * (end_a - start_a)
    axis = end_b - start_b
    t = (((points - start_b) * axis).sum(-1) / axis.square().sum(-1).clamp_min(1e-300)).clamp(0, 1)
    return (points - start_b - t[..., None] * axis).norm(dim=-1).amin(0)


def _capsule_pairs():
    start_a, end_a, start_b, end_b = (_randn(64, 3, seed=seed) for seed in (20, 21, 22, 23))
    x, y = torch.tensor([1.0, 0, 0], dtype=DT), torch.tensor([0, 1.0, 0], dtype=DT)
    special = [  # parallel, antiparallel, collinear overlapping / apart, nearly parallel, crossing, points
        (0 * x, x, 0.3 * x + 0.5 * y, 1.8 * x + 0.5 * y),
        (0 * x, x, 1.8 * x + 0.5 * y, 0.3 * x + 0.5 * y),
        (0 * x, x, 0.5 * x, 2 * x),
        (0 * x, x, 3 * x, 2 * x),
        (0 * x, x, 0.2 * y - x, 0.2 * y + 2 * x + 1e-7 * y),
        (-x, x, -y, y),
        (0.2 * y, 0.2 * y, -x, x),
        (-x, x, 0.3 * y + 2 * x, 0.3 * y + 2 * x),
        (0.2 * y, 0.2 * y, x, x),
    ]
    for k, segments in enumerate(special):
        for tensor, value in zip((start_a, end_a, start_b, end_b), segments):
            tensor[k] = value
    return start_a, end_a, start_b, end_b


def test_capsule_capsule_matches_brute_force():
    start_a, end_a, start_b, end_b = _capsule_pairs()
    radius_a, radius_b = 0.3 * _randn(64, seed=24).abs(), 0.3 * _randn(64, seed=25).abs()
    expected = _segment_distance_brute(start_a, end_a, start_b, end_b) - radius_a - radius_b
    distance = collision.capsule_capsule(
        start_a, end_a, radius_a=radius_a, start_b=start_b, end_b=end_b, radius_b=radius_b
    )
    assert (expected < 0).any() and (expected > 0).any()
    assert torch.allclose(distance, expected, atol=1e-6)
    closed_form = torch.tensor([0.5, 0.5, 0, 1, 0.2, 0, 0.2, 1.09**0.5, 1.04**0.5], dtype=DT)
    assert torch.allclose(distance[:9] + radius_a[:9] + radius_b[:9], closed_form, atol=1e-6)


def test_capsule_capsule_gradients():
    start_a, end_a, start_b, end_b = _capsule_pairs()
    radius = torch.tensor(0.1, dtype=DT)

    def distance(*points):
        return collision.capsule_capsule(
            points[0], points[1], radius_a=radius, start_b=points[2], end_b=points[3], radius_b=radius
        )

    random = tuple(p[9:17].clone().requires_grad_() for p in (start_a, end_a, start_b, end_b))
    assert torch.autograd.gradcheck(distance, random)  # the closest points are unique on random segments
    jacobians = (
        torch.func.jacrev(distance, argnums=(0, 1, 2, 3))(*random),
        torch.func.jacfwd(distance, argnums=(0, 1, 2, 3))(*random),
    )
    assert all(torch.allclose(reverse, forward) for reverse, forward in zip(*jacobians))
    special = tuple(p.clone().requires_grad_() for p in (start_a, end_a, start_b, end_b))  # parallel, crossing, points
    assert all(torch.isfinite(grad).all() for grad in torch.autograd.grad(distance(*special).sum(), special))


G1_CAPSULES = {  # a subset of mjlab's G1 collision capsules, torso geoms moved into the URDF torso frame (-10 mm)
    "body_pelvis": [((0.0, 0.0, -0.08), (0.0, 0.0, -0.08), 0.07)],
    "body_left_hip_roll_link": [((0.02, 0.0, 0.0), (0.02, 0.0, -0.08), 0.06)],
    "body_right_hip_roll_link": [((0.02, 0.0, 0.0), (0.02, 0.0, -0.08), 0.06)],
    "body_left_hip_yaw_link": [((0.0, 0.0, -0.03), (-0.06, 0.0, -0.17), 0.055)],
    "body_torso_link": [((0.01, 0.0, 0.07), (0.01, 0.0, 0.19), 0.09), ((0.0, 0.0, 0.42), (0.0, 0.0, 0.42), 0.06)],
    "body_left_elbow_link": [((-0.01, 0.0, -0.01), (0.08, 0.0, -0.01), 0.035)],
    "body_left_wrist_pitch_link": [((-0.01, 0.0, 0.0), (0.06, 0.0, 0.0), 0.035)],
    "body_left_wrist_yaw_link": [((0.07, 0.0, 0.0), (0.15, -0.02, 0.0), 0.035)],
    "body_right_elbow_link": [((-0.01, 0.0, -0.01), (0.08, 0.0, -0.01), 0.035)],
    "body_right_wrist_pitch_link": [((-0.01, 0.0, 0.0), (0.06, 0.0, 0.0), 0.035)],
}


def test_g1_capsules_pose_pair_and_overlap_where_the_meshes_do():
    pytest.importorskip("robot_descriptions")
    from robot_descriptions import g1_description  # noqa: PLC0415

    g1 = br.load(g1_description.URDF_PATH, free_flyer=True, dtype=DT)
    capsules = collision.RobotCapsules.from_frames(g1, G1_CAPSULES)
    names = [g1.frame_names[frame].removeprefix("body_") for frame in capsules.frame_ids.tolist()]
    q = torch.stack([g1.random_configuration(torch.Generator().manual_seed(seed)) for seed in range(3)])
    state = forward_kinematics_state(g1, q)
    starts, ends = capsules.segments(state)
    for c, frame in enumerate(capsules.frame_ids.tolist()):
        transform = state.frame_transform[:, frame]
        assert torch.allclose(starts[:, c], transform[..., :3] @ capsules.starts[c] + transform[..., 3])
        assert torch.allclose(ends[:, c], transform[..., :3] @ capsules.ends[c] + transform[..., 3])
    pairs = collision.self_collision_pairs(g1, capsules)
    named = {(names[i], names[j]) for i, j in pairs.tolist()}
    # same body (the two torso capsules) and one-joint neighbours never pair
    assert ("torso_link", "torso_link") not in named and ("left_hip_roll_link", "left_hip_yaw_link") not in named
    assert ("left_elbow_link", "left_wrist_pitch_link") in named and ("pelvis", "left_hip_yaw_link") in named
    assert collision.self_capsule_distances(capsules, state, pairs=pairs).shape == (3, pairs.shape[0])
    neutral = collision.self_capsule_distances(capsules, forward_kinematics_state(g1, g1.q_neutral), pairs=pairs)
    overlapping = {(names[i], names[j]) for (i, j), d in zip(pairs.tolist(), neutral.tolist()) if d < 0}
    assert overlapping == {
        ("pelvis", "left_hip_roll_link"),
        ("pelvis", "right_hip_roll_link"),
        ("left_elbow_link", "left_wrist_pitch_link"),
        ("right_elbow_link", "right_wrist_pitch_link"),
    }


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_cuda_backward_is_bitwise_reproducible():
    """Posing many points per frame and RNEA's per-column gathers repeat an index: advanced indexing keeps their CUDA
    backward free of atomics (index_select's adds with them), so two backward passes agree to the bit."""
    pytest.importorskip("robot_descriptions")
    from robot_descriptions import g1_description  # noqa: PLC0415

    g1 = br.load(g1_description.URDF_PATH, free_flyer=True, dtype=DT, device="cuda")
    generator = torch.Generator(device="cuda").manual_seed(0)

    def randn(*shape):
        return torch.randn(*shape, dtype=DT, device="cuda", generator=generator)

    frame_ids = collision.RobotCapsules.from_frames(g1, G1_CAPSULES).frame_ids.repeat_interleave(20)  # 660 on 18
    radii = torch.zeros(len(frame_ids), dtype=DT, device="cuda")
    spheres = collision.RobotSpheres(frame_ids=frame_ids, offsets=0.1 * randn(len(frame_ids), 3), radii=radii)
    capsules = collision.RobotCapsules(
        frame_ids=frame_ids, starts=0.1 * randn(len(frame_ids), 3), ends=0.1 * randn(len(frame_ids), 3), radii=radii
    )
    q = g1.q_neutral.expand(64, -1).clone()
    q[:, 7:] += 0.3 * randn(64, g1.nq - 7)
    v, a = randn(64, g1.nv), randn(64, g1.nv)
    cases = {
        "centers": (lambda q: spheres.centers(forward_kinematics_state(g1, q)).square().sum(), (q,)),
        "segments": (lambda q: sum(s.square().sum() for s in capsules.segments(forward_kinematics_state(g1, q))), (q,)),
        "inverse_dynamics": (
            lambda q, v, a: br.inverse_dynamics(g1, forward_kinematics_state(g1, q), v=v, a=a).square().sum(),
            (q, v, a),
        ),
    }

    def gradients(fn, inputs):
        leaves = [x.clone().requires_grad_(True) for x in inputs]
        return torch.autograd.grad(fn(*leaves), leaves)

    for name, (fn, inputs) in cases.items():
        first, second = gradients(fn, inputs), gradients(fn, inputs)
        assert all(torch.equal(x, y) for x, y in zip(first, second)), name
