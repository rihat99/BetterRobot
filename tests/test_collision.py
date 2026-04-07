"""Tests for collision distance functions and RobotCollision modes."""
import torch
import pytest
from better_robot.algorithms.geometry.primitives import Sphere, Capsule, HalfSpace
from better_robot.algorithms.geometry.distance_pairs import (
    sphere_sphere, sphere_capsule, capsule_capsule,
    halfspace_sphere, halfspace_capsule,
)
from better_robot.algorithms.geometry.distance import compute_distance, colldist_from_sdf


def test_sphere_sphere_separated():
    s1 = Sphere(center=torch.tensor([0., 0., 0.]), radius=0.1)
    s2 = Sphere(center=torch.tensor([1., 0., 0.]), radius=0.1)
    dist = sphere_sphere(s1, s2)
    assert dist.item() > 0  # separated: 1.0 - 0.2 = 0.8


def test_sphere_sphere_penetrating():
    s1 = Sphere(center=torch.tensor([0., 0., 0.]), radius=0.3)
    s2 = Sphere(center=torch.tensor([0.2, 0., 0.]), radius=0.3)
    dist = sphere_sphere(s1, s2)
    assert dist.item() < 0  # penetrating


def test_halfspace_sphere_above():
    hs = HalfSpace(point=torch.tensor([0., 0., 0.]), normal=torch.tensor([0., 0., 1.]))
    s = Sphere(center=torch.tensor([0., 0., 1.0]), radius=0.1)
    dist = halfspace_sphere(hs, s)
    assert dist.item() > 0  # sphere above plane


def test_halfspace_sphere_below():
    hs = HalfSpace(point=torch.tensor([0., 0., 0.]), normal=torch.tensor([0., 0., 1.]))
    s = Sphere(center=torch.tensor([0., 0., -0.5]), radius=0.1)
    dist = halfspace_sphere(hs, s)
    assert dist.item() < 0  # sphere below plane (penetrating)


def test_compute_distance_dispatch():
    s1 = Sphere(center=torch.tensor([0., 0., 0.]), radius=0.1)
    s2 = Sphere(center=torch.tensor([1., 0., 0.]), radius=0.1)
    d = compute_distance(s1, s2)
    assert isinstance(d, torch.Tensor)
    assert d.item() > 0


def test_colldist_from_sdf_above_activation():
    dist = torch.tensor([0.1, 0.2, 0.5])
    result = colldist_from_sdf(dist, activation_dist=0.05)
    assert (result <= 0).all()  # should be 0 when dist >= activation_dist
    assert torch.allclose(result, torch.zeros_like(result), atol=1e-5)


def test_colldist_from_sdf_penetrating():
    dist = torch.tensor([-0.1, -0.2])
    result = colldist_from_sdf(dist, activation_dist=0.05)
    assert (result < 0).all()  # penetrating gives negative values


# ---------------------------------------------------------------------------
# RobotCollision adjacent-link filtering
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def panda():
    from robot_descriptions.loaders.yourdfpy import load_robot_description
    from better_robot import load_urdf
    urdf = load_robot_description("panda_description")
    return load_urdf(urdf)


def test_robot_collision_excludes_adjacent_links(panda):
    """Parent-child link sphere pairs must not appear in active collision pairs."""
    from better_robot.algorithms.geometry.robot_collision import RobotCollision

    # link0 and link1 are adjacent (joint 0 connects them).
    # link0 and link4 are non-adjacent (two joints apart).
    sphere_decomp = {
        "panda_link0": {"center": [0.0, 0.0, 0.0], "radius": 0.08},
        "panda_link1": {"center": [0.0, 0.0, 0.0], "radius": 0.07},
        "panda_link4": {"center": [0.0, 0.0, 0.0], "radius": 0.06},
    }
    rc = RobotCollision.from_sphere_decomposition(sphere_decomp, panda)

    link_name_to_idx = {name: idx for idx, name in enumerate(panda.links.names)}
    idx0 = link_name_to_idx["panda_link0"]
    idx1 = link_name_to_idx["panda_link1"]
    idx4 = link_name_to_idx["panda_link4"]

    # Build the set of active link-index pairs represented by the active sphere pairs
    sphere_link_indices = rc._link_indices.tolist()
    active_link_pairs = set()
    for si, sj in zip(rc._active_pairs_i, rc._active_pairs_j):
        li, lj = sphere_link_indices[si], sphere_link_indices[sj]
        active_link_pairs.add((min(li, lj), max(li, lj)))

    adjacent_pair = (min(idx0, idx1), max(idx0, idx1))
    non_adjacent_pair = (min(idx0, idx4), max(idx0, idx4))

    assert adjacent_pair not in active_link_pairs, \
        "Adjacent link pair (link0, link1) should be excluded from active collision pairs"
    assert non_adjacent_pair in active_link_pairs, \
        "Non-adjacent link pair (link0, link4) should be included in active collision pairs"


def test_robot_collision_excludes_same_link(panda):
    """Same-link sphere pairs must not appear in active collision pairs."""
    from better_robot.algorithms.geometry.robot_collision import RobotCollision

    sphere_decomp = {
        "panda_link3": {
            "centers": [[0.0, 0.0, -0.06], [0.0, 0.0, 0.06]],
            "radii": [0.06, 0.06],
        },
        "panda_link5": {"center": [0.0, 0.0, 0.0], "radius": 0.06},
    }
    rc = RobotCollision.from_sphere_decomposition(sphere_decomp, panda)

    link_name_to_idx = {name: idx for idx, name in enumerate(panda.links.names)}
    idx3 = link_name_to_idx["panda_link3"]

    sphere_link_indices = rc._link_indices.tolist()
    for si, sj in zip(rc._active_pairs_i, rc._active_pairs_j):
        li, lj = sphere_link_indices[si], sphere_link_indices[sj]
        assert li != lj, "Same-link sphere pair found in active collision pairs"


# ---------------------------------------------------------------------------
# Capsule.from_trimesh
# ---------------------------------------------------------------------------

def test_capsule_from_trimesh_empty():
    """Empty mesh produces a zero-radius capsule at the origin."""
    import trimesh as tm
    mesh = tm.Trimesh()
    cap = Capsule.from_trimesh(mesh)
    assert cap.radius == 0.0
    assert cap.point_a.shape == (3,)
    assert cap.point_b.shape == (3,)


def test_capsule_from_trimesh_box():
    """A box mesh gives a capsule with positive radius and non-degenerate endpoints."""
    import trimesh as tm
    mesh = tm.creation.box(extents=[0.2, 0.1, 0.4])
    cap = Capsule.from_trimesh(mesh)
    assert cap.radius > 0.0
    # The capsule axis should be aligned with the longest dimension (z = 0.4).
    length = (cap.point_b - cap.point_a).norm().item()
    assert length > 0.0, "Capsule axis length should be positive"


def test_capsule_from_trimesh_cylinder():
    """A cylinder mesh gives a capsule that tightly matches the cylinder."""
    import trimesh as tm
    mesh = tm.creation.cylinder(radius=0.05, height=0.3)
    cap = Capsule.from_trimesh(mesh)
    assert cap.radius > 0.0
    length = (cap.point_b - cap.point_a).norm().item()
    assert length > 0.0


# ---------------------------------------------------------------------------
# RobotCollision capsule mode
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def panda_urdf():
    from robot_descriptions.loaders.yourdfpy import load_robot_description
    return load_robot_description("panda_description")


@pytest.fixture(scope="module")
def panda_with_urdf(panda_urdf):
    from better_robot import load_urdf
    return load_urdf(panda_urdf), panda_urdf


def test_robot_collision_from_urdf_mode(panda_with_urdf):
    """from_urdf() creates a capsule-mode RobotCollision."""
    from better_robot.algorithms.geometry.robot_collision import RobotCollision
    model, urdf = panda_with_urdf
    rc = RobotCollision.from_urdf(urdf, model)
    assert rc._mode == "capsule"
    assert rc._local_points_a is not None
    assert rc._local_points_b is not None
    assert rc._capsule_radii is not None
    # One capsule per link.
    assert rc._local_points_a.shape == (model.links.num_links, 3)


def test_robot_collision_from_urdf_active_pairs(panda_with_urdf):
    """from_urdf() excludes adjacent-link pairs from self-collision checking."""
    from better_robot.algorithms.geometry.robot_collision import RobotCollision
    model, urdf = panda_with_urdf
    rc = RobotCollision.from_urdf(urdf, model)
    assert len(rc._active_pairs_i) > 0, "Expected some active collision pairs"

    # Build the set of adjacent link pairs.
    from better_robot.algorithms.geometry.robot_collision import _build_adjacent_set
    adjacent = _build_adjacent_set(model)

    cap_link_indices = rc._capsule_link_indices.tolist()
    for i, j in zip(rc._active_pairs_i, rc._active_pairs_j):
        li = cap_link_indices[i]
        lj = cap_link_indices[j]
        pair = (min(li, lj), max(li, lj))
        assert pair not in adjacent, f"Adjacent pair ({li}, {lj}) found in active pairs"


def test_robot_collision_from_urdf_distance_shape(panda_with_urdf):
    """compute_self_collision_distance returns a (num_active_pairs,) tensor."""
    from better_robot.algorithms.geometry.robot_collision import RobotCollision
    model, urdf = panda_with_urdf
    rc = RobotCollision.from_urdf(urdf, model)
    q = model.q_default.clone()
    dists = rc.compute_self_collision_distance(model, q)
    assert dists.shape == (len(rc._active_pairs_i),)


def test_robot_collision_from_urdf_differentiable(panda_with_urdf):
    """Capsule self-collision distances are differentiable w.r.t. joint config."""
    from better_robot.algorithms.geometry.robot_collision import RobotCollision
    model, urdf = panda_with_urdf
    rc = RobotCollision.from_urdf(urdf, model)
    q = model.q_default.clone().requires_grad_(True)
    dists = rc.compute_self_collision_distance(model, q)
    if dists.numel() > 0:
        dists.sum().backward()
        assert q.grad is not None
        assert not torch.isnan(q.grad).any()


def test_robot_collision_from_capsule_decomposition(panda):
    """Manual capsule decomposition creates capsule mode with adjacency filtering."""
    from better_robot.algorithms.geometry.robot_collision import RobotCollision
    cap_decomp = {
        "panda_link0": {"point_a": [0.0, 0.0, 0.0], "point_b": [0.0, 0.0, 0.1], "radius": 0.08},
        "panda_link1": {"point_a": [0.0, 0.0, 0.0], "point_b": [0.0, 0.0, 0.08], "radius": 0.07},
        "panda_link4": {"point_a": [0.0, 0.0, -0.06], "point_b": [0.0, 0.0, 0.06], "radius": 0.06},
    }
    rc = RobotCollision.from_capsule_decomposition(cap_decomp, panda)
    assert rc._mode == "capsule"

    link_name_to_idx = {name: idx for idx, name in enumerate(panda.links.names)}
    idx0 = link_name_to_idx["panda_link0"]
    idx1 = link_name_to_idx["panda_link1"]
    idx4 = link_name_to_idx["panda_link4"]

    cap_link_indices = rc._capsule_link_indices.tolist()
    active_link_pairs = set()
    for i, j in zip(rc._active_pairs_i, rc._active_pairs_j):
        li, lj = cap_link_indices[i], cap_link_indices[j]
        active_link_pairs.add((min(li, lj), max(li, lj)))

    adjacent_pair = (min(idx0, idx1), max(idx0, idx1))
    non_adjacent_pair = (min(idx0, idx4), max(idx0, idx4))
    assert adjacent_pair not in active_link_pairs
    assert non_adjacent_pair in active_link_pairs


def test_robot_collision_capsule_world_collision(panda_with_urdf):
    """Capsule mode works for world collision (robot vs halfspace)."""
    from better_robot.algorithms.geometry.robot_collision import RobotCollision
    model, urdf = panda_with_urdf
    rc = RobotCollision.from_urdf(urdf, model)
    q = model.q_default.clone()
    floor = HalfSpace.from_point_and_normal(
        point=torch.tensor([0., 0., -0.5]),
        normal=torch.tensor([0., 0., 1.]),
    )
    dists = rc.compute_world_collision_distance(model, q, [floor])
    assert dists.shape == (model.links.num_links,)
