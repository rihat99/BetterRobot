"""Tests for collision geometry primitives."""
import torch
import pytest
from better_robot.algorithms.geometry.primitives import (
    Sphere, Capsule, Box, HalfSpace, CollGeom
)


def test_sphere_transform():
    s = Sphere(center=torch.tensor([1.0, 0.0, 0.0]), radius=0.1)
    # Identity transform: [0,0,0, 0,0,0,1]
    identity = torch.tensor([0., 0., 0., 0., 0., 0., 1.])
    s2 = s.transform(identity)
    assert torch.allclose(s2.center, s.center, atol=1e-5)
    assert s2.radius == s.radius


def test_sphere_position_alias():
    s = Sphere(center=torch.tensor([1.0, 2.0, 3.0]), radius=0.5)
    assert torch.allclose(s.position, s.center)


def test_capsule_from_endpoints():
    p0 = torch.tensor([0., 0., 0.])
    p1 = torch.tensor([0., 0., 1.])
    c = Capsule.from_endpoints(p0, p1, radius=0.05)
    assert torch.allclose(c.point_a, p0)
    assert torch.allclose(c.point_b, p1)
    assert c.radius == 0.05


def test_capsule_center():
    p0 = torch.tensor([0., 0., 0.])
    p1 = torch.tensor([0., 0., 2.])
    c = Capsule.from_endpoints(p0, p1, radius=0.05)
    assert torch.allclose(c.center, torch.tensor([0., 0., 1.]))


def test_capsule_decompose_to_spheres():
    p0 = torch.tensor([0., 0., 0.])
    p1 = torch.tensor([0., 0., 4.])
    c = Capsule.from_endpoints(p0, p1, radius=0.1)
    spheres = c.decompose_to_spheres(n_segments=5)
    assert len(spheres) == 5
    assert all(isinstance(s, Sphere) for s in spheres)
    assert all(s.radius == 0.1 for s in spheres)
    # First sphere at p0, last at p1
    assert torch.allclose(spheres[0].center, p0, atol=1e-5)
    assert torch.allclose(spheres[-1].center, p1, atol=1e-5)


def test_box_center_alias():
    b = Box(position=torch.tensor([1., 2., 3.]), extent=torch.tensor([0.1, 0.2, 0.3]))
    assert torch.allclose(b.center, b.position)


def test_halfspace_transform():
    hs = HalfSpace(
        point=torch.tensor([0., 0., 0.]),
        normal=torch.tensor([0., 0., 1.]),
    )
    identity = torch.tensor([0., 0., 1., 0., 0., 0., 1.])  # translate z by 1
    hs2 = hs.transform(identity)
    assert torch.allclose(hs2.point, torch.tensor([0., 0., 1.]), atol=1e-5)
