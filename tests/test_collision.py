"""Tests for collision distance functions."""
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
