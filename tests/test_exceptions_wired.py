"""Verify the boundary error guards raise the right typed exceptions.

See ``docs/17_CONTRACTS.md §2``.
"""

from __future__ import annotations

import math

import pytest
import torch

from better_robot.exceptions import (
    DeviceMismatchError,
    ModelInconsistencyError,
    QuaternionNormError,
    ShapeError,
)
from better_robot.io.build_model import build_model
from better_robot.io.parsers.programmatic import ModelBuilder
from better_robot.kinematics.forward import forward_kinematics, forward_kinematics_raw


# ────────────────────────── fixtures ──────────────────────────

@pytest.fixture(scope="module")
def arm_model():
    b = ModelBuilder("arm")
    b.add_body("base", mass=0.5)
    b.add_body("link1", mass=1.0)
    b.add_joint("j1", kind="revolute", parent="base", child="link1",
                origin=torch.tensor([0., 0., 0.1, 0., 0., 0., 1.]),
                axis=torch.tensor([0., 0., 1.]),
                lower=-math.pi, upper=math.pi)
    return build_model(b.finalize())


@pytest.fixture(scope="module")
def free_flyer_model():
    b = ModelBuilder("floating")
    b.add_body("base", mass=1.0)
    b.add_body("link1", mass=0.5)
    b.add_joint("floating", kind="free_flyer", parent="world", child="base",
                origin=torch.tensor([0., 0., 0., 0., 0., 0., 1.]))
    b.add_joint("j1", kind="revolute", parent="base", child="link1",
                origin=torch.tensor([0., 0., 0.1, 0., 0., 0., 1.]),
                axis=torch.tensor([0., 0., 1.]),
                lower=-math.pi, upper=math.pi)
    return build_model(b.finalize())


# ────────────────────── ShapeError ──────────────────────

def test_shape_error_on_wrong_nq(arm_model):
    bad_q = torch.zeros(arm_model.nq + 3)
    with pytest.raises(ShapeError, match="trailing size"):
        forward_kinematics_raw(arm_model, bad_q)


def test_shape_error_on_wrong_nq_batched(arm_model):
    bad_q = torch.zeros(4, arm_model.nq + 1)
    with pytest.raises(ShapeError):
        forward_kinematics(arm_model, bad_q)


# ────────────────────── DeviceMismatchError ──────────────────────

def test_device_mismatch_error_is_subclass_of_valueerror():
    """DeviceMismatchError inherits ValueError so legacy except blocks catch it."""
    assert issubclass(DeviceMismatchError, ValueError)


def test_device_mismatch_on_cpu_cuda():
    """If CUDA is unavailable we skip — the guard is exercised regardless via docs contract tests."""
    if not torch.cuda.is_available():
        pytest.skip("no CUDA available")
    b = ModelBuilder("arm")
    b.add_body("base", mass=0.5)
    b.add_body("link1", mass=1.0)
    b.add_joint("j1", kind="revolute", parent="base", child="link1",
                origin=torch.tensor([0., 0., 0.1, 0., 0., 0., 1.]),
                axis=torch.tensor([0., 0., 1.]),
                lower=-math.pi, upper=math.pi)
    model = build_model(b.finalize())  # CPU
    q = torch.zeros(model.nq, device="cuda")
    with pytest.raises(DeviceMismatchError, match="device"):
        forward_kinematics_raw(model, q)


# ────────────────────── QuaternionNormError ──────────────────────

def test_quaternion_norm_error_on_floating_base(free_flyer_model):
    q = free_flyer_model.q_neutral.clone()
    # Deliberately zero out the quaternion (norm 0) — well outside the band.
    q[3:7] = 0.0
    with pytest.raises(QuaternionNormError, match="quaternion norm"):
        forward_kinematics_raw(free_flyer_model, q)


def test_quaternion_norm_accepts_small_drift(free_flyer_model):
    """A quaternion inside [0.9, 1.1] is silently accepted."""
    q = free_flyer_model.q_neutral.clone()
    q[3:7] = q[3:7] * 1.02  # ~2% over unit norm — fine
    # No exception:
    forward_kinematics_raw(free_flyer_model, q)


def test_fixed_base_bypasses_quaternion_check(arm_model):
    """The quaternion guard only fires for free-flyer roots."""
    q = arm_model.q_neutral.clone()
    # Arbitrary q value — no quaternion here, so no error should fire.
    forward_kinematics_raw(arm_model, q)


# ────────────────────── ModelInconsistencyError ──────────────────────

def test_model_inconsistency_on_bad_parents():
    """Directly call the invariant checker with a malformed parents tuple."""
    from better_robot.io.build_model import _check_topology_invariants

    with pytest.raises(ModelInconsistencyError, match="parents\\[0\\]"):
        _check_topology_invariants(
            parents=(0, 0, 1),   # parents[0] should be -1
            nqs=(0, 0, 1),
            nvs=(0, 0, 1),
            idx_qs=(0, 0, 0),
            idx_vs=(0, 0, 0),
            nq_total=1,
            nv_total=1,
        )


def test_model_inconsistency_on_out_of_order_parents():
    from better_robot.io.build_model import _check_topology_invariants

    with pytest.raises(ModelInconsistencyError, match="topological sort"):
        _check_topology_invariants(
            parents=(-1, 2, 0),   # parents[1]=2 > 1, violates topo sort
            nqs=(0, 1, 1),
            nvs=(0, 1, 1),
            idx_qs=(0, 0, 1),
            idx_vs=(0, 0, 1),
            nq_total=2,
            nv_total=2,
        )


def test_model_inconsistency_on_q_slice_gap():
    from better_robot.io.build_model import _check_topology_invariants

    with pytest.raises(ModelInconsistencyError, match="q-slicing"):
        _check_topology_invariants(
            parents=(-1, 0, 1),
            nqs=(0, 1, 1),
            nvs=(0, 1, 1),
            idx_qs=(0, 0, 2),   # gap! should be (0, 0, 1)
            idx_vs=(0, 0, 1),
            nq_total=2,
            nv_total=2,
        )


def test_model_inconsistency_on_nq_mismatch():
    from better_robot.io.build_model import _check_topology_invariants

    with pytest.raises(ModelInconsistencyError, match="nq"):
        _check_topology_invariants(
            parents=(-1, 0),
            nqs=(0, 1),
            nvs=(0, 1),
            idx_qs=(0, 0),
            idx_vs=(0, 0),
            nq_total=5,   # wrong total
            nv_total=1,
        )
