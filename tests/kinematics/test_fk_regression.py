"""FK regression oracle.

Compares current forward-kinematics output against
``tests/kinematics/fk_reference.npz``. The file is committed and
re-generated only by an explicit run of ``_generate_fk_reference.py``
after a deliberate algorithmic change.

The arrays follow the model's one joint and frame order (parents first, the URDF's own order kept
wherever it already is), and the model values they come from are the URDF's at full precision; the
file's ``oracle_version`` names the generator that wrote it. The comparison sign-aligns quaternions
(``_sign_align_poses``) so a gauge change surfaces as the deliberate-regeneration decision it is
rather than a wall of sign flips. Pinocchio parity (``tests/parity/pinocchio``) remains the
fp64 rotation-matrix ground truth.

The oracle file is tracked; a missing ``fk_reference.npz`` fails the test.
"""

from __future__ import annotations

import functools
from pathlib import Path

import numpy as np
import pytest
import torch

import better_robot as br
from tests.kinematics import _generate_fk_reference as generator

REF_PATH = Path(__file__).parent / "fk_reference.npz"


def _sign_align_poses(actual: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Flip each pose quaternion to the sign of its dot with the reference.

    The matrix FK lane emits canonical-sign quaternions, so a pose may carry
    ``-q`` (the same rotation) relative to the stored oracle. Aligning by the
    dot sign keeps the regression gauge-robust at the original tolerance instead
    of loosening it. ``actual``/``reference`` are ``(..., 7)`` poses.
    """
    aligned = np.array(actual, copy=True)
    actual_q = aligned[..., 3:7]
    reference_q = reference[..., 3:7]
    dots = np.sum(actual_q * reference_q, axis=-1, keepdims=True)
    aligned[..., 3:7] = actual_q * np.where(dots < 0.0, -1.0, 1.0)
    return aligned


@pytest.fixture(scope="module")
def reference():
    return np.load(REF_PATH)


def test_oracle_provenance_matches_its_generator(reference) -> None:
    """The stored provenance keys are the generator's, and ``generated_with`` names a real public function."""
    meta = dict(entry.split("=", 1) for entry in reference["meta"])
    assert sorted(meta) == ["fk_dtype", "generated_at", "generated_with", "generation_seed", "oracle_version"]
    assert meta["oracle_version"] == generator.ORACLE_VERSION
    assert meta["generation_seed"] == str(generator.GENERATION_SEED)
    assert meta["generated_with"] == generator.GENERATED_WITH
    assert meta["fk_dtype"] == "float64"
    root, *path = meta["generated_with"].split(".")
    assert root == "better_robot"
    assert functools.reduce(getattr, path, br) is br.forward_kinematics


@pytest.mark.parametrize("name", generator.ROBOTS)
def test_fk_matches_oracle(name, reference) -> None:
    model = generator.ROBOTS[name]()
    qs = torch.from_numpy(reference[f"{name}_q"])
    expected_joints = reference[f"{name}_joint_pose_world"]
    expected_frames = reference[f"{name}_frame_pose_world"]
    for i in range(qs.shape[0]):
        state = br.forward_kinematics(model, qs[i])
        np.testing.assert_allclose(
            _sign_align_poses(state.joint_poses.detach().cpu().numpy(), expected_joints[i]),
            expected_joints[i],
            atol=1e-10,
            rtol=1e-10,
            err_msg=f"{name}: joint pose mismatch at sample {i}",
        )
        np.testing.assert_allclose(
            _sign_align_poses(state.frame_poses.detach().cpu().numpy(), expected_frames[i]),
            expected_frames[i],
            atol=1e-10,
            rtol=1e-10,
            err_msg=f"{name}: frame pose mismatch at sample {i}",
        )
