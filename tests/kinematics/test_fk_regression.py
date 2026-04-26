"""FK regression oracle.

Compares current forward-kinematics output against
``tests/kinematics/fk_reference.npz`` at ``atol=1e-10`` in fp64. The
file is committed and re-generated only by an explicit run of
``_generate_fk_reference.py`` after a deliberate algorithmic change.

Skips automatically if ``fk_reference.npz`` is missing — the oracle file
ships only after the first commit of the bench infrastructure.

See ``docs/conventions/testing.md §4.5``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

import better_robot as br

REF_PATH = Path(__file__).parent / "fk_reference.npz"


def _load_panda():
    pytest.importorskip("robot_descriptions")
    from robot_descriptions import panda_description
    return br.load(panda_description.URDF_PATH, dtype=torch.float64)


def _load_g1():
    pytest.importorskip("robot_descriptions")
    from robot_descriptions import g1_description
    return br.load(g1_description.URDF_PATH, free_flyer=True, dtype=torch.float64)


@pytest.fixture(scope="module")
def reference():
    if not REF_PATH.exists():
        pytest.skip(f"oracle file {REF_PATH.name} not present; run _generate_fk_reference.py")
    return np.load(REF_PATH)


@pytest.mark.parametrize("name,loader", [("panda", _load_panda), ("g1", _load_g1)])
def test_fk_matches_oracle(name, loader, reference) -> None:
    model = loader()
    qs = torch.from_numpy(reference[f"{name}_q"])
    expected_joints = reference[f"{name}_joint_pose_world"]
    expected_frames = reference[f"{name}_frame_pose_world"]
    for i in range(qs.shape[0]):
        data = br.forward_kinematics(model, qs[i], compute_frames=True)
        np.testing.assert_allclose(
            data.joint_pose_world.detach().cpu().numpy(),
            expected_joints[i],
            atol=1e-10,
            rtol=1e-10,
            err_msg=f"{name}: joint pose mismatch at sample {i}",
        )
        np.testing.assert_allclose(
            data.frame_pose_world.detach().cpu().numpy(),
            expected_frames[i],
            atol=1e-10,
            rtol=1e-10,
            err_msg=f"{name}: frame pose mismatch at sample {i}",
        )
