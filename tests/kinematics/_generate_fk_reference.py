"""Generate ``fk_reference.npz`` — frozen FK regression oracle.

Captures forward-kinematics output of the Panda + G1 (free-flyer)
robots at fixed seeds in fp64. Run this once when the oracle needs to
be bumped (after a deliberate algorithmic change) — the result is
committed to git as :file:`fk_reference.npz` and read by
:mod:`tests.kinematics.test_fk_regression`.

Re-running this script with the same seed and pinned versions must
reproduce the same file bit-for-bit (modulo NumPy float printing).

Run:

    uv run python tests/kinematics/_generate_fk_reference.py

See ``docs/conventions/16_TESTING.md §4.5`` and
``docs/claude_plan/accepted/12_regression_and_benchmarks.md``.
"""

from __future__ import annotations

import datetime as _dt
from pathlib import Path

import numpy as np
import torch

import better_robot as br

ORACLE_VERSION = "2"
GENERATION_SEED = 0
N_CONFIGS = 32  # keep small — < 200 KB total


def _panda():
    from robot_descriptions import panda_description

    return br.load(panda_description.URDF_PATH, dtype=torch.float64)


def _g1():
    from robot_descriptions import g1_description

    return br.load(g1_description.URDF_PATH, free_flyer=True, dtype=torch.float64)


def _sample_qs(model, n: int, *, seed: int) -> torch.Tensor:
    rng = torch.Generator().manual_seed(seed)
    nq = model.nq
    out = torch.zeros(n, nq, dtype=torch.float64)
    free_flyer = model.joint_models[1].kind == "free_flyer" if model.njoints > 1 else False
    if free_flyer:
        out[:, 0:3] = (torch.rand(n, 3, generator=rng, dtype=torch.float64) - 0.5) * 0.4
        quats = torch.randn(n, 4, generator=rng, dtype=torch.float64)
        out[:, 3:7] = quats / quats.norm(dim=-1, keepdim=True)
        joint_offset = 7
    else:
        joint_offset = 0
    if nq > joint_offset:
        out[:, joint_offset:] = (
            torch.rand(n, nq - joint_offset, generator=rng, dtype=torch.float64) - 0.5
        ) * 0.6
    return out


def main() -> None:
    out_path = Path(__file__).parent / "fk_reference.npz"
    payload: dict[str, np.ndarray] = {}

    for name, build_fn in (("panda", _panda), ("g1", _g1)):
        model = build_fn()
        qs = _sample_qs(model, N_CONFIGS, seed=GENERATION_SEED)
        joint_poses = []
        frame_poses = []
        for i in range(N_CONFIGS):
            data = br.forward_kinematics(model, qs[i], compute_frames=True)
            joint_poses.append(data.joint_pose_world.detach().cpu().numpy())
            frame_poses.append(data.frame_pose_world.detach().cpu().numpy())
        payload[f"{name}_q"] = qs.numpy()
        payload[f"{name}_joint_pose_world"] = np.stack(joint_poses)
        payload[f"{name}_frame_pose_world"] = np.stack(frame_poses)

    payload["meta"] = np.array(
        [
            f"oracle_version={ORACLE_VERSION}",
            f"generation_seed={GENERATION_SEED}",
            f"generated_with=better_robot.forward_kinematics(compute_frames=True)",
            "fk_dtype=float64",
            f"generated_at={_dt.datetime.utcnow().strftime('%Y-%m-%d')}",
        ]
    )

    np.savez_compressed(out_path, **payload)
    size_kb = out_path.stat().st_size / 1024
    print(f"wrote {out_path.name} ({size_kb:.1f} KB)")


if __name__ == "__main__":
    main()
