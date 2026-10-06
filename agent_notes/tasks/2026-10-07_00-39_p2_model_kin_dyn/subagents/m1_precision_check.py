"""M1: how exact is a float64 model loaded from URDF? Run before and after the parser change.

Prints, for Panda and G1 (fixed base): largest deviation of placement quaternions from unit norm,
largest frame translation / rotation difference against pinocchio over 16 random configurations.
Usage: m1_precision_check.py out.json
"""

from __future__ import annotations

import dataclasses
import json
import sys

import numpy as np
import pinocchio as pin
import torch
from robot_descriptions import g1_description, panda_description

import better_robot as br
from better_robot.io import build_model, parse_urdf
from better_robot.lie import so3


def parity(path: str) -> dict[str, float]:
    """Full-coordinate model (mimic stripped, as in tests/test_pinocchio/conftest.py) against pinocchio."""
    ir = parse_urdf(path)
    ir.joints = [dataclasses.replace(j, mimic_source=None, mimic_multiplier=1.0, mimic_offset=0.0) for j in ir.joints]
    model = build_model(ir, dtype=torch.float64)
    pin_model = pin.buildModelFromUrdf(path)
    pin_data = pin_model.createData()
    assert pin_model.nq == model.nq
    names = {n: pin_model.getFrameId(n[5:]) for n in model.frame_names if n.startswith("body_") and pin_model.existFrame(n[5:])}
    generator = torch.Generator().manual_seed(0)
    worst_t = worst_r = 0.0
    for _ in range(16):
        q = torch.rand(model.nq, generator=generator, dtype=torch.float64) - 0.5
        data = br.forward_kinematics(model, q, compute_frames=True)
        pin.forwardKinematics(pin_model, pin_data, q.numpy())
        pin.updateFramePlacements(pin_model, pin_data)
        for name, frame_id in names.items():
            pose = data.frame_pose_world[model.frame_id(name)]
            worst_t = max(worst_t, float(np.abs(pose[:3].numpy() - pin_data.oMf[frame_id].translation).max()))
            worst_r = max(worst_r, float(np.abs(so3.to_matrix(pose[3:]).numpy() - pin_data.oMf[frame_id].rotation).max()))
    placement = model.joint_placements[..., 3:]
    return {"frames_compared": len(names), "quaternion_norm_error": float((placement.norm(dim=-1) - 1).abs().max()),
            "translation_vs_pinocchio": worst_t, "rotation_vs_pinocchio": worst_r}


def main() -> None:
    out = {"panda": parity(panda_description.URDF_PATH), "g1_fixed_base": parity(g1_description.URDF_PATH)}
    print(json.dumps(out, indent=1))
    json.dump(out, open(sys.argv[1], "w"), indent=1)


if __name__ == "__main__":
    main()
