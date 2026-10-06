"""Does the exact SE3 Jacobian change batched IK convergence? Panda, B=1024, default LM, CPU float32."""

from __future__ import annotations

import json
import math
import sys

import torch
from robot_descriptions import panda_description

import better_robot as br
from better_robot.kinematics.forward import forward_kinematics
from better_robot.lie import tangents
from better_robot.residuals import pose as pose_module
from better_robot.tasks import solve_ik


def old_right_jacobian_inv_se3(xi: torch.Tensor) -> torch.Tensor:
    """The 9-term series used before commit 0953f28."""
    ad = tangents._ad_matrix(xi)
    eye = torch.eye(6, dtype=xi.dtype, device=xi.device).expand_as(ad)
    result, power = eye.clone(), eye.clone()
    for k in range(1, 10):
        power = power @ ad
        result = result + ((-1.0) ** k) / math.factorial(k + 1) * power
    return torch.linalg.inv(result)


def run(batch: int, spread: float) -> dict:
    model = br.load(panda_description.URDF_PATH)
    frame = "body_panda_hand" if "body_panda_hand" in model.frame_name_to_id else model.frame_names[-1]
    torch.manual_seed(1)
    q_goal = model.q_neutral + spread * torch.randn(batch, model.nq)
    q_goal = torch.minimum(torch.maximum(q_goal, model.lower_pos_limit), model.upper_pos_limit)
    target = forward_kinematics(model, q_goal, compute_frames=True).frame_pose_world[..., model.frame_id(frame), :]
    result = solve_ik(model, {frame: target.detach()})
    reached = result.frame_pose(frame)
    position_error = (reached[..., :3] - target[..., :3]).norm(dim=-1)
    quat_dot = (reached[..., 3:] * target[..., 3:]).sum(-1).abs().clamp(max=1.0)
    rotation_error = 2.0 * torch.acos(quat_dot)
    quantiles = torch.tensor([0.5, 0.9, 0.99, 1.0])
    return {
        "converged": float(result.converged.float().mean()),
        "iters_median": float(result.iters.float().median()),
        "position_error_m_q50_q90_q99_max": position_error.quantile(quantiles).tolist(),
        "rotation_error_rad_q50_q90_q99_max": rotation_error.quantile(quantiles).tolist(),
        "frac_pos_below_1mm": float((position_error < 1e-3).float().mean()),
    }


if __name__ == "__main__":
    torch.set_num_threads(8)
    out = {}
    for spread in (0.3, 1.0):
        out[f"exact_spread{spread}"] = run(1024, spread)
        pose_module.right_jacobian_inv_se3 = old_right_jacobian_inv_se3
        out[f"old_series_spread{spread}"] = run(1024, spread)
        pose_module.right_jacobian_inv_se3 = tangents.right_jacobian_inv_se3
    json.dump(out, open(sys.argv[1], "w"), indent=1)
    print(json.dumps(out, indent=1))
