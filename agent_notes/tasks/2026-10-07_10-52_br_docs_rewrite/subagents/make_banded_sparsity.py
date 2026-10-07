"""Spy plot of the Gauss-Newton normal matrix of a small Panda trajectory problem (from examples/02_trajectory.py)."""
import math, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from robot_descriptions import panda_description
import better_robot as br
from better_robot import optim, terms

READY = [0.0, -math.pi / 4, 0.0, -3 * math.pi / 4, 0.0, math.pi / 2, math.pi / 4, 0.04]
model = br.load(panda_description.URDF_PATH, dtype=torch.float64)
q0 = torch.tensor(READY, dtype=torch.float64).clamp(model.lower_pos_limit, model.upper_pos_limit)

def normal_matrix(knots):
    target = br.kinematics.forward_kinematics_state(model, q0[None]).frame_pose[0, model.frame_id("body_panda_hand")].repeat(1, knots, 1)
    q = q0.repeat(1, knots, 1) + 0.05 * torch.randn(1, knots, q0.shape[0], dtype=torch.float64)
    problem = optim.Problem({"q": q}, manifolds={"q": terms.RobotManifold(model)}, data={"target": target}, time=("q",))
    problem.cached(terms.kinematic_state(model), name="state")
    problem.residual("pose", terms.frame_pose(model, "body_panda_hand", target="target"), window=1)
    problem.residual("velocity", terms.finite_difference(model, order=1, dt=0.05), weight=1e-3)
    problem.residual("acceleration", terms.finite_difference(model, order=2, dt=0.05), weight=1e-4)
    normal, _ = problem.linearize(problem.variables)
    print(knots, type(normal).__name__)
    return normal.dense()[0]

fig, axes = plt.subplots(1, 2, figsize=(12, 6), dpi=100, facecolor="white")
for ax, knots in zip(axes, (6, 12)):
    H = normal_matrix(knots)
    ax.spy(H.abs() > 1e-12, markersize=2.5 if knots == 6 else 1.4, color="#1f4e79")
    n = H.shape[0]
    ax.set_title(f"T = {knots} knots, {n}×{n} normal matrix", fontsize=13)
    ax.tick_params(labelsize=9)
fig.suptitle("Sparsity of problem.linearize(x)[0].dense() for a Panda trajectory (pose + velocity + acceleration terms)", fontsize=12)
fig.tight_layout()
fig.savefig(sys.argv[1], dpi=100, facecolor="white")
