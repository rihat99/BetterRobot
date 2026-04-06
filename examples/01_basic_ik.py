"""Interactive IK example for Franka Panda.

Usage:
    uv run python examples/01_basic_ik.py

Open http://localhost:8080 in your browser.
Drag the transform handle to move the target end-effector pose.
Click *Restart* to reset the robot and target to the default configuration.

Solver options (pass via IKConfig):
    jacobian="analytic"   — geometric Jacobian (faster, default here)
    jacobian="autodiff"   — torch.func.jacrev (works for any custom cost)
    solver="lm"           — Levenberg-Marquardt (default)
    solver="gn"           — Gauss-Newton (no damping, faster on easy problems)
    solver="adam"         — Adam gradient descent
    solver="lbfgs"        — L-BFGS with strong Wolfe line search
"""
import time
import better_robot as br
from robot_descriptions.loaders.yourdfpy import load_robot_description


def main() -> None:
    urdf = load_robot_description("panda_description")
    model = br.load_urdf(urdf)
    print(f"Loaded Panda: {model.links.num_links} links, {model.joints.num_actuated_joints} actuated joints")

    vis = br.Visualizer(urdf, model)
    vis.add_target("panda_hand", scale=0.15)
    vis.add_timing_display()
    vis.add_restart_button()

    ik_cfg = br.IKConfig(rest_weight=0.001, jacobian="analytic")

    cfg = model._default_cfg.clone()
    vis.reset_targets(model, cfg)
    vis.update(cfg)

    print("Drag the transform handle to set the IK target. Press Ctrl+C to quit.")

    while True:
        if vis.restart_requested:
            cfg = model._default_cfg.clone()
            vis.reset_targets(model, cfg)

        t0 = time.perf_counter()
        cfg = br.solve_ik(
            model,
            targets=vis.get_targets(),
            cfg=ik_cfg,
            initial_cfg=cfg,
            max_iter=20,
        )
        vis.set_timing((time.perf_counter() - t0) * 1000)
        vis.update(cfg)
        time.sleep(1.0 / 30.0)


if __name__ == "__main__":
    main()
