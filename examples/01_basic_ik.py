"""Interactive IK example for Franka Panda.

Usage:
    uv run python examples/01_basic_ik.py

Open http://localhost:8080 in your browser.
Drag the transform handle to move the target end-effector pose.
Click *Restart* to reset the robot and target to the default configuration.
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
    vis.add_restart_button()

    cfg = model._default_cfg.clone()
    vis.reset_targets(model, cfg)
    vis.update(cfg)

    print("Drag the transform handle to set the IK target. Press Ctrl+C to quit.")

    while True:
        if vis.restart_requested:
            cfg = model._default_cfg.clone()
            vis.reset_targets(model, cfg)

        cfg = br.solve_ik(
            model,
            targets=vis.get_targets(),
            cfg=br.IKConfig(rest_weight=0.001, jacobian="analytic"),
            initial_cfg=cfg,
            max_iter=20,
        )
        vis.update(cfg)
        time.sleep(1.0 / 30.0)


if __name__ == "__main__":
    main()
