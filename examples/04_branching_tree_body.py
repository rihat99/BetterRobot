"""Display a programmatically-built branching-tree skeleton using SkeletonMode.

Demonstrates that the viewer works on robots built with the generic
``build_kinematic_tree_model`` — no URDF and no collision geometry —
SkeletonMode is always available.

Usage:
    uv run python examples/04_branching_tree_body.py [--no-viewer]
"""

import argparse

import torch

import better_robot as br
from better_robot.io.builders.kinematic_tree import build_kinematic_tree_model


def make_branching_tree() -> br.Model:
    """Build a 24-body free-flyer + 23-spherical tree with arbitrary offsets."""
    joint_names = tuple(f"j{index}" for index in range(24))
    parents = (-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 12, 12, 12, 13, 14, 16, 17, 18, 19, 20, 21)
    translations = torch.zeros(24, 3)
    translations[1:, 2] = 0.15  # give every non-root joint a visible offset
    return build_kinematic_tree_model(
        name="branching_tree",
        joint_names=joint_names,
        parents=parents,
        translations=translations,
        root_kind="free_flyer",
        child_kind="spherical",
        preserve_joint_order=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-viewer", action="store_true", help="Skip the viser viewer (useful in CI)")
    args = parser.parse_args()

    model = make_branching_tree()
    print(f"Branching-tree body: njoints={model.njoints}, nq={model.nq}, nframes={model.nframes}")

    # Forward kinematics at neutral pose
    q0 = model.q_neutral
    data = br.forward_kinematics(model, q0, compute_frames=True)
    print(f"Root position: {data.joint_pose_world[0, :3]}")

    if not args.no_viewer:
        from better_robot.viewer import Visualizer  # noqa: PLC0415 - optional viewer extra

        viewer = Visualizer(model, port=8081)
        viewer.update(q0)
        print("\nOpening viewer at http://localhost:8081 — press Ctrl-C to exit.")
        print("Only SkeletonMode is available (no URDF meshes for a programmatic body).")
        viewer.show(block=True)
    else:
        print("Viewer skipped (--no-viewer).")


if __name__ == "__main__":
    main()
