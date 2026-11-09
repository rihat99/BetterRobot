"""Display a branching tree built with ``ModelBuilder`` as a skeleton.

The viewer works on a robot built in Python, with no URDF and no geometry: ``robot_skeleton`` draws the joints
and the bones between them.

Usage:
    uv run python examples/07_branching_tree_body.py [--no-viewer | --check]
"""

import argparse

import torch

import better_robot as br


def make_branching_tree() -> br.Model:
    """A 24-body tree: a free-flyer root and 23 spherical joints, each 15 cm above its parent."""
    names = [f"j{index}" for index in range(24)]
    parents = (-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 12, 12, 12, 13, 14, 16, 17, 18, 19, 20, 21)
    builder = br.ModelBuilder("branching_tree")
    for name in names:
        builder.add_body(name)
    builder.add_free_flyer_root("root", child=names[0])
    for index in range(1, len(names)):
        builder.add_spherical(
            names[index],
            parent=names[parents[index]],
            child=names[index],
            origin=torch.tensor([0.0, 0.0, 0.15, 0.0, 0.0, 0.0, 1.0]),
        )
    return br.io.build_model(builder.finalize())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-viewer", action="store_true", help="Skip the viser viewer (useful in CI)")
    parser.add_argument("--check", action="store_true", help="build the scene on the stand-in server and exit")
    args = parser.parse_args()

    model = make_branching_tree()
    print(f"Branching-tree body: njoints={model.njoints}, nq={model.nq}, nframes={model.nframes}")

    # Forward kinematics at neutral pose
    q0 = model.q_neutral
    state = br.forward_kinematics(model, q0)
    print(f"Root position: {state.joint_poses[0, :3]}")

    if not args.no_viewer:
        from better_robot.testing import FakeServer  # noqa: PLC0415
        from better_robot.viewer import GroundLayer, Viewer, robot_skeleton  # noqa: PLC0415 - optional viewer extra

        viewer = Viewer(port=8081, server=FakeServer() if args.check else None)
        viewer.add(
            [GroundLayer(), robot_skeleton(model, state.joint_poses.numpy())]
        )  # no meshes for a programmatic body
        viewer.show()
        if args.check:
            print(f"{len(viewer.layers)} layers built on the stand-in server")
            return
        print(f"\nOpening viewer at {viewer.url} — press Ctrl-C to exit.")
        viewer.server.sleep_forever()
    else:
        print("Viewer skipped (--no-viewer).")


if __name__ == "__main__":
    main()
