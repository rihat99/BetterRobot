"""Display a 24-joint SMPL-like body skeleton using SkeletonMode.

Demonstrates that the viewer works on programmatically-built robots with
no URDF and no collision geometry — SkeletonMode is always available.

Usage:
    uv run python examples/04_smpl_like_body.py [--no-viewer]
"""
import argparse
import torch
import better_robot as br
from better_robot.io.builders.smpl_like import build_smpl_like_model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-viewer", action="store_true",
                        help="Skip the viser viewer (useful in CI)")
    args = parser.parse_args()

    model = build_smpl_like_model()
    print(f"SMPL-like body: njoints={model.njoints}, nq={model.nq}, nframes={model.nframes}")

    # Forward kinematics at neutral pose
    q0 = model.q_neutral
    data = br.forward_kinematics(model, q0, compute_frames=True)
    print(f"Root position: {data.oMi[0, :3]}")

    if not args.no_viewer:
        from better_robot.viewer import Visualizer
        viewer = Visualizer(model, port=8081)
        viewer.update(q0)
        print("\nOpening viewer at http://localhost:8081 — press Ctrl-C to exit.")
        print("Only SkeletonMode is available (no URDF meshes for a programmatic body).")
        viewer.show(block=True)
    else:
        print("Viewer skipped (--no-viewer).")


if __name__ == "__main__":
    main()
