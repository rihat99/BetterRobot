"""Reproduce the docs screenshots and clips with ``Viewer.record`` (headless Chromium + ffmpeg).

    uv run python agent_notes/tasks/2026-10-07_16-44_br_bh_polish_contracts_viewer_docs/subagents/make_clips.py --out docs/

Env: ``BETTER_ROBOT_CHROME`` (Chromium / chrome-headless-shell binary), ``FFMPEG`` (ffmpeg for ``.webm``),
``LD_LIBRARY_PATH`` with ``libasound.so.2`` on machines without it. Writes ``images/{panda_rest,panda_ik,
g1_floating,trajectory_strip}.png``, ``images/readme_ik.gif`` and ``clips/<name>.webm`` + ``clips/<name>.png``.
Example results are taken from each example's ``main`` locals, so the clips follow the examples.
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from robot_descriptions import g1_description, panda_description

import better_robot as br
from better_robot import optim, terms
from better_robot.viewer import GroundLayer, SpheresLayer, Viewer, robot_layers
from better_robot.viewer.capture import fit_camera, write_frames

EXAMPLES = Path(__file__).resolve().parents[4] / "examples"
SIZE, FPS, HAND = (960, 540), 24.0, "body_panda_hand"


_OPEN: list[Viewer] = []


def new_viewer() -> Viewer:
    """A viewer on port 8094 (the only port this script uses); stops the previous one first."""
    while _OPEN:
        _OPEN.pop().server.stop()
    _OPEN.append(Viewer(port=8094, fps=FPS))
    return _OPEN[-1]


def example_locals(name: str) -> tuple[object, dict]:
    """Run ``examples/<name>.py::main`` and return the module and ``main``'s locals at return."""
    spec = importlib.util.spec_from_file_location(name, EXAMPLES / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    captured: dict = {}

    def profiler(frame, event, arg) -> None:  # noqa: ANN001
        if event == "return" and frame.f_code is module.main.__code__:
            captured.update(frame.f_locals)

    sys.setprofile(profiler)
    try:
        module.main()
    finally:
        sys.setprofile(None)
    return module, captured


def upsample(x: torch.Tensor, factor: int) -> torch.Tensor:
    """Linear interpolation of ``x (T, D)`` to ``(factor * (T - 1) + 1, D)`` (revolute joints only)."""
    alpha = torch.linspace(0, 1, factor + 1, dtype=x.dtype)[:-1, None, None]
    steps = (1 - alpha) * x[None, :-1] + alpha * x[None, 1:]
    return torch.cat([steps.transpose(0, 1).reshape(-1, x.shape[-1]), x[-1:]])


def ground_for(model: br.Model, q: torch.Tensor, size: float) -> GroundLayer:
    """A ground grid of ``size`` metres just under the lowest frame of ``q (T, nq)``."""
    z = br.forward_kinematics_state(model, q).frame_translation[..., 2].min()
    return GroundLayer(size=size, height=min(float(z), 0.0))


def shoot(viewer: Viewer, steps: list[int], camera=None) -> list[np.ndarray]:  # noqa: ANN001
    """Frames of ``steps`` as ``(H, W, 3)`` arrays."""
    with tempfile.TemporaryDirectory() as tmp:
        viewer.record(Path(tmp) / "f", steps=steps, size=SIZE, camera=camera)
        return [np.asarray(Image.open(p).convert("RGB")) for p in sorted(Path(tmp, "f").glob("*.png"))]


def clip(out: Path, name: str, frames: list[np.ndarray], poster: int | None = None) -> None:
    """Write ``clips/<name>.webm`` and its poster PNG; print size and frame count."""
    path = write_frames(frames, out / "clips" / f"{name}.webm", fps=FPS)
    Image.fromarray(frames[len(frames) // 2 if poster is None else poster]).save(out / "clips" / f"{name}.png")
    print(f"{name}.webm {path.stat().st_size / 1e6:.2f} MB, {len(frames)} frames", flush=True)


def target_marker(target: torch.Tensor) -> SpheresLayer:
    """A translucent red ball at the position of ``target (1, 7)``."""
    return SpheresLayer("target", target[:, :3].numpy(), np.array([0.035]), colors=(0.9, 0.2, 0.2), opacity=0.55)


def still(out: Path, name: str, viewer: Viewer, step: int = 0) -> None:
    Image.fromarray(shoot(viewer, [step])[0]).save(out / "images" / f"{name}.png")
    print(f"{name}.png", flush=True)


def batched_ik(panda: br.Model, copies: int = 8, iterations: int = 12) -> tuple[torch.Tensor, torch.Tensor]:
    """LM iterates ``(iterations + 1, copies, nq)`` of ``copies`` Panda IK problems and their targets ``(copies, 7)``."""
    generator = torch.Generator().manual_seed(1)
    mid = (panda.lower_pos_limit + panda.upper_pos_limit) / 2
    q0 = mid.repeat(copies, 1)
    q_goal = mid + 0.6 * (torch.rand(copies, panda.nq, generator=generator, dtype=torch.float64) - 0.5) * (
        panda.upper_pos_limit - panda.lower_pos_limit
    )
    target = br.forward_kinematics_state(panda, q_goal).frame_pose[:, panda.frame_id(HAND)]
    iterates = [q0]
    for k in range(1, iterations + 1):
        problem = optim.Problem(
            {"q": q0.clone()},
            manifolds={"q": terms.RobotManifold(panda)},
            bounds={"q": (panda.lower_pos_limit, panda.upper_pos_limit)},
            data={"target": target},
        )
        problem.cached(terms.kinematic_state(panda), name="state")
        problem.residual("pose", terms.frame_pose(panda, HAND, target="target"))
        iterates.append(optim.solve(problem, optim.LevenbergMarquardt(), iters=k).x["q"])
    return torch.stack(iterates), target


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("docs"))
    parser.add_argument("--only", nargs="*", default=None, help="subset of: stills batched trajectory g1 layers obstacle")
    args = parser.parse_args()
    out, only = args.out, set(args.only or ["stills", "batched", "trajectory", "g1", "layers", "obstacle"])
    (out / "clips").mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    panda = br.load(panda_description.URDF_PATH, dtype=torch.float64)
    ready = torch.tensor([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785, 0.04, 0.04], dtype=torch.float64)[: panda.nq]

    if "stills" in only or "batched" in only:
        iterates, targets = batched_ik(panda)
    if "stills" in only:
        viewer = new_viewer()
        viewer.add(ground_for(panda, ready[None], 1.2))
        for layer in robot_layers(panda, ready[None]):
            viewer.add(layer, group="panda")
        still(out, "panda_rest", viewer)

        viewer = new_viewer()
        viewer.add(ground_for(panda, iterates[-1, :1], 1.2))
        for layer in robot_layers(panda, iterates[-1, :1], frames=[HAND]):
            viewer.add(layer, group="panda")
        viewer.add(target_marker(targets[:1]))
        still(out, "panda_ik", viewer)

        g1 = br.load(g1_description.URDF_PATH, free_flyer=True, dtype=torch.float64)
        q = g1.q_neutral[None]
        viewer = new_viewer()
        viewer.add(ground_for(g1, q, 2.0))
        pelvis = next(n for n in g1.frame_names if "pelvis" in n)
        for layer in robot_layers(g1, q, frames=[pelvis], opacity=0.45):
            viewer.add(layer, group="g1")
        still(out, "g1_floating", viewer)

    if "stills" in only or "trajectory" in only:
        module, local = example_locals("02_trajectory")
        q = local["q"].detach()
        if "stills" in only:
            viewer = new_viewer()
            row = np.array([0.82, 0.57, 0.0])  # across the 3/4 view fit_camera picks for z-up, so the poses do not recede
            viewer.add(GroundLayer(size=3.6, height=0.0), group="ground", offset=tuple(1.35 * row))
            for index, knot in enumerate(np.linspace(0, len(q) - 1, 4).round().astype(int)):
                for layer in robot_layers(panda, q[knot : knot + 1], name=f"robot{index}"):
                    viewer.add(layer, group=f"pose{index}", offset=tuple(0.9 * index * row))
            still(out, "trajectory_strip", viewer)
        if "trajectory" in only:
            dt = float(getattr(module, "DT", 0.05))
            velocity = torch.gradient(q, spacing=dt, dim=0)[0]
            acceleration = torch.gradient(velocity, spacing=dt, dim=0)[0]
            tau = br.inverse_dynamics(panda, br.forward_kinematics_state(panda, q), v=velocity, a=acceleration)
            q_fine, tau_fine = upsample(q, 2), upsample(tau, 2)
            viewer = new_viewer()
            viewer.add(ground_for(panda, q_fine, 1.2))
            for layer in robot_layers(panda, q_fine, skeleton=True, tau=tau_fine, opacity=0.35):
                viewer.add(layer, group="panda")
            clip(out, "panda_trajectory", shoot(viewer, list(range(len(q_fine)))))

    if "batched" in only:
        q_fine = upsample(iterates.reshape(len(iterates), -1), 6).reshape(-1, *iterates.shape[1:])
        viewer = new_viewer()
        for index in range(q_fine.shape[1]):
            offset = (0.9 * (index // 4), 0.9 * (index % 4), 0.0)
            viewer.add(GroundLayer(size=0.85, height=0.0, cell_size=0.1, section_size=0.85), group=f"ik{index}", offset=offset)
            for layer in robot_layers(panda, q_fine[:, index], frames=[HAND]):
                viewer.add(layer, group=f"ik{index}")
            viewer.add(target_marker(targets[index : index + 1]), group=f"ik{index}")
        steps = list(range(len(q_fine))) + [len(q_fine) - 1] * 12
        clip(out, "batched_ik", shoot(viewer, steps), poster=len(steps) - 1)
        viewer.record(out / "images" / "readme_ik.gif", steps=steps[::2], size=(720, 405), fps=12.0)
        print(f"readme_ik.gif {(out / 'images' / 'readme_ik.gif').stat().st_size / 1e6:.2f} MB, {len(steps[::2])} frames")

    if "g1" in only:
        module, local = example_locals("03_clip_fit_phases")
        model, q = local["model"], local["result"].x["q"][0].detach()
        viewer = new_viewer()
        viewer.add(ground_for(model, q, 3.0))
        pelvis = next(n for n in model.frame_names if "pelvis" in n)
        for layer in robot_layers(model, q, frames=[pelvis]):
            viewer.add(layer, group="g1")
        clip(out, "g1_clip_fit", shoot(viewer, list(range(len(q)))))

    if "layers" in only:
        phase = torch.linspace(0, 2 * torch.pi, 80, dtype=torch.float64)[:, None]
        q = ready + 0.35 * torch.sin(phase + torch.arange(panda.nq) * 0.7)
        viewer = new_viewer()
        ground = viewer.add(ground_for(panda, q, 1.2))
        layers = robot_layers(panda, q, skeleton=True, frames=[HAND, "body_panda_link4"])
        for layer in layers:
            viewer.add(layer, group="panda")
        mesh = [layer for layer in layers if "skeleton" not in layer.name and "frame" not in layer.name]
        skeleton = [layer for layer in layers if "skeleton" in layer.name]
        frames_ = [layer for layer in layers if "frame" in layer.name]
        camera = fit_camera(viewer._points(list(range(len(q)))), up=viewer.theme.up_vector)
        stages = [mesh, mesh + skeleton, mesh + skeleton + frames_, [*layers, ground]]
        frames = []
        for stage, visible in enumerate(stages):
            for layer in [*layers, ground]:
                viewer.set_visible(layer, layer in visible)
            frames += shoot(viewer, list(range(20 * stage, 20 * stage + 20)), camera=camera)
        clip(out, "viewer_layers", frames, poster=len(frames) - 1)

    if "obstacle" in only:
        module, local = example_locals("06_obstacle_trajectory")
        q_fine = upsample(local["q"].detach(), 3)
        viewer = new_viewer()
        viewer.add(ground_for(panda, q_fine, 1.4))
        for layer in robot_layers(panda, q_fine, spheres=local["spheres"]):
            viewer.add(layer, group="panda")
        viewer.add(SpheresLayer("obstacle", local["obstacle"][None].numpy(), local["radius"].reshape(1).numpy(), opacity=0.8))
        clip(out, "obstacle_trajectory", shoot(viewer, list(range(len(q_fine)))))
    while _OPEN:
        _OPEN.pop().server.stop()
    print(f"total {time.perf_counter() - start:.0f} s")


if __name__ == "__main__":
    main()
