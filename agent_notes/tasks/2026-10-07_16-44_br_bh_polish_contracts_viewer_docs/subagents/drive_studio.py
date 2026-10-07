"""Drive HumanStudio on a real viser server (127.0.0.1:8095) and screenshot it.

Run from the BH repo, output to a log FILE (never a pipe):
  CUDA_VISIBLE_DEVICES="" BETTERHUMAN_MODELS_DIR=$PWD/models BETTER_ROBOT_CHROME=... LD_LIBRARY_PATH=... \
    uv run python <this> <data_dir> <png_dir> > drive.log 2>&1
"""

from __future__ import annotations

import faulthandler
import os
import sys
import time
from pathlib import Path

import numpy as np
import viser
from viser._gui_api import GuiEvent

from better_human.viz.studio.app import HumanStudio
from better_robot.viewer.capture import HeadlessClient, fit_camera

DATA, PNG = Path(sys.argv[1]), Path(sys.argv[2])
PNG.mkdir(parents=True, exist_ok=True)
TIMES: dict[str, float] = {}


def step(name: str, fn, *args):  # noqa: ANN001, ANN201
    faulthandler.dump_traceback_later(30, exit=True)
    start = time.perf_counter()
    out = fn(*args)
    TIMES[name] = time.perf_counter() - start
    faulthandler.cancel_dump_traceback_later()
    print(f"OK {name} {TIMES[name]:.3f}s status={studio.status if 'studio' in globals() else ''!r}", flush=True)
    return out


def handle(label: str):  # noqa: ANN201
    found = [h for h in server.gui._gui_input_handle_from_uuid.values() if h.label == label]
    assert len(found) == 1, (label, len(found))
    return found[0]


def click(target) -> None:  # noqa: ANN001
    target = handle(target) if isinstance(target, str) else target
    for cb in target._impl.update_cb:
        cb(GuiEvent(client_id=None, client=None, target=target))


def set_value(target, value) -> None:  # noqa: ANN001
    (handle(target) if isinstance(target, str) else target).value = value


print("start", flush=True)
server = step("server", viser.ViserServer, "127.0.0.1", 8095)
studio = step("init_smplx", lambda: HumanStudio(family="smplx", port=8095, server=server, data_dir=DATA))
for family in ("smpl", "mhr", "smplx"):
    set_value(studio.family_input, family)
    step(f"model_{family}", click, "Load")
    assert studio.session.model_key.startswith(family), studio.session.model_key

shape0 = sorted(h.label for h in server.gui._gui_input_handle_from_uuid.values() if h.label.endswith("[0]"))
print("shape sliders", shape0, flush=True)
group = shape0[0].split("[")[0]
step("shape_tick_skin", set_value, shape0[0], 1.5)
step("random", click, f"Random {group}")
step("reset", click, f"Reset {group}")
density0 = studio.density_input.value  # a number input unless per-part density tables exist
set_value(studio.density_input, 1100.0 if isinstance(density0, float) else studio.density_input.options[-1])
step("density", click, "Apply density")
set_value(studio.density_input, density0)
step("density_back", click, "Apply density")
set_value(studio.pose_name, "drive_pose")
step("save_pose", click, "Save pose")
set_value(studio.saved_poses, "drive_pose")
step("load_pose", click, "Load pose")
step("reset_pose", click, "Reset pose")
step("opacity", set_value, "Opacity", 0.6)
step("radius", set_value, "Joint radius", 0.02)
step("explode", set_value, "Explode", 0.3)
step("explode0", set_value, "Explode", 0.0)
step("opacity1", set_value, "Opacity", 1.0)
seg = studio.segment
for mode in ("segmentation", "parts", "skin"):
    step(f"mode_{mode}", set_value, seg.color_mode, mode)
    assert studio.layers.mesh.mode == mode
step("builtin", click, "Load built-in")
step("mode_parts2", set_value, seg.color_mode, "parts")
step("shape_tick_parts", set_value, shape0[0], 1.0)
step("shape_tick_parts2", set_value, shape0[0], 0.5)
step("add_part", seg.add_part, "drive_part")
step("assign", seg.assign, np.arange(200))
set_value(seg.save_name, "drive_seg")
step("save_seg", click, "Save segmentation")
assert (studio.paths.root / "segmentations").exists(), list(studio.paths.root.iterdir())

fr = studio.frames
print("frame sets", fr.saved.options, flush=True)
set_value(fr.saved, "contacts")
step("load_contacts", click, "Load frames")
n0 = len(studio.session.frame_set)
step("add_frame", fr.add_at, studio.session.vertices.reshape(-1, 3)[100].detach().cpu().numpy())
assert len(studio.session.frame_set) == n0 + 1
set_value(fr.save_name, "drive_frames")
step("save_frames", click, "Save frames")
assert "drive_frames" in fr.saved.options, fr.saved.options
set_value(fr.saved, "drive_frames")
step("reload_frames", click, "Load frames")
assert len(studio.session.frame_set) == n0 + 1

step("select_joint", studio.select_joint, 18)
pose = studio.gizmo._world_pose(18).copy()
pose[:3] += 0.05
h = studio.gizmo.handle_for(18)
step("gizmo_drag", studio.gizmo._drag_locked, h, pose)
step("deselect", click, "Deselect joint")

# screenshots
verts = studio.session.vertices.reshape(-1, 3).detach().cpu().numpy()
eye, look = fit_camera(verts, up=(0.0, 1.0, 0.0))
with HeadlessClient(studio.viewer, size=(800, 900)) as cam:
    cam.set_camera(eye, look, (0.0, 1.0, 0.0))
    import imageio.v3 as iio

    for mode in ("skin", "segmentation", "parts"):
        step(f"shot_{mode}", set_value, seg.color_mode, mode)
        time.sleep(1.0)
        iio.imwrite(PNG / f"studio_{mode}.png", cam.render(900, 800))
    step("skin_for_frames", set_value, seg.color_mode, "skin")
    time.sleep(1.0)
    iio.imwrite(PNG / "studio_frames_skeleton.png", cam.render(900, 800))

for name in list(studio.session.frame_set):
    step(f"delete_{name}", fr.delete, name)
assert len(studio.session.frame_set) == 0
step("empty_frames_mode", set_value, seg.color_mode, "segmentation")
print("TIMES", {k: round(v, 3) for k, v in TIMES.items()}, flush=True)
server.stop()
print("DONE", flush=True)
os._exit(0)
