"""Drive HumanStudio on a real viser server (127.0.0.1:8082): pointer click, selection row, joint select, screenshots.

Copy of 2026-10-08_13-02 drive_studio_p2.py plus a scene click dispatched through the callback that
``Viewer.on_pointer`` registered on the real viser server (``server.scene._scene_pointer_cb``).

Run from the BH worktree, log to a file:
  CUDA_VISIBLE_DEVICES="" BETTERHUMAN_MODELS_DIR=$PWD/models CHROME=... LD_LIBRARY_PATH=... \
    uv run python <this> <data_dir> <png_dir> <shot_studio.mjs> > drive.log 2>&1
Adapted from the previous task's drive_studio.py (same click helper).
"""

from __future__ import annotations

import subprocess
import sys
import time
from types import SimpleNamespace
from pathlib import Path

import numpy as np
import viser
from viser._gui_api import GuiEvent

from better_human.viz.studio.app import HumanStudio
from better_robot.viewer.capture import fit_camera

DATA, PNG, SHOT = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
PNG.mkdir(parents=True, exist_ok=True)
URL = "http://127.0.0.1:8082"


def click(target, value=None) -> None:  # noqa: ANN001
    if value is not None:
        target.value = value
    for cb in target._impl.update_cb:
        cb(GuiEvent(client_id=None, client=None, target=target))


def shot(name: str) -> None:
    time.sleep(1.0)
    subprocess.run(["node", SHOT, URL, str(PNG / f"{name}.png")], check=True, timeout=120)
    print(f"shot {name}: status={studio.status!r} selection={studio.selection_text.content!r}", flush=True)


server = viser.ViserServer("127.0.0.1", 8082)
try:
    studio = HumanStudio(family="smplx", port=8082, server=server, data_dir=DATA)
    eye, look = fit_camera(studio.session.vertices.detach().cpu().numpy(), up=(0.0, 1.0, 0.0))
    server.initial_camera.position, server.initial_camera.look_at = tuple(eye), tuple(look)
    print("selection:", studio.selection_text.content, flush=True)
    shot("studio_before_select")
    entries = [e for e in server.scene._scene_pointer_cb if e.event_type == "click"]
    print("pointer slots:", [(e.event_type, getattr(e.callback, "__name__", e.callback)) for e in entries], flush=True)
    joint = studio.session.state.joints.detach().cpu().reshape(-1, 3)[18].numpy().astype(float)
    origin = joint + np.array([0.0, 0.0, 2.0])
    event = SimpleNamespace(client=None, client_id=None, ray_origin=tuple(origin), ray_direction=tuple(joint - origin),
                            screen_pos=((0.5, 0.5),), modifier=None, event_type="click")
    entries[0].callback(event)
    print("after pointer click:", studio.selection_text.content, "| status:", studio.status, flush=True)
    shot("studio_pointer_click")
    studio.select_joint(18)
    studio.gizmo._on_drag(studio.gizmo.handle_for(18), studio.gizmo._world_pose(18))
    studio.frames.add_at(studio.session.vertices[3000].detach().cpu().numpy())
    print("after select:", studio.selection_text.content, "| depth_test:",
          studio.gizmo.gizmo.handle._impl.props.depth_test, "| scale:", studio.gizmo.scale, flush=True)
    shot("studio_joint_selected")
    for action in ("Deselect frame", "Deselect joint", "Reset pose"):
        click(studio.actions[action])
        print(f"clicked {action!r}: {studio.selection_text.content!r} active_frame={studio.frames.active!r} "
              f"gizmo={studio.gizmo.gizmo is not None}", flush=True)
    shot("studio_after_deselect")
finally:
    server.stop()
    print("server stopped", flush=True)
