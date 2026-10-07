# Visualize a robot

The optional viewer displays robots, trajectories, forces and plots in a
browser. Install it alongside the demo descriptions to follow this guide with
Panda:

```bash
python -m pip install '.[viewer,meshes,demos]'
```

Layers are built from arrays before any server exists, so the setup below is
safe in the documentation test environment, where `viser` is intentionally
absent; only `Viewer()` needs it.

```{testcode}
import torch
import better_robot as br
from better_robot.viewer import GroundLayer, robot_layers
from robot_descriptions import panda_description

model = br.load(panda_description.URDF_PATH)
q = torch.stack([model.random_configuration() for _ in range(30)])  # (T, nq)
layers = [GroundLayer()] + robot_layers(model, q, skeleton=True, frames=["body_panda_hand"])

print([layer.name for layer in layers], layers[1].num_frames)
```

```{testoutput}
['ground', 'robot', 'robot skeleton', 'robot frames'] 30
```

With the viewer extra installed, the interactive part is:

```text
from better_robot.viewer import Viewer

viewer = Viewer()                 # http://127.0.0.1:8080
for layer in layers:
    viewer.add(layer, group="panda")
viewer.add_plot("torque", tau, labels=[...])   # optional (T, K) series
viewer.play()                     # blocks; viewer.show(k) jumps to a frame
```

Groups are the unit for toggling and teardown: `viewer.remove_group("panda")`
removes every layer of the group with one scene call, and `add(..., offset=
(1.0, 0.0, 0.0))` moves a group aside for side-by-side comparisons.

For live tools, keep the `RobotView` and rewrite it in place:

```text
from better_robot.viewer import RobotView
from better_robot.viewer.tools import PoseGizmo

robot = RobotView(model, q0)
for layer in robot:
    viewer.add(layer)

def follow(target):               # target: (7,) xyzw pose from the gizmo
    q = solve(target)
    robot.update(q)
    viewer.show()

PoseGizmo(viewer, "hand", pose0, on_change=follow)
```

Recording needs no browser: `viewer.record("clip.gif")` renders the visible
layers with the torch rasterizer (`.png` for frames, `.mp4` with `ffmpeg`),
and `viewer.save("clip.viser")` writes a file for viser's static client.

Viewer symbols live under `better_robot.viewer`; they are not top-level
`better_robot` exports. Programmatically built models without mesh metadata
get a skeleton only (`skeleton=True`).
