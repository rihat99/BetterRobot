# Visualize a robot

You open the browser viewer, show the Panda moving along a trajectory, step through it, and record it to a
GIF without a browser. This page needs the `viewer` and `meshes` extras (see [Install](installation.md)).

## Make a clip to show

The viewer shows clips: every layer holds a whole time series, `q (T, nq)`, and a shared timeline picks the
frame. Here is a 30-frame clip that swings the Panda's first joint:

```python
import tempfile
from pathlib import Path

import torch
import better_robot as br
from robot_descriptions import panda_description
from better_robot.viewer import Viewer, robot_layers, RobotView

model = br.load(panda_description.URDF_PATH, dtype=torch.float64)
q_rest = model.q_neutral.clamp(model.lower_pos_limit, model.upper_pos_limit)
clip = q_rest.repeat(30, 1)                       # (T, nq)
clip[:, 0] = torch.linspace(-1.0, 1.0, 30, dtype=torch.float64)
```

Any `(T, nq)` tensor works: the `result.x["q"][0]` of [the trajectory tutorial](trajectories.md) is one.

## Open the viewer and add the robot

`Viewer()` starts a local [viser](https://viser.studio) server on `127.0.0.1:8080`; open `viewer.url` in a
browser. `robot_layers` turns a model and a clip into layers: the visual meshes by default, plus a skeleton,
collision geometry or frame axes on request:

```python
viewer = Viewer()
for layer in robot_layers(model, clip, frames=["body_panda_hand"]):
    viewer.add(layer)
print(viewer.url, len(viewer.layers))
```

Forward kinematics runs once for the whole clip, not once per frame.

## Step through the timeline

`viewer.show(frame)` moves the timeline and pushes that frame to every visible layer:

```python
viewer.show(0)
viewer.show(29)
```

`viewer.play()` runs the timeline at the viewer's `fps` and blocks until you stop it, so call it last in a
script:

```python no-run
viewer.play()
```

## Update the robot in place

When you keep a handle on the robot, use `RobotView` instead of `robot_layers`: it owns its layers, and
`update(q)` recomputes them for a new clip of the same length, for example after each solver step. Call
`viewer.show()` afterwards to push the edited arrays:

```python
view = RobotView(model, clip, name="panda_live", skeleton=True)
for layer in view.layers:
    viewer.add(layer, group="live")
view.update(clip.flip(0))
viewer.show()
```

## Record a GIF without a browser

`viewer.record` renders the visible layers offline with a PyTorch rasterizer and writes `.gif`, `.mp4` or
`.png` frames. Without a `camera` it frames the robot automatically:

```python
viewer.remove_group("live")
out = Path(tempfile.mkdtemp()) / "panda_swing.gif"
viewer.record(out, size=(320, 240), frames=range(0, 30, 3))
print(out.exists(), out.stat().st_size > 0)
```

The same layers render in CI and on headless machines, since `record` never touches the server.
[Viewer](../concepts/viewer.md) explains layers, groups, tools and themes.
