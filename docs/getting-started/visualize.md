# Visualize a robot

You open the browser viewer, show the Panda moving along a trajectory, step through it, and record it to a
GIF through a headless browser. This page needs the `viewer` and `meshes` extras (see [Install](installation.md)).

<figure markdown="span"><video autoplay loop muted playsinline preload="metadata" poster="../../clips/viewer_layers.png" width="960"><source src="../../clips/viewer_layers.webm" type="video/webm"></video>
<figcaption>One <code>RobotVisual</code> four ways: meshes, the maximal skeleton, the minimal skeleton, and the maximal one with joint frames, switched with <code>set_style</code> and <code>set_frames</code>.</figcaption></figure>

## Make a clip to show

The viewer shows clips: every layer holds a whole time series, `q (T, nq)`, and a shared timeline picks the
step. Here is a 30-step clip that swings the Panda's first joint:

```python
import tempfile
from pathlib import Path

import torch
import better_robot as br
from robot_descriptions import panda_description
from better_robot import Viewer
from better_robot.viewer import RobotVisual

URDF = panda_description.URDF_PATH
model = br.load(URDF, dtype=torch.float64)
q_rest = model.q_neutral
clip = q_rest.repeat(30, 1)                       # (T, nq)
clip[:, 0] = torch.linspace(-1.0, 1.0, 30, dtype=torch.float64)
```

Any `(T, nq)` tensor works: the `result.x["q"][0]` of [the trajectory tutorial](trajectories.md) is one.

## Open the viewer and add the robot

`Viewer()` starts a local [viser](https://viser.studio) server on `127.0.0.1:8080`; open `viewer.url` in a
browser. A `RobotVisual` is the robot as one layer: it runs forward kinematics once for the whole clip, draws the
meshes of the description's `Geometry` (found under `root`, the file you loaded) and, with `frames=`, the axes of
the frames you name. The viewer adds the ground grid itself (`Viewer(ground=False)` leaves it out), so it holds two
layers:

```python
viewer = Viewer()
geometry = br.io.load_geometry(URDF, model)
robot = viewer.add(RobotVisual(model, clip, geometry=geometry, root=URDF, frames=["panda_hand"]))
print(viewer.url, len(viewer.layers), robot.style)
```
```output
http://127.0.0.1:8080 2 meshes
```

The sidebar's Robot section switches the robot between its meshes and two skeleton styles, and
`robot.set_style("maximal")` does the same from code.

## Step through the timeline

`viewer.show(step)` moves the timeline and pushes that step to every visible layer:

```python
viewer.show(0)
viewer.show(29)
```

`viewer.serve()` plays the timeline at the viewer's `fps` and blocks until you stop it, so call it last in a
script:

```python { .no-run data-reason="blocks until Ctrl-C" }
viewer.serve()
```

## Update the robot in place

To show a new clip of the same length, for example after each solver step, call `update`: one forward-kinematics
pass, then the robot redraws itself. Here a second robot, drawn as a skeleton, plays the clip backwards:

```python
live = viewer.add(RobotVisual(model, clip, name="panda_live", style="maximal"), tags="live")
live.update(clip.flip(0))
```

## Record a video without a browser window

`viewer.record` draws the live scene in a headless Chromium and writes an `.mp4` or `.webm` video, or a `.png`
still. Without a `camera` it frames the robot automatically:

```python { .no-run data-reason="records through a headless Chromium; tests/viewer/test_record.py runs record" }
viewer.remove("live")
out = viewer.record(Path(tempfile.mkdtemp()) / "panda_swing.webm", size=(640, 360), steps=range(0, 30, 3))
print(out.exists(), out.stat().st_size > 0)
```

It needs Chromium (`$BETTER_ROBOT_CHROME` or on `PATH`) and, for `.mp4`, ffmpeg; see
[Record a video of a scene](../guides/record-a-video.md).

## Where next

- [Viewer](../concepts/viewer.md) — layers, tags, tools and themes in depth.
- [Fit a clip](../guides/fit-a-clip.md) — optimize a motion against point data and watch both layers together.
- [Optimize a trajectory](trajectories.md) — back to the problem that produced this clip.
