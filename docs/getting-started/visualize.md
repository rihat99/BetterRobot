# Visualize a robot

You open the browser viewer, show the Panda moving along a trajectory, step through it, and record it to a
GIF through a headless browser. This page needs the `viewer` and `meshes` extras (see [Install](installation.md)).

<figure markdown="span"><video autoplay loop muted playsinline preload="metadata" poster="../../clips/viewer_layers.png" width="960"><source src="../../clips/viewer_layers.webm" type="video/webm"></video>
<figcaption>One scene built up layer by layer: mesh, then skeleton, frames and ground, each switched on with <code>set_visible</code>; the layer list on the left toggles each one.</figcaption></figure>

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
from better_robot.viewer import FramesLayer, RigidMeshLayer, robot_meshes, robot_skeleton

model = br.load(panda_description.URDF_PATH, dtype=torch.float64)
q_rest = model.q_neutral
clip = q_rest.repeat(30, 1)                       # (T, nq)
clip[:, 0] = torch.linspace(-1.0, 1.0, 30, dtype=torch.float64)
```

Any `(T, nq)` tensor works: the `result.x["q"][0]` of [the trajectory tutorial](trajectories.md) is one.

## Open the viewer and add the robot

`Viewer()` starts a local [viser](https://viser.studio) server on `127.0.0.1:8080`; open `viewer.url` in a
browser. Forward kinematics runs once for the clip; `robot_meshes` turns the description's `Geometry` into the
meshes of a `RigidMeshLayer` that follows the joint poses, and a `FramesLayer` draws frame axes:

```python
viewer = Viewer()
geometry = br.io.load_geometry(panda_description.URDF_PATH, model)
state = br.forward_kinematics(model, clip)
poses = state.joint_poses.numpy()                                        # (T, njoints, 7)
meshes = robot_meshes(geometry, root=Path(panda_description.REPOSITORY_PATH))
hand = FramesLayer("hand", state.frame_poses[:, [model.frame_id("panda_hand")]].numpy())
viewer.add([RigidMeshLayer("panda", meshes, body_poses=poses), hand])   # an iterable of layers
print(viewer.url, len(viewer.layers))
```
```output
http://127.0.0.1:8080 2
```

Forward kinematics runs once for the whole clip, not once per step.

## Step through the timeline

`viewer.show(step)` moves the timeline and pushes that step to every visible layer:

```python
viewer.show(0)
viewer.show(29)
```

`viewer.play()` runs the timeline at the viewer's `fps` and blocks until you stop it, so call it last in a
script:

```python { .no-run data-reason="blocks until Ctrl-C" }
viewer.play()
```

## Update the robot in place

A layer keeps its arrays by reference. To redraw the robot for a new clip of the same length, for example after
each solver step, write the new poses into the layer's arrays and call `viewer.show()`:

```python
skeleton = robot_skeleton(model, poses, name="panda_live")
viewer.add([skeleton], group="live")
backwards = br.forward_kinematics(model, clip.flip(0)).joint_poses.numpy()
skeleton.joints[:] = backwards[:, 1:, :3]
skeleton.rotations[:] = backwards[:, 1:, 3:]
viewer.show()
```

## Record a GIF without a browser

`viewer.record` screenshots the live scene in a headless Chromium and writes `.webm`, `.gif` or
`.png` frames. Without a `camera` it frames the robot automatically:

```python { .no-run data-reason="records through a headless Chromium; tests/viewer/test_capture.py runs record" }
viewer.remove_group("live")
out = Path(tempfile.mkdtemp()) / "panda_swing.gif"
viewer.record(out, size=(320, 240), steps=range(0, 30, 3))
print(out.exists(), out.stat().st_size > 0)
```

It needs Chromium (`$BETTER_ROBOT_CHROME` or on `PATH`) and, for `.webm`, ffmpeg; see `better_robot.viewer.capture`.

## Where next

- [Viewer](../concepts/viewer.md) — layers, groups, tools and themes in depth.
- [Fit a clip](../guides/fit-a-clip.md) — optimize a motion against point data and watch both layers together.
- [Optimize a trajectory](trajectories.md) — back to the problem that produced this clip.
