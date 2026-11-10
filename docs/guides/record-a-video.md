# Record a video of a scene

`viewer.record(path)` turns the scene a `Viewer` shows into a `.webm` video, a `.gif` or PNG frames, without a
window: a headless Chromium opens the scene, steps through it and screenshots each step. This guide builds a
short clip of a Panda swinging one joint, records it, and points the live browsers at the same view.

## Build the scene

```python
from pathlib import Path

import torch
import better_robot as br
from robot_descriptions import panda_description
from better_robot import Viewer
from better_robot.viewer import RigidMeshLayer, robot_meshes

model = br.load(panda_description.URDF_PATH, dtype=torch.float64)
clip = model.q_neutral.repeat(60, 1)                                   # (T, nq)
clip[:, 0] = torch.linspace(-1.0, 1.0, 60, dtype=torch.float64)
geometry = br.io.load_geometry(panda_description.URDF_PATH, model)
poses = br.forward_kinematics(model, clip).joint_poses.numpy()         # (T, njoints, 7)
meshes = robot_meshes(geometry, root=Path(panda_description.REPOSITORY_PATH))
viewer = Viewer()
viewer.add([RigidMeshLayer("panda", meshes, body_poses=poses)])
print(viewer.timeline.nsteps)
```
```output
60
```

## Record it

`viewer.record(path)` opens the scene in a headless Chromium (`better_robot.viewer.capture`), shows each
step and screenshots it. The output format follows the suffix: `.webm` (needs `ffmpeg`), `.gif`, `name.png`
for numbered frames, or a directory of frames. Options are `steps`, `size` (width, height), `fps`, `gpu`
and `camera`.

```python { .no-run data-reason="records through a headless Chromium; tests/viewer/test_capture.py runs record" }
import tempfile
from pathlib import Path

out = Path(tempfile.mkdtemp()) / "panda.webm"
viewer.record(out, size=(960, 540), steps=range(0, 60, 2))
print(out.exists())
```

Without a `camera`, `record` fits a 3/4 view to the visible layers' `Layer.world_points` over the recorded
steps, upright along the theme's `up` axis. Pass `camera=(position, look_at)` for a fixed view. The browser
binary comes from `$BETTER_ROBOT_CHROME` or the `PATH`; the module docstring of `capture` explains the
one-time setup.

`viewer.aim(steps=...)` points the live browsers with the same 3/4 view: the connected ones and every new one
(viser's initial camera, which "Reset View" returns to). It fits the bounding box of what is visible at each step,
so a clip of robot meshes frames in milliseconds. It frames the current step by default; pass `steps`, such as
`range(0, viewer.timeline.nsteps, 10)`, to frame a whole clip, and `azimuth=`/`elevation=` to look from another
side.

## Where next

- [Viewer](../concepts/viewer.md): layers, groups and the one timeline that `record` steps through.
- [Visualize a robot](../getting-started/visualize.md): a first scene, shown live.
