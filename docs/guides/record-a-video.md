# Record a video of a scene

`viewer.record(path)` turns the scene a `Viewer` shows into an `.mp4` or `.webm` video, or one `.png` still,
without a window: a headless Chromium runs viser's own client, draws each step once at the output size and
encodes it in the page. This guide builds a short clip of a Panda swinging one joint, records it, and points the
live browsers at the same view.

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

```python { .no-run data-reason="records through a headless Chromium; tests/viewer/test_record.py runs record" }
import tempfile

out = viewer.record(Path(tempfile.mkdtemp()) / "panda.mp4")   # 1920 x 1080, every step, at the timeline's fps
print(out.name)
```

The suffix of the path picks the output:

| Suffix | Output | Needs |
|---|---|---|
| `.mp4` | H.264, encoded in the page; ffmpeg only writes the container | ffmpeg: `$BETTER_ROBOT_FFMPEG` or on `PATH` |
| `.webm` | VP9, or VP8 where the browser refuses VP9 | nothing |
| `.png` | one still of the current step, or of the one step in `steps` | nothing |
| none | `.mp4` when ffmpeg is found, else `.webm`; `record` returns the path it wrote | |

A browser without an H.264 encoder records VP9 and ffmpeg transcodes it, which takes longer than the rendering.
The options:

- `size=(width, height)`: the pixels of the file, both even for a video, up to 3840 x 2160 and beyond to the
  renderer's limit. The page draws at a pixel ratio of 1, so one canvas pixel is one pixel of the file.
- `steps`: frame `k` of the file shows `steps[k]`; every step by default.
- `fps`: the timeline's by default; `bitrate`: bits per second, 8 Mbit/s at 1920 x 1080 scaled by the pixel count.
- `camera=(position, look_at)`: each `(3,)` for a fixed view or `(len(steps), 3)` for one that moves with the steps;
  without it, the view `viewer.aim(steps=steps, fov=fov)` chooses (the bounding sphere of the visible layers'
  `Layer.bounds`, upright along the theme's `up` axis). `fov` is the vertical field of view in radians.
- `gpu`: the `nvidia-smi` index of the GPU to draw on, through Vulkan; `None`, the default, draws on the CPU
  (SwiftShader, which ships with Chromium). A full-HD frame of a Panda takes about 130 ms on the CPU and 5 to 8 ms
  on a GPU.

The frames are what a browser shows of the scene, background included: a dark theme records a dark clip. Playback
pauses while `record` runs, and the step shown before comes back after.

`viewer.aim(steps=...)` points the live browsers with the same 3/4 view: the connected ones and every new one
(viser's initial camera, which "Reset View" returns to). It reads only the layers' boxes, at most 32 evenly spaced
steps of the clip, so it takes milliseconds. It frames every step by default; pass `steps` to frame a part of the
clip, and `azimuth=`/`elevation=` to look from another side.

## Set up the browser

The browser is `$BETTER_ROBOT_CHROME`, else `chromium`, `google-chrome` or `chrome-headless-shell` on the `PATH`;
`npx playwright install chromium-headless-shell` downloads the last one. A server without `libasound.so.2` makes
Chromium exit at start: put a copy of the library (from the `libasound2` package) on `LD_LIBRARY_PATH`. Drawing on
a GPU needs `nvidia-smi` and Mesa's Vulkan device-select layer (`mesa-vulkan-drivers`), which puts the chosen GPU
first; `record` raises, naming `gpu=None`, when Chromium does not draw there.

## Where next

- [Viewer](../concepts/viewer.md): layers, tags and the one timeline that `record` steps through.
- [Visualize a robot](../getting-started/visualize.md): a first scene, shown live.
