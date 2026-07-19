# Visualize a robot

The optional viewer displays a robot in a browser. Install it alongside the
demo descriptions if you want to follow this guide with Panda:

```bash
python -m pip install '.[viewer,demos]'
```

Importing and configuring a {py:class}`better_robot.viewer.Visualizer` is lazy:
it does not start a server until you update or show the scene. That makes the
public setup itself safe in the documentation test environment, where `viser`
is intentionally absent.

```{testcode}
import better_robot as br
from better_robot.viewer import Visualizer
from robot_descriptions import panda_description

model = br.load(panda_description.URDF_PATH)
viewer = Visualizer(model, port=8080)

print(viewer.last_q.shape)
```

```{testoutput}
torch.Size([8])
```

With the viewer extra installed, the interactive part is:

```text
viewer.update(model.q_neutral)
viewer.show(block=False)

# Later, when the surrounding application exits:
viewer.close()
```

`show()` blocks by default. Pass `block=False` when a notebook, GUI, or service
already owns the event loop.

To play a {py:class}`better_robot.Trajectory`, call
`player = viewer.add_trajectory(trajectory)`, then
`player.show_frame(index)` or `player.play(fps=30.0)`. To display IK, call
`viewer.add_ik_result(result)`. Viewer symbols deliberately live under
`better_robot.viewer`; they are not top-level `better_robot` exports.

Models loaded from URDF can use their visual geometry. Programmatically built
models without mesh metadata fall back to a skeleton view.
