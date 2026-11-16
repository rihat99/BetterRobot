# viewer/ traps

The design is `docs/concepts/viewer.md`; read it first.

- Public callables take at most two positional parameters (a layer: its name and main array); the rest are
  keyword-only.
- Layers take numpy `(T, ...)` arrays (`as_frames` promotes static input to `T = 1`) and keep them by
  reference: in-place edits plus `viewer.show()` update the scene. Per step write arrays into existing
  handles; never create or remove nodes during playback.
- `Viewer.show` is the only step-push path (one `server.atomic()` block). Keep `set_step` free of
  torch and FK; adapters precompute.
- Time: one integer `step` timeline; a layer with `times` (seconds) is mapped by `Viewer._index`, and
  `set_step` always receives a sample index. A `(T,)` `valid` (per-sample visibility) hides the layer's frame
  `/<tag>/<name>` in `Viewer._push`, without touching the layer's own switch; layers that hide single elements read
  a `(T, N)` one in `set_step`. Tools register scene-pointer callbacks only through `Viewer.on_pointer` /
  `clear_pointer` (one callback at a time, a second registration raises; on viser's `scene.on_click` /
  `on_rect_select`).
- Quaternion work calls `better_robot.lie` on `torch.from_numpy` views: no numpy copy of `lie`. viser takes
  scalar-first `wxyz`: `poses[step][:, [6, 3, 4, 5]]`, never `poses[step, :, [6, 3, 4, 5]]` (numpy moves the list
  axis first when a scalar index sits beside it).
- The viewer's outside imports are torch, numpy, viser, pillow and trimesh, nothing else (no BetterHuman, no
  pyrender or imageio: `tests/contract/test_optional_imports.py`); viser is imported in `viewer.py` and trimesh in
  `robot.py` and the mesh layers, each inside the function that needs it.
- Default host is `127.0.0.1`; tests use the stand-in `better_robot.testing.FakeServer`, plus one loopback smoke test.
- A layer that should frame the camera of `Viewer.record` and `Viewer.aim` implements `bounds(step)` (a world box
  `(lo (3,), hi (3,))`, cheap: no per-vertex work); the default `None` keeps it out of the fit.
- Sidebar callbacks run under `viewer.lock`, and the playback thread takes that lock to push a step: a callback must
  never wait for the thread (`Timeline.pause` joins it); the Timeline's own controls stop it with `_halt`.
- `Theme.apply` switches viser's environment map off: any map washes solid colours to pastel. Meshes share the rig.
- viser draws a plain mesh's `color` and a point cloud's colours as sRGB, but the colours of batched meshes, arrows,
  line segments and a glTF's vertices as linear, which looks pale: send those through `theme.linear_uint8`, the
  others through `theme.rgb_uint8` (`tests/viewer/test_record.py` reads the pixels). Lines are unlit, yet the page's
  filmic tone mapping still bends them by up to 12 levels. A GLB without a material is a black metal under this rig.
