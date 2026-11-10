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
  `/<group>/<name>` in `Viewer._push`, without touching the layer's own switch; layers that hide single elements read
  a `(T, N)` one in `set_step`. Tools register scene-pointer callbacks only through `Viewer.on_pointer` /
  `clear_pointer` (one callback at a time, a second registration raises; on viser's `scene.on_click` /
  `on_rect_select`).
- `_lie.py` runs every quaternion operation through `better_robot.lie` on numpy arrays: no numpy copy of `lie`.
- The viewer's outside imports are torch, numpy, viser, pillow and trimesh, nothing else (no BetterHuman, no
  pyrender or imageio: `tests/contract/test_optional_imports.py`); viser is imported in `viewer.py` and trimesh in
  `robot.py`, each inside the function that needs it.
- Default host is `127.0.0.1`; tests use the stand-in `better_robot.testing.FakeServer`, plus one loopback smoke test.
- A layer that should frame the camera of `Viewer.record` and `Viewer.aim` implements `world_points(step)` (world
  positions `(N, 3)`); the default `None` keeps it out of the fit.
