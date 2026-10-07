# viewer — agent guide

One viewer on viser. Read `docs/concepts/viewer.md` first.

- `viewer.py` (`Viewer`), `timeline.py`, `layer.py`, `theme.py`, `_math.py` are the core; `layers/`
  are the built-in layers; `robot.py` is the only file that knows `Model`; `tools/` are gizmo and
  picking; `render/` is the offline rasterizer.
- Layers take numpy `(T, ...)` arrays (`as_frames` promotes static input to `T = 1`) and keep them by
  reference: in-place edits plus `viewer.show()` update the scene. Per frame write arrays into existing
  handles; never create or remove nodes during playback.
- `Viewer.show` is the only frame-push path (one `server.atomic()` block). Keep `set_frame` free of
  torch and FK; adapters precompute.
- viser is imported in `viewer.py` only; trimesh in `robot.py` only; no pyrender/imageio
  (`tests/contract/test_layer_dependencies.py`). The core never imports BetterHuman.
- Default host is `127.0.0.1`; tests use `tests/viewer/fake_server.py`, plus one loopback smoke test.
- A renderable layer implements `geometry(frame)` (world triangles); `render/rasterizer.py` is pure
  torch and must stay device-agnostic.
