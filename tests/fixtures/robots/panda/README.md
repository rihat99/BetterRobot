# Panda test robot

- Source: the URDF and the collision meshes of commit `d0d9098` (2023-11-29) of example-robot-data,
  https://github.com/Gepetto/example-robot-data/blob/d0d9098/robots/panda_description/urdf/panda.urdf and
  https://github.com/Gepetto/example-robot-data/tree/d0d9098/robots/panda_description/meshes/collision.
- Licence: Apache-2.0, the description's own `robots/panda_description/LICENSE`, copied as `LICENSE`.
- Changed: the nine collision `filename` attributes name `meshes/collision/<link>.stl` relative to the URDF instead of
  `package://example-robot-data/robots/panda_description/...`. The visual COLLADA meshes (11 MB) are not tracked;
  the eleven visual `package://` URIs are unchanged and resolve to no file.
