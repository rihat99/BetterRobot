# Panda test robot

- Source: `robots/panda_description/urdf/panda.urdf` and `robots/panda_description/meshes/collision/*.stl` of
  https://github.com/Gepetto/example-robot-data at commit `d0d9098` (2023-11-29).
- Licence: Apache-2.0, the description's own `robots/panda_description/LICENSE`, copied as `LICENSE`.
- Changed: the nine collision `filename` attributes name `meshes/collision/<link>.stl` relative to the URDF instead of
  `package://example-robot-data/robots/panda_description/...`. The visual COLLADA meshes (11 MB) are not tracked;
  the eleven visual `package://` URIs are unchanged and resolve to no file.
