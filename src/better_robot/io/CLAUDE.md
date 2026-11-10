# io/ traps

The stages of `build_model` are described in `model_from_ir.py`, each parser in its own module docstring.

- Format cases stay in the parsers: kinematics, dynamics and model never learn a file format, and none of them
  may import `io` (`tests/contract/test_layer_dependencies.py`).
- A check of a robot description goes into `build_model`, the one boundary every description passes
  (`ModelBuilder`, URDF, MJCF, an `IRModel` written by hand), not into one parser.
- MJCF goes through MuJoCo's compiler (record 0036): never read MJCF XML yourself.
- A new format is a function that returns an `IRModel`, passed to `build_model`; there is no parser registry
  (`tests/contract/test_extension_points.py`, record 0025).
