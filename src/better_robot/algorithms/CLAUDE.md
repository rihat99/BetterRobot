# algorithms/ — Robot Algorithms

Free functions operating on `RobotModel`. Depends on `math/` and `models/`.

## Subpackages

### `kinematics/` — Forward kinematics, Jacobians, kinematic chains

See `kinematics/CLAUDE.md`.

`forward.py` and `chain.py` contain the canonical, full implementations of FK and chain traversal. `RobotModel.forward_kinematics()` and `RobotModel.get_chain()` are thin delegates.

### `geometry/` — Collision geometry, distance functions, RobotCollision

See `geometry/CLAUDE.md`.

Fully implemented: primitive types (Sphere, Capsule, Box, HalfSpace, Heightmap), pairwise distance functions, `compute_distance` dispatcher, `colldist_from_sdf` smoothing, and `RobotCollision` sphere-decomposition model.

## Design Principle

All algorithms are **free functions** with `RobotModel` as the first argument:

```python
fk = forward_kinematics(model, cfg)                            # not model.fk(cfg)
J  = compute_jacobian(model, cfg, link_idx, target, pw, ow)   # not model.jacobian(...)
```

`RobotModel` also exposes convenience methods (`model.forward_kinematics(cfg)`) that delegate to these free functions for ergonomics. The free functions are the canonical implementations.
