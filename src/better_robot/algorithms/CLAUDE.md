# algorithms/ — Robot Algorithms

Free functions operating on `RobotModel`. Depends on `math/` and `models/`.

## Subpackages

### `kinematics/` — Forward kinematics, Jacobians, kinematic chains

See `kinematics/CLAUDE.md`.

### `geometry/` — Collision geometry primitives

See `geometry/CLAUDE.md`.

## Design Principle

All algorithms are **free functions** with `RobotModel` as the first argument:

```python
fk = forward_kinematics(model, cfg)                            # not model.fk(cfg)
J  = compute_jacobian(model, cfg, link_idx, target, pw, ow)   # not model.jacobian(...)
```

`RobotModel` also exposes convenience methods (`model.forward_kinematics(cfg)`) that delegate to these free functions for ergonomics. The free functions are the canonical implementations.
