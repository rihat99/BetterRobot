# 01 Naming

Two naming regimes coexist on purpose. *Math code* (the packages `lie`, `spatial`, `kinematics`, `dynamics`) uses
Drake's *monogram notation* for local variables so that a formula can be checked frame by frame. Everything a
user sees (public functions, parameters, dataclass fields, module names) uses full descriptive words.

## C-NAME-1 Monogram notation inside math code (MUST)

A monogram name encodes *what* quantity, *between which frames*, *expressed in which frame*:
`X_AB` is the pose of frame B in frame A; `p_AoBo_W` is the position from A's origin to B's origin expressed in W;
`V_WB` is the spatial velocity of B measured in W; `R_AB` a rotation. A trailing `o` marks a frame's origin
(`Ao` = origin of A). A `J_` prefix names the Jacobian of what follows: `J_V_WB` is the Jacobian of the spatial
velocity of B in W with respect to the configuration.
Composition reads left to right: `X_WB = X_WA @ X_AB`. Use these names for locals and private helpers in
`lie`, `spatial`, `kinematics`, `dynamics` and nowhere else.

```python no-run
# good  (illustrative; real entry point: kinematics/state.py:151)
X_WB = X_WA @ X_AB
V_WB = adjoint(X_WB) @ V_B
# bad
pose = parent_pose @ joint_pose     # which frames? cannot be checked by eye
```

Enforcement: `[ruff:N806,N803]` [planned: L-7] are *ignored* for these four packages (per-file ignores in `pyproject.toml`);
`[review]` checks that the frames in the name match the frames in the formula.

## C-NAME-2 Descriptive names in the public API (MUST)

Public functions, parameters and fields are full words in `snake_case`; classes `CapWords`; constants
`UPPER_CASE`. No abbreviations beyond the glossary set (`q`, `v`, `nq`, `nv`, `fk`, `ik`, `lm`). A user never has to
know monogram notation to call the library.

```python no-run
# good
def forward_kinematics_state(model: Model, q: Tensor) -> KinematicState: ...
# bad
def fk_st(m, q): ...
```

Enforcement: `[ruff:N801,N802,N803,N806,N815,N816]` [planned: L-7] outside the math packages; `tests/contract/test_naming.py`.

## C-NAME-3 Order words follow the data order (MUST)

A name that mentions a layout must match the layout in `02-tensors`: quaternions are `xyzw`, twists are
`linear_angular`. Never name something `wxyz` or `angular_linear` unless it converts to that layout, and then the
name says `to_` / `from_`.

```python no-run
# good
quat_xyzw = pose[..., 3:7]
def twist_to_angular_first(V: Tensor) -> Tensor: ...
# bad
quat = pose[..., 3:7]          # order hidden; a reader must open the docs
```

Enforcement: `[review]`.

## C-NAME-4 Units and frames in parameter names when ambiguous (SHOULD)

Angles are radians, lengths metres, time seconds; a parameter in another unit carries the unit (`timeout_ms`).
A frame argument is `frame`, a joint `joint`, a body `body`; the viewer's time axis is `step` (`nsteps`,
`set_step`). The word *frame* always means a robot frame, never an animation frame.

```python no-run
# good
viewer.set_step(10)
# bad
viewer.set_frame(10)      # collides with the robot-frame meaning
```

Enforcement: `[review]`.
