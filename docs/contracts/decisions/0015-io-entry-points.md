# 0015 `io` has one `load`, staged `build_model` and one `ModelBuilder`

**Status:** Accepted (owner). **Date:** 2026-11-01

## Context
`io/build_model.py::build_model` has a 565-line body with 19 banners and nesting depth 7. `load` takes both
`free_flyer=` and `root_joint=` and lists `format="builder"`, which nothing handles; `parse` accepts a path, a callable
or a `yourdfpy.URDF`. Two builders exist: `ModelBuilder` and `io.builders.build_kinematic_tree_model`, which calls the
private `_push_joint`. `add_revolute_x(name, **kwargs)` hides its signature, `parent_body=` sits beside `parent=`, and
`_LEGACY_KIND_HELPER` keeps the kind aliases `"ball"`, `"free"`, `"floating"` and `"world"`.

## Decision
We will provide `load(path, *, format=None, root_joint=None, device=None, dtype=torch.float32)`,
`parse(path, *, format=None)`, a `build_model` made of named stages feeding one `Model(...)` call that clamps
`q_neutral` into the limits, and one `ModelBuilder` with `add_revolute(..., axis=)`, `add_prismatic(..., axis=)`, one
`parent=` and `add_shape(body, shape, *, collision=True)`, and remove `io.builders`, the legacy table and the aliases.

## Example
```python { .no-run }
model = br.load("g1.urdf", root_joint=br.JointFreeFlyer(), dtype=torch.float64)
builder = br.ModelBuilder("arm")
builder.add_revolute("elbow", parent="upper_arm", child="forearm", axis=(0, 1, 0), lower=-2.0, upper=2.0)
# build_model stages: _root, _order, _joints, _limits, _inertias, _frames, _mimic_reduction, _neutral -> Model(...)
```

## Consequences
Each stage is a function with a named result; the 11 `clamp` lines for `q_neutral` collapse into the final
constructor. Every signature is visible: no `**kwargs` family. A new format needs no builder: it returns an `IRModel`
(0025).

## Alternatives
Make `build_kinematic_tree_model` the documented path and `ModelBuilder` private: BetterHuman's three rigs are written
against `ModelBuilder` and would be rewritten. Pinocchio's `buildModels::urdf(path, rootJoint)` also takes the root
joint as one argument; Drake adds joints through typed `Add*Joint` calls; MuJoCo's `MjSpec` adds a joint through one
call with a type and an axis.

## Compatibility
Breaks `free_flyer=`, `add_revolute_x/y/z`, `parent_body=`, `io.builders.*` and the callable and `yourdfpy.URDF` forms
of `parse`. Migration: BetterHuman `families/mhr_rig.py` (a dict of `add_revolute_x/y/z`), one `add_frame(parent_body=)`,
its smoke test, and BetterRobot example 07. Reversible only by another break.

## Enforcement
`tests/contract/test_keyword_only.py` (C-API-2) and `tests/contract/test_public_surface.py`; new
`tests/io/test_build_stages.py` (every fixture robot has `q_neutral` inside its limits and each stage returns its
named result).

## Open questions
None.
