# Parsers and the intermediate representation

A URDF file, an MJCF file, and a robot assembled in Python describe the same
kind of object in different languages. BetterRobot converts all three to one
small intermediate representation, then builds and validates a `Model` in one
place.

```text
URDF parser ─┐
MJCF parser ─┼─> IRModel ─> build_model ─> Model
Python builder┘
```

Without that middle form, every parser would need its own rules for topology,
joint indices, limits, inertias, mimic reduction, frames, and device tensors.
Those implementations would eventually disagree. The IR makes parsers
responsible for reading formats and makes `build_model` responsible for robot
semantics. See {ref}`decision-one-ir` for the trade-off.

## What the IR contains

`IRModel` contains lists of `IRBody`, `IRJoint`, and `IRFrame` records.
Geometry records preserve visual and collision metadata for later consumers.
The records use clear physical values: parent and child body names, joint
kind and axis, pose, limits, mass, center of mass, and inertia.

The IR is a build boundary, not a storage format. Its Python dataclasses may
change as loaders improve. Save the original robot description and parse it
again after an upgrade instead of pickling an `IRModel`.

## Parsing and building have different jobs

A parser must:

- interpret the source format's coordinate and naming rules;
- turn every source transform into BetterRobot's scalar-last pose layout;
- retain useful visual, collision, mimic, and limit metadata; and
- report malformed source data with the source element's name.

`build_model` then:

- validates that the body/joint graph is a tree;
- chooses concrete joint models;
- establishes stable topological and coordinate order;
- builds reduced-coordinate maps for mimic joints;
- packs model tensors on the requested device and dtype; and
- attaches frames, inertias, limits, and source metadata.

This division also lets tests feed synthetic IR directly to model building,
without first inventing a temporary XML file.

## The loading surface

`better_robot.load` is the ordinary entry point. It selects the URDF or MJCF
parser from a path suffix and then calls the shared builder. `free_flyer=True`
adds a free-flyer root; `root_joint=` provides a more explicit choice.

`ModelBuilder` offers the same destination for a model created in Python.
Direct `parse_urdf`, `parse_mjcf`, and `build_model` calls remain available
under `better_robot.io` when a caller needs parser-specific options.

The step-by-step recipes are in {doc}`/guides/load_a_robot`.

## Assets stay separate from topology

URDF and MJCF descriptions often refer to mesh files by relative, package, or
cached-download paths. Asset resolvers turn those references into local paths.
They do not change kinematic topology, and the model can still be used for
kinematics when visual assets are unavailable.

The viewer consumes visual geometry and its resolver. Collision metadata is
retained, but collision computation remains an incomplete capability described
in {doc}`/reference/collision_and_geometry`.

## Extension boundary

`register_parser(suffix, function)` can add suffix-based loading at runtime.
The function returns an `IRModel`; it does not bypass shared validation. This
keeps a new file format small and makes its resulting robot obey the same
contracts as built-in formats.

Read {doc}`model_and_data` for the built object, or {doc}`viewer` for the
consumer of visual geometry.
