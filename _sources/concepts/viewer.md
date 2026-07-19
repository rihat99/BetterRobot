# Viewer

Visualization is useful, but it must not become a dependency of robot math.
The viewer is therefore the top layer of BetterRobot. It reads `Model`,
`Data`, and `Trajectory`; no kinematics, dynamics, residual, or optimizer
module imports it.

Interactive rendering uses [viser](https://viser.studio/main/), installed
through the `viewer` extra. Heavy rendering packages are imported only when a
feature needs them, so importing the robotics core does not require a browser
server or mesh library.

## Scene, mode, and backend

The viewer separates three jobs:

- `Scene` owns one robot's visible layers and updates them from new data.
- A `RenderMode` decides how the robot looks. `SkeletonMode` draws joints and
  links; `URDFMeshMode` uses visual geometry kept by the parser.
- A `RendererBackend` creates and updates primitives. `ViserBackend` is the
  public interactive implementation.

That separation keeps scene logic testable without opening a browser and lets
render modes share overlays and playback. The in-memory backend used by tests
is an implementation aid, not a public viewer import.

## Modes and overlays

`SkeletonMode` works for any model because it needs only joint placements.
`URDFMeshMode` also needs parser IR and resolvable visual assets. Missing mesh
geometry should not prevent the skeleton or the robot calculations from
working.

Overlays live under `better_robot.viewer.overlays`:

| Overlay | What it shows |
|---|---|
| `GridOverlay` | a ground grid and world axes |
| `FrameAxesOverlay` | named robot frames |
| `TargetsOverlay` | target poses and interactive transform controls |
| `ForceVectorsOverlay` | caller-supplied force arrows |

An overlay consumes already-computed tensors. It never triggers hidden
kinematics or changes a solver problem.

## Interaction and playback

`Visualizer` is the convenient viser-backed facade. It can update a robot
configuration, expose draggable IK targets, and show or hide layers.
`TrajectoryPlayer` selects integer frames from a `Trajectory` and updates the
scene. Both interactive display and playback may block; the how-to guide shows
where to choose nonblocking display.

Targets are callbacks into user code. Dragging a marker does not silently run
IK. This keeps expensive work, convergence handling, and event-loop ownership
visible to the application.

## Public surface

The main qualified imports are `Visualizer`, `Scene`, `SkeletonMode`,
`URDFMeshMode`, `RenderMode`, `RenderContext`, `RendererBackend`,
`ViserBackend`, `TrajectoryPlayer`, and `PrimitiveHandle`. They live under
`better_robot.viewer`, not at the package root.

Primitive handles can update color and scale without exposing backend objects.
Render modes can be added explicitly to a scene; the extension convention is
described in {doc}`/conventions/extension`.

## Dependency boundary

The viewer may depend on tasks and every lower robot layer. Nothing may depend
on the viewer. Contract tests enforce that direction and keep optional
rendering imports isolated.

For a complete setup and update loop, use {doc}`/guides/visualize`. For how
visual geometry reaches a model, read {doc}`parsers_and_ir`.
