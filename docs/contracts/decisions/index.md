# Architecture decision records

An ADR records one decision that shapes the library: the context, the choice, its consequences, the
alternatives, and which contract rules enforce it. Accepted ADRs are never edited except their Status line;
a change is a new ADR that supersedes the old one. When to write one: see [00-index](../00-index.md).

Template: `# NNNN Title` / Status (Proposed | Accepted | Superseded by NNNN) + Date / Context (3–6 lines) /
Decision ("We will …") / Example / Consequences / Alternatives / Compatibility / Enforcement (rule IDs, ruff
codes, `tests/contract/*`) / Open questions (empty once Accepted). File name `NNNN-kebab-title.md`. Records 0009 and
later follow this template; 0001 to 0008 predate its Example, Compatibility and Open questions sections.

| ADR | Title | Status |
|-----|-------|--------|
| [0001](0001-torch-only.md) | Torch is the only implementation | Accepted |
| [0002](0002-immutable-fk-state.md) | A forward-kinematics query returns an immutable state | Accepted |
| [0003](0003-terms-plain-functions.md) | Terms are plain functions | Accepted |
| [0004](0004-viser-viewer.md) | One viewer, built on viser | Accepted |
| [0005](0005-nq-nv.md) | Configurations and velocities have different sizes | Superseded by [0013](0013-joint-kinds.md) |
| [0006](0006-serialization.md) | Serialization and format versioning of Model and IR | Superseded by [0032](0032-br-serialization-non-goal.md) |
| [0007](0007-derivatives-contract.md) | Derivatives contract: autograd first, analytic later | Accepted |
| [0008](0008-geometry-brick.md) | Geometry is a separate brick | Superseded by [0016](0016-geometry-out-of-model.md) |
| [0009](0009-model-flat-frozen-record.md) | `Model` is one flat frozen record | Accepted |
| [0010](0010-equality-and-hashing.md) | Identity equality for models and states, one by-value static key | Accepted |
| [0011](0011-caches-and-plans.md) | Plans live on the model; nothing is keyed on a tensor version | Accepted |
| [0012](0012-one-joint-order.md) | One joint order: stable source order | Accepted |
| [0013](0013-joint-kinds.md) | Joint kinds are final classes under an abstract `JointModel` | Accepted |
| [0014](0014-tensor-functions-for-lie-and-spatial.md) | Lie and spatial algebra are tensor functions only | Accepted |
| [0015](0015-io-entry-points.md) | `io` has one `load`, staged `build_model` and one `ModelBuilder` | Accepted |
| [0016](0016-geometry-out-of-model.md) | Geometry is a value, not a field of `Model` | Accepted |
| [0017](0017-optim-api.md) | `optim` takes strings for reads and an immutable `Problem` | Accepted |
| [0018](0018-frame-names.md) | A body's frame carries exactly the body's name | Accepted |
| [0019](0019-root-namespace.md) | The root names its subpackages and 14 first-level names | Accepted |
| [0020](0020-error-tree.md) | One error tree under `BetterRobotError` | Accepted |
| [0021](0021-validation-ladder.md) | One structural validator per entry, opt-in value checks, one internal-checks switch | Accepted |
| [0022](0022-viewer-boundary.md) | The viewer stays in-tree, with the public API the studio needs | Accepted |
| [0023](0023-naming-sweep.md) | Naming sweep: full words, plural axes, `_ids` for integer indices | Accepted |
| [0024](0024-surface-trim.md) | Remove the public names nobody uses | Accepted |
| [0025](0025-extension-model.md) | Extension without registries: five bricks, one way in each | Accepted |
| [0026](0026-coupling-standard-page.md) | One coupling-standard page in BetterRobot, tested by every consumer | Accepted |
| [0027](0027-shared-vocabulary.md) | One glossary for `KinematicState` and `HumanState` | Accepted |
| [0028](0028-batch-axes.md) | States broadcast `B...` against model values `I...`; `optim` keeps `(B, *event)` | Accepted |
| [0029](0029-frames-and-identifiers.md) | Names at every user boundary, ids inside | Accepted |
| [0030](0030-formats-in-the-standard.md) | The formats that cross repositories, listed in the standard | Accepted |
| [0031](0031-error-vocabulary-across-the-stack.md) | BetterHuman raises BetterRobot's errors and roots its own under them | Accepted |
| [0032](0032-br-serialization-non-goal.md) | BetterRobot writes no native model file | Accepted |
| [0033](0033-test-tiers.md) | Three test tiers, each one local command; a missing oracle fails | Accepted |
| [0034](0034-python-312.md) | Python 3.12 is the floor and the development version | Accepted |
| [0035](0035-shared-testing-module.md) | Shared test and docs code lives once, in `better_robot.testing` | Accepted |
| [0036](0036-mjcf-compile-through-mujoco.md) | MJCF loads through MuJoCo's compiler, or not at all | Accepted |
| [0037](0037-urdf-typed-adapter.md) | URDF parsing goes through one typed adapter | Accepted |
| [0038](0038-remove-downloader-and-registry.md) | No network downloader and no parser registry in `io` | Accepted |
| [0039](0039-test-robot-fixtures.md) | The test robots are tracked fixtures | Accepted |
| [0040](0040-model-value-memo.md) | A value-side memo on `Model`, keyed by grad mode | Accepted |
