# Information architecture of the new BetterRobot docs (MkDocs Material)

Diátaxis: tutorials (Getting started), how-to (Guides), explanation (Concepts), reference (API, glossary,
changelog, roadmap), plus Contributing (the old "conventions"). Every page title is a claim or a task, not a
label. Each concept page opens with one figure and a 3-line summary; deep material goes under
`??? note "Details"` admonitions. Code blocks are runnable snippets (the snippet test executes them).

```
docs/
  index.md                      landing: what it is, 20-line example, three entry buttons
  getting-started/
    installation.md             uv, extras (viewer), GPU note
    first-model.md              load URDF/MJCF, inspect structure, ModelBuilder
    forward-kinematics.md       forward_kinematics_state, frames, batching            [panda_rest.png]
    inverse-kinematics.md       Problem + terms.ik + solver, batched targets         [panda_ik.png]
    floating-base.md            free-flyer root, q layout, integrate/difference       [g1_floating.png]
    trajectories.md             time variables, window terms, banded solve            [trajectory_strip.png]
    gpu-and-batching.md         device/dtype, batch axes, timings
    visualize.md                Viewer, RobotView, record a clip
  concepts/
    architecture.md             package DAG, layering rule, what is public            [architecture.svg + archify map]
    model-and-state.md          ModelStructure / ModelValues / Model / KinematicState  [model_values_state.svg]
    joints-bodies-frames.md     tree, joint models, placements, q/v indices           [joints_bodies_frames.svg]
    lie-and-spatial.md          SO3/SE3 ops, pose/twist layout, frames of reference   [lie_spatial.svg]
    kinematics-and-jacobians.md FK, frame Jacobians, LOCAL_WORLD_ALIGNED, time variation [jacobians.svg]
    dynamics.md                 RNEA/CRBA/ABA on the state, centroidal, gravity, external forces
    optimization.md             variables, terms, Problem.loss vs linearize, solvers  [optim_problem.svg]
    time-and-structure.md       banded/arrow normal equations, windows, phases        [banded_normal.svg, banded_sparsity.png]
    collision.md                geometry, distances, collision terms
    viewer.md                   layers, groups, timeline, tools, offline render       [viewer_layers.svg]
    io.md                       parsers, IR, build_model, ModelBuilder
    design-decisions.md         why torch only, why state objects, why terms are functions
  guides/
    load-a-robot.md
    write-a-term.md             reads, jac, window, check_jacobian
    own-the-loop.md             torch optimizers on Problem.loss, solve_layer
    hard-constraints.md         solve_constrained
    fit-a-clip.md               phases, weight schedules
    differentiate.md            gradients through FK/dynamics/solves
    benchmark-and-profile.md
    use-with-betterhuman.md     one page pointing at BH
  reference/
    api/                        mkdocstrings pages, one per public module
    glossary.md
    changelog.md
    roadmap.md
  contributing/
    conventions.md              naming, style, shapes, validation at boundaries
    testing.md                  parity gates, contract tests, snippet tests
    extending.md                new joint model, new term, new layer, new parser
mkdocs.yml                      Material, light/dark palettes, navigation.tabs, search, mkdocstrings (numpy
                                or google docstring style — pick the one the code already uses), MathJax
```
