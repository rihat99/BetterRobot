# 0019 The root names its subpackages and 14 first-level names

**Status:** Accepted (owner). **Date:** 2026-11-01

## Context
The root `__all__` has 22 names and lists no subpackage. Contract C-API-5 promises inverse kinematics and the viewer at
the root, and neither is there. Apps write `import better_robot as br` 18 times and `br.forward_kinematics_state` 12
times, and reach `kinematics`, `optim` and `viewer` by paths that the root does not list, which principle 1.6 counts
as private.

## Decision
We will list in the root `__all__` the subpackages `exceptions`, `lie`, `spatial`, `model`, `kinematics`, `dynamics`,
`collision`, `io`, `optim`, `terms` and `viewer`, and the names `load`, `Model`, `ModelBuilder`, `JointFreeFlyer`,
`forward_kinematics`, `KinematicState`, `frame_jacobian`, `joint_jacobian`, `inverse_dynamics`, `mass_matrix`,
`forward_dynamics`, `center_of_mass`, `solve_ik` and `Viewer`.

## Example
```python { .no-run }
import better_robot as br
model = br.load("panda.urdf")                                   # level 1: the root names
state = br.forward_kinematics(model, q)
result = br.solve_ik(model, {"panda_hand": target}, q0=q0)
problem = br.terms.ik_problem(model, {"panda_hand": target}, q0=q0)   # level 2; level 3: br.optim.run(..., Phase)
```

## Consequences
A first script needs only the root. Time-variation and centroidal names stay package-qualified. The shipped stand-in
test helpers (0035) add `testing` as a twelfth subpackage in the same list. Cost: `Viewer` at the root must stay a lazy
attribute, so that importing the root does not import viser.

## Alternatives
The same without `Viewer` at the root (viser is lazy either way): the viewer then needs its own import line in the
first script. Pinocchio exposes its whole API flat at `pinocchio.*`; pyroki's root exports `Robot`, the costs, the
collision module and the viewer, which is the shape chosen here.

## Compatibility
Adds names and removes `Frame`, `Joint`, `Body`, `Trajectory`, `SE3`, `ModelStructure`, `ModelValues` and the
time-variation and centroidal names from the root (0024). `forward_kinematics_state` is renamed (0023).

## Enforcement
`tests/contract/test_public_surface.py` (the root `__all__` is exactly this list; C-API-4: every dotted path's segments
are listed in their parent's `__all__`) and `tests/contract/test_submodule_public_imports.py` (C-API-5); the consumer
import tests in BetterHuman, BetterVideoReconstruction and BetterRetarget.

## Open questions
None.
