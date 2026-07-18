# 03 — Open the core

> **Implementation log (2026-07-18):** T1–T6 are complete. The seven raw
> passes now expose frozen named results; dynamics owns optional `Data`; IK and
> contact outputs preserve requested graphs; retired stubs/aliases are gone;
> and a 16-test public VJP net pins the differentiability claim. Findings and
> deviations are recorded in `03_results.md`. **Completed:** full gate `1439
> passed, 2 skipped, 16 deselected`; Sphinx HTML/doctest green; source net
> **−177 lines** (budget: ≤ +150).

**Goal:** every block a robotics user could legitimately want is public,
uniformly named, and differentiable; the surface tells the truth. The audit
evidence (2026-07-18, confirmed by autograd probes): the core already
back-propagates to `q` and model values everywhere — FK, Jacobians, RNEA,
ABA, CRBA, centroidal, integrate/difference, residuals. What's wrong is
exposure, consistency, and the task facades.

**Size budget:** net ≤ +150 lines (exports and wiring, minus deleted stubs).

**Contract-test authorization:** `tests/contract/test_public_api.py`
(top-level additions), `test_submodule_public_imports.py` (raw-pass exports,
`ReferenceFrame` removal), `test_roadmap_stub_inventory.py` (stub deletions).
No other contract file changes.

---

## T1 — Raw passes: one convention, all public

Today one concept has three visibility tiers and two return styles:
`forward_kinematics_raw` (exported, returns a tuple), `frame_placements_raw`
/ `rnea_raw` / `aba_raw` / `crba_raw` / `ccrba_raw` (plain names, not in
`__all__`, dynamics ones return frozen `*Result` dataclasses),
`_compute_joint_jacobians_raw` (underscore-private).

Unify: every raw pass is named `<thing>_raw`, exported in its package
`__all__`, and returns a frozen `*Result` dataclass with named fields
(`forward_kinematics_raw` gains `FKResult(joint_pose_world,
joint_pose_local)`; rename `_compute_joint_jacobians_raw` →
`joint_jacobians_raw`). Update internal callers (dynamics passes, the Warp
bridge) mechanically. These are the documented tensor-in/tensor-out seam —
after this phase they are the LEGO blocks the docs advertise.

## T2 — The seam and the namespaces

- Add `ModelStructure` and `ModelValues` to the top-level `better_robot`
  exports (contract-test update).
- Make `br.spatial` attribute-reachable like every other subpackage (it is
  import-only today; `hasattr(br, "spatial")` is False).
- `ReferenceFrame` enum (`kinematics/__init__.py:25`) is defined but never
  used — the Jacobian functions take `Literal` strings. Delete the enum
  (ledger row); the strings are the API.

## T3 — `Data` optional for dynamics

`rnea` / `aba` / `crba` / `ccrba` / centroidal wrappers currently require a
caller-allocated `Data` purely to mutate it. New signatures:
`rnea(model, q, v, a, *, fext=None, data=None) -> Tensor` — allocate
internally when `data is None`, fill it when provided. Same for siblings.
Update tasks/examples/tests mechanically. (FK already returns a fresh `Data`;
Jacobians keep their Data-based flow — the FK→Jacobian sequencing with
`KinematicsLevel` is a real cache contract, not ceremony; `joint_jacobians_raw`
from T1 is the sequencing-free block.)

## T4 — Wire the differentiable solve

- `solve_ik` gains `differentiable: bool = False`. When True, it calls
  `LevenbergMarquardt.solve(..., differentiate="implicit")` instead of `run`
  — the target poses are already declared as differentiable parameters
  (`ik.py:294,350`); today the wiring simply stops at `run`. Result: `
  result.q` carries a graph back to the targets. Document the smoothness/
  convergence assumption plainly; raise the implicit path's honest error
  when ineligible.
- `solve_contact_forces` stops `.detach()`-ing `fext_local` and
  `generalized_force` in its result (`contact_forces.py:393-394`); the
  forward RNEA is differentiable and callers may want those gradients.
  Detached solver internals stay detached.
- Add tests: gradcheck-style probes through `solve_ik(...,
  differentiable=True)` w.r.t. a target pose on Panda; gradient flow from
  `solve_contact_forces` outputs.

## T5 — The surface tells the truth

- Delete the exported stubs that only raise `NotImplementedError`. Dynamics:
  `compute_minverse` (`crba.py:139-148`), `compute_coriolis_matrix`
  (`rnea.py:334`), `semi_implicit_euler`/`symplectic_euler`/`rk4`
  (`integrators.py:59,73,87` — delete the module if empty),
  `compute_centroidal_dynamics_derivatives` (`derivatives.py:106`).
  Residuals: the Yoshikawa manipulability stub
  (`residuals/manipulability.py:17`) and the stub collision residual exports
  (`residuals/collision.py:21`, exported via `residuals/__init__.py:33`) —
  the collision *package* decision stays owner-gated in `for_future.md`, but
  stub exports advertising nothing leave the surface now. Check
  `residuals/limits.py:112` (velocity/acceleration-limit analytic Jacobians
  that raise): if phase 2's protocol collapse already removed them, verify;
  otherwise the raising methods go and AD serves those residuals. Ledger
  rows for everything; `for_future.md` records the features. Update the
  roadmap docs page (its stub inventory shrinks accordingly) and its
  contract test.
- Delete the deprecated `nle` alias (`rnea.py:304`); `bias_forces` is the
  name.
- Remove the stray empty `src/better_robot/utils/` directory (holds only
  `__pycache__`).

## T6 — Gradcheck the guarantee

Add one test module (e.g. `tests/autograd/test_public_differentiability.py`)
that pins the LEGO guarantee: for each public entry point — FK (+frames),
frame/joint Jacobians, rnea/aba/crba, centroidal, integrate/difference, one
residual — assert gradients flow to `q` (and to a swapped `ModelValues`
tensor where meaningful) on a small Panda case, float64. Keep it fast
(seconds, not gradcheck-everything). This is the regression net that keeps
"differentiable by guarantee" true.

## Acceptance

- `from better_robot.dynamics import rnea_raw` etc. all work and appear in
  `__all__`; `br.ModelStructure`, `br.ModelValues`, `br.spatial` resolve.
- `rnea(model, q, v, a)` works without a `Data`.
- `solve_ik(..., differentiable=True)` returns a `q` with a `grad_fn`
  reaching the target; the new autograd test module is green.
- `grep -rn "NotImplementedError" src/better_robot/dynamics/ src/better_robot/residuals/`
  → none of the enumerated stub exports remains (collision internals may
  keep raises only if their exports were removed); `nle` gone; ledger
  updated.
- Full gate green; results file per standing rule 8.
