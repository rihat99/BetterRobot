# 14 · Dynamics milestone plan — D0 to D7, sequenced concretely

★★ **Structural.** Builds on
[06_DYNAMICS.md §5](../../design/06_DYNAMICS.md) and resolves it
against the proposals in this folder. Closes the open dynamics items
in [18_ROADMAP.md §1](../../status/18_ROADMAP.md).

## Problem

[06 §5](../../design/06_DYNAMICS.md) lists eight dynamics milestones:

> 1. **D0 — skeleton.** All functions present, all raise
>    NotImplementedError, all have docstrings and signatures.
> 2. **D1 — center_of_mass.**
> 3. **D2 — rnea.**
> 4. **D3 — crba.**
> 5. **D4 — aba.**
> 6. **D5 — centroidal.**
> 7. **D6 — derivatives.**
> 8. **D7 — three-layer action model.**

D0 has landed. D1 is partially landed (`center_of_mass` works, the
centroidal map is a stub). D2–D7 are stubs. The order is correct;
what's missing is the binding to:

- the typed value types ([Proposal 05](05_value_types_audit.md)),
- the backend dispatch ([Proposal 02](02_backend_abstraction.md)),
- the cache invariants ([Proposal 07](07_data_cache_invariants.md)),
- the trajectory type ([Proposal 08](08_trajectory_lock_in.md)),
- the regression oracle ([Proposal 12](12_regression_and_benchmarks.md)).

A milestone that lands without these bindings creates work later. We
should sequence them in.

## The proposal

A revised D2–D7 sequence with explicit acceptance criteria and
cross-references.

### D1 (in progress) — Center of mass

**Already implemented** (`center_of_mass`). What remains:

- Wire `compute_centroidal_map` (currently stub) and
  `compute_centroidal_momentum` so [03 §7 Inertia.apply](../../design/03_LIE_AND_SPATIAL.md)
  composes with the centroidal pass.
- Land a regression entry for COM at neutral and at random q in the
  `fk_reference.npz` extension ([Proposal 12](12_regression_and_benchmarks.md)).

**Cross-references in this folder**: [Proposal 05](05_value_types_audit.md)
(use `Inertia.se3_action`, `Inertia.add` instead of raw 10-vector
arithmetic).

D1 close-out runs **in parallel** with D2 RNEA — the two share no
implementation surface beyond the spatial value types.

### D2 — RNEA (inverse dynamics)

Two-pass Featherstone:

1. **Forward pass** (root → leaves):
   - For each joint `j` in topo order:
     - `S_j = jm.joint_motion_subspace(qj)` — `(B..., 6, nv_j)`
     - `v_j = parent.v_j  +  S_j  @  vj_slice`
     - `a_j = parent.a_j  +  S_j  @  aj_slice + cross(parent.v_j, S_j) @ vj_slice`
2. **Backward pass** (leaves → root):
   - `f_j = I_j  @  a_j  +  cross(v_j, I_j  @  v_j)`
   - `tau_j = S_j.T  @  f_j`
   - `parent.f += Ad(T_{p,j}).T @ f_j`

The implementation lives in `dynamics/rnea.py`. It uses:

- `spatial.Motion` / `spatial.Force` value types
  ([Proposal 05](05_value_types_audit.md)) for readability.
- `spatial.ops.cross_mm` / `cross_mf` (already present).
- `lie.se3.adjoint` / `adjoint_inv` — routed through
  `current().lie` ([Proposal 02](02_backend_abstraction.md)).

**Acceptance**:

- `tau = rnea(model, data, q, v, a)` matches the closed-form torque
  for a 1-DOF revolute pendulum to fp64 ulp precision.
- For Panda at random `q, v, a`, output matches a Pinocchio reference
  (loaded via the optional [Proposal 12 §12.B](12_regression_and_benchmarks.md))
  to `1e-8` (fp64).
- `bias_forces(model, data, q, v) == rnea(model, data, q, v, zeros)`.
- `Data._kinematics_level` is unaffected (RNEA does not advance the
  kinematics cache; it computes its own internal `joint_velocity_world`,
  `joint_acceleration_world`).

### D3 — CRBA (joint-space inertia)

Backward pass over composite-rigid-body inertias:

- For each joint `j` in reverse topo order:
  - `Ic_j = I_j + sum(child Ic transformed up)`
- Build `M(q)` column-by-column: `M[:, idx_v_j : idx_v_j + nv_j] = (Ic_j @ S_j).T  @ S_j`.

**Acceptance**:

- `M(q_neutral)` is symmetric positive definite for Panda and G1.
- `M(q) @ a + bias_forces(q,v) == rnea(q, v, a)` (algebraic identity)
  to fp64 ulp.
- `data.mass_matrix` is filled and `Data._kinematics_level` stays
  unchanged.

### D4 — ABA (forward dynamics)

Articulated Body Algorithm. The complex one. Implementation per
Featherstone ch. 7 — three passes (forward velocity, backward bias
forces and articulated inertia, forward acceleration).

**Acceptance**:

- `aba(q, v, rnea(q, v, a))` returns `a` to fp64 ulp.
- `aba(q, v, tau) == M(q)^{-1}  @  (tau - bias_forces(q, v))` (
  algebraic identity) for any small Panda example.
- Pinocchio cross-check on G1.

### D5 — Centroidal

`compute_centroidal_map(model, data, q)` populates
`data.centroidal_momentum_matrix` (the `Ag` matrix in Pinocchio
notation).

**Acceptance**:

- `Ag(q) @ v == centroidal_momentum_at_com(model, data, q, v)`,
  computed independently from FK + `Inertia.apply`.
- `compute_centroidal_momentum(model, data, q, v)` matches the same
  result.
- Test against published values for a known humanoid pose
  (e.g. half-sitting on G1).

### D6 — Derivatives

The Carpentier-Mansard analytic derivative formulas:

- `compute_rnea_derivatives(model, data, q, v, a) -> (∂τ/∂q, ∂τ/∂v, ∂τ/∂a=M)`
- `compute_aba_derivatives(...)`.

These go into the **backward** pass of
`torch.autograd.Function.apply` wrappers around `rnea` and `aba`. So
each algorithm "owns" its forward and its backward kernel — no
autodiff through the topological recursion. This is what
[14 §2.3](../../conventions/14_PERFORMANCE.md) means by:

> RNEA / ABA Jacobians (phase D3): **analytic via Carpentier &
> Mansard**. Not autodiff.

**Acceptance**:

- Wrap `rnea` in `RneaFunction(torch.autograd.Function)` whose
  `forward` calls the imperative kernel and `backward` calls
  `compute_rnea_derivatives`.
- `torch.autograd.gradcheck(rnea, ...)` passes at fp64 with
  `atol=1e-8`.
- `torch.func.jacrev(lambda q: rnea(model, data, q, v, a))(q)` matches
  `compute_rnea_derivatives(...)` output to fp64 ulp.

### D7 — Three-layer action model (Crocoddyl pattern)

`dynamics/action/{differential,integrated,action}.py` per
[06 §6](../../design/06_DYNAMICS.md). With D2/D3/D4/D6 in place this is
mechanical:

- `DifferentialActionModelFreeFwd.calc(data, x, u)` calls `aba(...)`.
- `DifferentialActionModelFreeFwd.calc_diff(data, x, u)` calls
  `compute_aba_derivatives(...)`.
- `IntegratedActionModelEuler.calc(...)` calls `state.integrate(x, dt * f)`.
- `IntegratedActionModelRK4.calc(...)` is the standard 4-stage tableau
  in the state manifold.

**Acceptance**:

- A toy pendulum problem solves under DDP (placeholder optimiser if
  DDP is not yet implemented; or LM in the meantime) and reaches a
  hand-computed reference solution.
- The `Trajectory` type ([Proposal 08](08_trajectory_lock_in.md)) is
  the return type of trajectory rollouts.

## Sequencing — what blocks what

```
        D2 ──→ D3
       ↗   ↘
 (D1) ↗     ↘
       ↘     D4 ──→ D6 ──→ D7
        ↘          ↗
         D5 ──────
```

- **D1 does *not* block D2.** Earlier drafts had D1 close-out as a
  precondition for D2 RNEA; the gpt-plan review pointed out
  correctly that RNEA is independent of the centroidal map. RNEA
  needs stable Lie/spatial conventions and frozen value-types
  ([Proposals 01, 05](01_lie_typed_value_classes.md)) — *not* a
  finished centroidal pass.
- D1 (centroidal map close-out) sits in parallel with D2.
- D2 ↦ D3 is direct.
- D2 ↦ D4 is direct.
- D5 sits in parallel with D2/D3/D4 — pure kinematics.
- D6 needs D2 and D4.
- D7 needs D6 (and D4).

A reasonable calendar (D1 and D2 run in parallel):

| Quarter | Milestones |
|---------|------------|
| Q+1 | D2 RNEA + tests + Pinocchio cross-check.  D1 close-out (centroidal map) lands in parallel. |
| Q+2 | D3 CRBA; D5 centroidal map; benchmarks for FK / RNEA / CRBA. |
| Q+3 | D4 ABA; integrators (Euler, RK4, symplectic). |
| Q+4 | D6 derivatives + autograd.Function wrappers; gradcheck. |
| Q+5 | D7 three-layer action model; toy DDP. |

This is roughly 5 quarters of dynamics work. The dynamics layer is
the largest remaining chunk; it lands in calibrated steps.

## JointModel hooks needed before D2 RNEA lands

The current RNEA assumes a per-joint **bias acceleration** of zero
(`c_J = 0`). That assumption holds for revolute, prismatic, and
free-flyer joints — but not for spherical, anatomical, or coupled
joints. Bake the hook into the `JointModel` Protocol now, while
only the simple kinds exist, so dynamics does not have to refactor
the surface later. From the gpt-plan review:

```python
class JointModel(Protocol):
    # … existing kinematic surface (joint_transform, motion_subspace,
    # integrate, difference, neutral, random_configuration) …

    def joint_bias_acceleration(
        self,
        q_slice: torch.Tensor,
        v_slice: torch.Tensor,
    ) -> torch.Tensor:                              # (B..., 6)
        """``c_J`` — the joint's spatial bias acceleration in its
        own frame. For revolute / prismatic / free-flyer this is
        identically zero; for coupled or anatomical joints it is the
        derivative of the motion subspace ``S_J(q)`` along ``v``.

        Default implementation: returns zeros. Custom joints override.
        """

    def joint_motion_subspace_derivative(
        self,
        q_slice: torch.Tensor,
        v_slice: torch.Tensor,
    ) -> torch.Tensor:                              # (B..., 6, nv_j)
        """``Ṡ_J(q, v)`` — derivative of the motion subspace.
        Used by RNEA's forward pass and by analytic dynamics
        derivatives. Default: zeros (correct for constant
        ``S_J``)."""
```

Both methods land with **default implementations of zero** in the
abstract base. The simple joint kinds inherit the defaults; the
coupled/anatomical joints in
[Proposal 09](09_human_body_extension_lane.md) override them. This
keeps the seam small while making it explicit, which is exactly the
gpt-plan-driven correction.

The acceptance criterion for D2 RNEA reads as:

> RNEA computes `tau` via `S_J(q)` and `c_J(q, v) = 0` for the
> built-in joint kinds; for any joint kind whose
> `joint_bias_acceleration(...)` returns non-zero, the RNEA forward
> pass uses it. Tested against a synthetic coupled-joint fixture
> and against Pinocchio for the standard kinds.

## Cross-references

- [Proposal 02](02_backend_abstraction.md) — RNEA/ABA route through
  `current().dynamics`.
- [Proposal 03](03_replace_pypose.md) — pure-PyTorch backend
  removes the autograd cliff D6 has to work around.
- [Proposal 05](05_value_types_audit.md) — `Motion`, `Force`,
  `Inertia` are the readable language for D2's kernel.
- [Proposal 07](07_data_cache_invariants.md) — RNEA's intermediate
  buffers (`joint_forces`, internal velocity stack) are part of the
  cache-invalidation set.
- [Proposal 08](08_trajectory_lock_in.md) — D7 returns
  `Trajectory`.
- [Proposal 12](12_regression_and_benchmarks.md) — every milestone
  ships with a regression entry in `fk_reference.npz` (extended) or
  a sibling `dynamics_reference.npz`.
- [`better_human/smpl/dynamics.py`](../../../../better_human/) — the
  "poorly written but useful as inspiration" reference per
  [06 §9](../../design/06_DYNAMICS.md). Rewrite from scratch using
  `spatial/`; do not port file-for-file.

## Acceptance for the dynamics layer as a whole

When D7 closes:

- `rnea`, `aba`, `crba`, `bias_forces`, `compute_centroidal_map`,
  `compute_rnea_derivatives`, `compute_aba_derivatives`,
  `compute_crba_derivatives`,
  `compute_centroidal_dynamics_derivatives` are all implemented.
- All pass Pinocchio cross-checks at fp64.
- `gradcheck` passes through `aba` and `rnea` end-to-end.
- A toy DDP/iLQR loop on a 1-DOF pendulum closes; reference test
  pinned.
- `dynamics_reference.npz` is committed; CI guards numerical drift.
