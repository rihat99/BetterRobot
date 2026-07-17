# M3 mimic reduced-coordinate design

Public coordinates omit mimic targets; concrete joint recursions retain a full
layout. With `Gq = q_expansion`, `g0 = q_offset`, and
`Gv = v_expansion`:

- `q_full = q @ Gq.T + g0`, `v_full = v @ Gv.T`;
- `J = J_full @ Gv`, `tau = tau_full @ Gv`;
- `M = Gv.T @ M_full @ Gv`, `A_g = A_g_full @ Gv`.

ABA is the intentional exception: full ABA followed by reduction is not a
constrained solve. Mimic models use projected CRBA/RNEA and solve
`M_reduced ddq = tau - bias`; non-mimic models retain the articulated-body
fast path. The prototype Warp FK lane explicitly falls back to torch.

Supported endpoints are concrete scalar bounded revolute, prismatic, and
helical joints. Chains compose recursively; cycles and manifold/custom/fixed
endpoints fail at build. Bounds are sign-aware and reduced generalized
capacities use the matching absolute- or squared-scale accumulation.

Tests use an explicitly unconstrained BetterRobot twin as the full-space
oracle. Pinocchio's default Panda loader leaves the second finger independent,
so it is not used as a constrained-mimic oracle.
