# M3 parity safety-net matrix

Audited on 2026-07-17 before T3.1 test additions. `P` is a Pinocchio
numeric oracle, `L` is a BetterRobot batched-vs-scalar-loop oracle, and `R`
is a local roundtrip/property oracle. `U`, `Q1`, `QM`, and `V` mean
unbatched, one-axis q batch, multi-axis q batch, and batched model values.

## Pre-extension coverage

| Joint kind(s) | Oracle/pass | Batch | Base | Dtype |
|---|---|---|---|---|
| fixed, `revolute_rz`, `prismatic_py`, `prismatic_unaligned` | P: Panda FK, frame Jacobian, RNEA, CRBA, ABA, integrate/difference | U | fixed | fp64 |
| same Panda kinds | L: RNEA, CRBA, ABA | Q1 | fixed | fp64 |
| `revolute_rx/ry/rz`, free-flyer, fixed | P: G1 RNEA | U | free-flyer | fp64 |
| spherical, `revolute_rz`, fixed | P: programmatic-chain RNEA | U | fixed | fp64 |
| planar, translation, helical, revolute-unbounded/unaligned | FK-runs-only checks; no numeric/property oracle | U | fixed | default fp32 |
| mimic | build rejection only | — | — | — |

The audited holes are therefore `QM` everywhere, Pinocchio-backed batched FK
and frame Jacobians, all fp32 numeric parity, all `V` parity, and meaningful
coverage of the five last-row joint families. Fixed- and free-flyer-base
scalar fp64 dynamics are represented, but free-flyer coverage is RNEA-only.

## T3.1 closure

T3.1 adds Pinocchio-looped FK and frame-Jacobian checks at `Q1` and `QM`,
an fp32 Panda FK/RNEA band, and programmatic Pinocchio/property cases for the
missing joint families. It also adds the reusable
`assert_value_batched_matches_loop` scaffold. Actual `V` cases remain T3.2
work because they require the completed value-broadcast sweep; the helper is
the committed oracle contract, not a claim that value batching is covered.

| Added oracle | Joint kinds | Batch | Base | Dtype |
|---|---|---|---|---|
| P: FK + all frame-Jacobian references | Panda kinds | Q1, QM | fixed | fp64 |
| P: FK + RNEA | Panda kinds | U | fixed | fp32 |
| P: FK + RNEA; R: integrate/difference | planar, translation, helical, revolute-unbounded/unaligned | U | fixed | fp64 |

Finding for T3.7: planar FK and RNEA match Pinocchio, but the two libraries'
`integrate` translations differ today (BR adds world-coordinate
`dx,dy`; Pinocchio applies the SE(2) body-coordinate retraction). T3.1 locks
BR's self-roundtrip only and does not disguise that convention decision as
Pinocchio parity.
