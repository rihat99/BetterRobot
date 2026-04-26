# Batched by default

Every BetterRobot tensor carries a leading batch dimension. There is
no "unbatched mode" and no `if x.dim() == 1:` branching in the hot
path: a single configuration is `(1, nq)`, sixty-four configurations
are `(64, nq)`, and a batch-of-time-batched ones are
`(*B, T, nq)`.

This matters for two reasons.

1. **GPU throughput.** Solver inner loops want a single Cholesky on a
   `(*B, nv, nv)` tensor, not a Python loop over the batch.
2. **`torch.compile` friendliness.** Static joint-type dispatch
   (`tuple[JointModel]`) plus a fixed loop over `model.topo_order`
   compile cleanly per `(shape, dtype, device)` triple.

## Floating-base joints

Free-flyer / planar / spherical joints have `nq != nv` (a quaternion
parameterisation of a 6-DoF twist, etc.). `model.integrate(q, v)` and
`model.difference(q0, q1)` dispatch per joint, so callers never need to
special-case the floating base.
