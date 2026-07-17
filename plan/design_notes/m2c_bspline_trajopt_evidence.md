# M2c B-spline trajopt evidence

Pre-gate probe on `542abdb` (2026-07-17) reproduced all three T2c.4 failures:

- Manifold safety: with seed 7, `T=12`, `C=5`, and unit quaternion samples,
  `BSplineTrajectory.expand()` produced quaternion norms in
  `[0.151953, 1.145826]`.
- Bounds: a fixed-base call with finite `lower`/`upper` reached the B-spline
  `_ChainRuleProblem` with both stored bounds equal to `None`.
- Multi-stage: selecting `LMThenLBFGS` raised
  `TypeError: _ChainRuleProblem.__init__() missing ... 'dq_dz'` during
  `dataclasses.replace`.

Decision: reject every non-`KnotTrajectory` robot trajopt parameterisation at
the facade. Fixed-base B-splines would still misrepresent bounds and
multi-stage support, so a floating-base-only rejection is insufficient.
`BSplineTrajectory` remains a Euclidean numerical basis utility, not a safe
robot-trajopt capability. M5 owns a real fix: manifold interpolation and
retraction, a matching chain-rule Jacobian, feasible bounds, and replace-safe
solver state.
