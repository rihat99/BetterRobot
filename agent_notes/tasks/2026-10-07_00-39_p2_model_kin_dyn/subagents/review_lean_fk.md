# Review: lean FK, KinematicState, parser precision (mars_aux, clean context, 2026-10-07)

Scope: commits 0ebac62 and a0c3ab1. Reviewer scripts were in /tmp/claude-1003/review_fk/ (not kept).

## Findings and dispositions
1. MAJOR. A revolute axis that is not unit length (possible through `ModelBuilder`; the URDF/MJCF parsers
   normalise) made `forward_kinematics_state` return matrices that are not rotations (difference 3.88 against
   `br.forward_kinematics` for an axis of length 2). FIXED: the plan normalises rotary axes.
2. MINOR. The cached placement matrices went stale after an in-place edit of `values.joint_placements`.
   FIXED: the cache is keyed on the tensors' version counters.
3. MINOR (older, widened). A first FK call inside `torch.inference_mode()` stored inference tensors in the plan
   and broke later autograd calls. FIXED: plans are built outside inference mode; the placement cache is skipped
   in inference mode.
All three are pinned by `tests/kinematics/test_state.py::test_state_review_regressions`.

## Verified correct by the reviewer
Mixed-kind model (unaligned revolute and prismatic, unbounded revolute, non-root free-flyer, planar, spherical,
helical, mimic, branches) against the per-joint dispatch: 7e-16, also with batched placements. Tree levels.
G1 floating base, batch (3,2): frame Jacobians in all three references against the Data API (2e-15) and autograd.
`torch.func` jacrev/jacfwd/vmap. No graph leak from the cache. Batched placements with singleton axes.
Oracle regeneration and tighter tolerances judged justified; no test got weaker.

## Not checked by the reviewer
CUDA, `use_compile`, MJCF change contents, `terms/robot.py`, gradients along the quaternion norm, zero-coordinate
models, batch shape `()`, `frame_jacobian` with mimic against autograd, benchmark claims.
