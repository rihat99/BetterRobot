# Review of the collision layer (mars_aux, clean context, commit d79e570 as of 4503509)

1. MAJOR — `self_collision_pairs` treated fixed joints as moving joints: on Panda, link7 (joint 8, revolute) and the
   hand (joint 10, fixed, via fixed joint 9) counted as grandparent/grandchild, so the (link7, hand) pair was included
   although the bodies are rigidly attached; it penetrated in 100% of 4000 random configurations (-0.023 m at READY),
   making any `self_clearance` built from the example's spheres unsatisfiable. Fix applied (72de682): joints are
   mapped to the nearest ancestor with coordinates before the same/parent-child test; test
   `test_pairs_ignore_fixed_joints`.
2. MINOR — commit d79e570 ("feat: minimal collision layer") also carries the Warp deletions (kinematics/_warp_*,
   dynamics/_warp_*, tests/warp, baselines) that belong to 688efdb: both agents worked in the same tree and the
   collision commit swept the other agent's staged deletions. Not pushed; splitting means rewriting `dev` history —
   left to the owner.
3. MINOR — docs: `docs/reference/collision_and_geometry.md` and four tracked autodoc pages described the removed stubs.
   Fixed (72de682): page rewritten, API pages regenerated.
4. NIT — gradient exactly zero (not NaN) with a centre on a capsule axis, coincident centres, or at the centre of a
   cube; subgradient at box corners. Documented in the reference page.
5. NIT — outside an SDF grid the value is held at the border with zero gradient. Documented.

Confirmed: all five distance functions against brute force / closed forms (max 2.6e-8 for the sampled capsule, else
≤1e-12); gradcheck passes everywhere except the box-corner kink; `centers` matches `frame_pose_of` to 1e-16 on a
(2, 5) batch; term shapes `(B, T, S*K)`, `window=1`, `check_jacobian` 5e-17; the example reproduces the report's
numbers (-0.092 m, +0.016 m, +0.020 m); layer rule respected; no stubs left.
