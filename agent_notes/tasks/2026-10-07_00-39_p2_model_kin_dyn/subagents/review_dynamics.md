# Review of the world-frame dynamics core (mars_aux, clean context, commit 4503509)

Scripts: scratchpad `review_dynamics/{f32.py,g.py,memo.py,data_fields.py}` (session-local, not kept).

1. MAJOR — float32 error grows with distance from the world origin (`world_inertia` about the world origin,
   columns use `cross(translation, angular)`). G1 free-flyer, 64 random q, float32 vs float64 max relative error:
   offset 0 m: rnea 4.7e-7, crba 5.1e-7; 10 m: 4.8e-6, 8.0e-5; 100 m: 6.5e-5, 6.1e-3 (old code: constant 4e-7 /
   2e-7). At 100 m the float32 M had smallest eigenvalue -0.040 and Cholesky raised.
   Fix applied (72de682): all world quantities about the first joint's position per sample; gravity's linear part
   shifted accordingly; COM shifted for the centroidal map. After: 100 m rnea 5.8e-6, crba 5.8e-6, M positive
   definite (remaining growth is the float32 representation of the 100 m configuration itself).
2. MAJOR — memo trap: `state._memo["world_kinematics"]` held the 6x6 inertia, so after an in-place edit of
   `body_inertias` `inverse_dynamics` was stale while `mass_matrix` (reading packed lazily) was fresh; a state reused
   with another model returned model 1's torques. Fix applied: memo holds model-independent geometry only; inertias
   are read from the model on each call, cached on the values keyed on `body_inertias._version` (same policy as
   `placement_matrices`). Test `test_state_reuse_sees_edited_inertias_and_other_models`.
3. MINOR — `joint_forces[..., 0, :]` included `fext[0]`; old code and pinocchio ignore it. Fix applied:
   `support[0, 0] = False`. Data fields now match the old code on all joints (differences 1.5e-5 = old float32 gravity).
4. MINOR / not verified — composite joints: the column subspace table is constant and the rate term uses the owner's
   velocity; a composite joint's subspace depends on q. Pre-existing (the old code assumed constant subspaces too);
   nothing tests composite joints in dynamics against pinocchio. Left open.
5. Symmetric completion of the mass matrix verified correct for 1-dof, spherical and free-flyer blocks.

Claims: float64 results and Data fields hold; gradcheck passes for inverse_dynamics (q, v, a, fext, body_inertias,
gravity), forward_dynamics (q, v, tau), mass_matrix (q, joint_placements), centroidal_map (q); vmap and jacrev run;
re-timed Panda CRBA rows match the table (53.5 ms / 307 ops at 4096). torch.compile not checked.
