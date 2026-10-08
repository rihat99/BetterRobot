# w2_bh — BetterHuman slice F (worktree /home/rikhat.akizhanov/better/dev/BetterHuman, uncommitted)

## Done
1. Studio on new BR API (`viz/studio/app.py`): `viewer.on_pointer("click"|"rect-select", ...)` / `viewer.clear_pointer()`
   replace raw `server.scene.on_pointer_event`/`remove_pointer_callback`; private `_nearest_on_ray` removed, BR
   `better_robot.viewer.tools.nearest_on_ray` used. No other BR-API fallout found in BH src/tests/docs (full suite green).
2. `Rig.corrective_features(spec, model, q, *, identity=None)` in the Protocol, `RegressedRig`, the 2nd rig in
   `core/rig.py`, and MHR `AuthoredRig`; `HumanState` passes `identity=self.identity.params`.
   `HumanState._memo` is now `init=False`, so `dataclasses.replace(state, identity=...)` gets a fresh memo
   (before, replace shared the old dict). Test `tests/core/test_contact_frames.py::test_rig_corrective_features_receives_the_baked_identity`
   (spy sees `identity.params` by identity). Breaking note in `docs/contracts/bh-addendum.md` "Rig protocol changes".
3. `core/spec.py`: `META_KEYS` frozenset (13 keys) + FamilySpec docstring with type/shape per key;
   `tests/core/test_spec_meta.py` regex-scans core/ and families/ for `meta[...]`, `meta.get(...)`, `"k" in meta`.
4. Addendum "Batching": `(B, T, nq)` time-axis paragraph with BR viewer `fps`/`times` semantics.
5. `HumanState.vertices` and `.frame_poses` are properties (memoised); all call sites in src/tests/README/docs updated.
6. `test_corrective_slice_is_exact_when_dropped_joint_is_at_rest`: float32 rtol 1e-6/atol 1e-7, float64 rtol 1e-15.
   DEVIATION: float64 is NOT bitwise exact (1 / 6 elements differ by 1.4e-14 abs, 2e-16 rel = one ulp), so the
   brief's "exact in float64" is impossible; float64 now asserts one-ulp tolerance. Owner should confirm.
7. Studio: `frames.active = None` on `load_model`; editors/app use `.detach().cpu().numpy()`;
   `mhr_rig._same_device` treats `cuda` (no index) as the current device, so `build_model` reuses the cached tree.
8. Addendum "Scope": families in-tree only during alpha (no `register_family`); retargeting and fitting are not BH's job.

## Gates
- BH pytest: 337 passed, 1 skipped (before the item-6 tolerance fix, only that test failed; it now passes;
  full suite NOT re-run after that one-line test change). ruff check clean. mkdocs build --strict clean.

## Not done (budget ran out)
- No test for the MHR device cache hit (needs CUDA or a mocked `torch.cuda.current_device`).
- No test for `frames.active` reset on model switch.
- Live studio check on 127.0.0.1:8082 + screenshots NOT run.
- `docs/concepts/architecture.md:52` updated to the new signature; no other doc pass.
