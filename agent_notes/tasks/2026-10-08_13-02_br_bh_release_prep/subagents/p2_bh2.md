# P2 sweep — slice `bh`, second half (BetterHuman, worktree `/home/rikhat.akizhanov/better/dev/BetterHuman`)

Follow-up to `p2_bh.md` (not redone). Budget 50k tokens, machine load average 897-970 during the run (13 users),
so micro-benchmarks were skipped (rule: skip above 100). No source file was changed.

## Files read
- End to end: `core/contact_frames.py`, `core/classic.py`, `families/mhr_tree.py`, `families/mhr_rig.py`
  (lines 20-98), `viz/human_layers.py` (helpers, `frames_layer`, `forces_layer`, `HumanView.update`).
- Partly: `io/load.py` (resolve + `load_model_npz`, docstrings skipped), `viz/studio/state.py` (lock, `set_q`,
  `rotate_joint`, `move_root`, `load_body` signature).
- Grep-only (sync/device/except/pickle/lock hazards): `io/{schema,convert_chumpy,__init__}.py`,
  `families/{mhr,mhr_archive,mhr_config,tables}.py` (`tables.py` has no functions, data only),
  `families/assets/densities.py`, `viz/studio/{layers,persistence,posing,segment_editor,frames_editor,__main__}.py`.
- NOT read: `io/schema.py` and `io/convert_chumpy.py` bodies, `families/mhr.py` / `mhr_archive.py` bodies,
  `viz/studio/{layers,persistence,posing}.py` bodies. `app.py`, `gizmo.py`: out of scope (other agent).

## Fixed (what, where, test)
None. Nothing found met the "small local behaviour bug" bar; the items below are either unreachable through the
public entry points, setup-time cost, or design choices.

## Reported, not changed (ranked; severity; evidence; suggested fix)
1. **Studio editors call `.numpy()` without `.cpu()` (low, latent).** `viz/studio/frames_editor.py:71`
   (`self.session.vertices.detach().numpy()`), `viz/studio/segment_editor.py:90` and `:99` (`part_ids.numpy()`).
   Every other studio/viz path uses `.detach().cpu().numpy()` (`layers.py:44-387`, `gizmo.py:76`,
   `human_layers.py:28`). A `StudioSession` built from a CUDA body would raise on paint/frame-pick. Not reachable
   today: the app always builds the body with `bh.load(...)` on CPU (`app.py:86`, no device option). Suggested
   fix: add `.cpu()` at the three sites (3 one-word edits) if CUDA sessions are ever wanted.
2. **`AuthoredRig.build_model` device check compares `torch.device(device)` with the tensor's device (low, perf).**
   `families/mhr_rig.py` `build_model`: `torch.device("cuda") != torch.device("cuda:0")`, so passing
   `device="cuda"` rebuilds the whole MHR tree (`build_mhr_tree`, Python loops over the archive entries) on every
   `Body` construction (`core/body.py:48,112` call it once each) even when the cached tree is already on that GPU. Not measured (load). Suggested fix: compare
   `robot.joint_placements.device.type`/index after normalising, or `torch.empty(0, device=device).device`.
3. **`quaternion_to_axis_angle` has zero gradient in the negative-scalar singular band (info).** Probe (float64):
   `q=(0,0,0,-1)` and `(1e-9,0,0,-1)` -> `(2*pi,0,0)`, grad all zeros; `(0,0,0,1)` -> grad `(2,2,2,0)` (correct);
   generic `q` round-trips to 5.6e-17. No NaN anywhere. The constant fallback is intentional (documented
   sign-preserving converter); only matters if someone optimises through it at exactly 2*pi.
4. **MHR hard-coded sizes (info).** `mhr_tree.py` uses `num_classes=125 * 3` and `mhr_rig.py` uses
   `unflatten(-1, (125, 3))`; guarded by the `(nq, nv, nq_full, nv_full) == (132, 131, 179, 178)` check, so a
   different MHR LOD/archive fails loudly rather than silently. Could derive `125 = len(joint_names) - 2`.
5. **`accumulated_scales` assumes parents precede children** (`accumulated[parent]` with `parent < joint`), true for
   the MHR archive; per-bake Python loop over 127 joints (bake-time, not per pose). Info.
6. **`corrective_features` uses a dense one-hot matmul `(..., E) @ (E, 375)`** where an `index_add` would do;
   E is ~130, so cost is tiny. Info, not measured.
7. **Docs: `CLAUDE.md` benchmark command omits `--out`** (`CLAUDE.md:13`:
   `uv run python benchmarks/run.py --device cpu --threads 8`), but `--out` is required (`run.py` exits 2 without
   it, confirmed by `p2_bh.md`). `README.md:141` gives the full command. Suggested fix: append `--out <path>` in
   `CLAUDE.md` (or give `--out` a default).

Checked and found correct: contact-frame vertex normals (face/normal ordering of `index_add_` with
`repeat_interleave(3)` matches `faces.reshape(-1)`), `_basis_from_z` seed choice, SMPL pelvis-pivoted root ->
free-flyer (`transl + J0`), MHR mimic multiplier `weight / host_weight` and host-scaled sorted limits,
Euler accumulation of several archive rows on one joint, `human_layers._frames` single-pose -> `T=1`,
`io.load_model_npz` uses `allow_pickle=False`.

## Performance notes (measured)
- Micro-benchmarks skipped: load average 897-970 (> 100).
- `benchmarks/run.py --device cpu --threads 8 --out artifacts/perf/bh_cpu.json`: see Gate results. Numbers taken
  under load ~900 are NOT comparable with stored results; use only for "it runs".

## Gate results
- `benchmarks/run.py --device cpu --threads 8 --out .../artifacts/perf/bh_cpu.json`: exit 0, all SMPL / SMPL-X / MHR
  cells up to batch 8192 done; raw json `artifacts/perf/bh_cpu.json` (20 KB), log `artifacts/perf/bh_cpu.log`.
  Taken at load ~900: timings not interpreted.

- Full BH suite (worktree state at ~13:30, includes `p2_bh`'s `spec.py` fix and whatever other agents had edited):
  `3 failed, 332 passed, 1 skipped, 13 warnings in 540.04s (0:09:00)`. Failures:
  `tests/core/test_reduction.py::test_corrective_slice_is_exact_when_dropped_joint_is_at_rest` (known);
  `tests/docs/test_snippets.py::test_page_snippets_execute[docs/getting-started.md]` and
  `tests/docs/test_snippets.py::test_getting_started_outputs_fresh[getting-started.md]` (NEW, not diagnosed:
  budget). These were not in `p2_bh`'s run, and this slice changed no file. Likely causes: a concurrent edit of
  `docs/getting-started.md`/studio/API, or the `drop_joints` fix changing a printed output. The orchestrator should
  run `pytest tests/docs/test_snippets.py -q` and read the stderr. Log:
  `/tmp/claude-1003/-home-rikhat-akizhanov-better/d9220856-907e-4a64-b1a1-f6cd72ec6355/scratchpad/bh2_full_suite.log`.

- Tests of touched files: none (no file touched). `ruff`: not needed (no file touched).
