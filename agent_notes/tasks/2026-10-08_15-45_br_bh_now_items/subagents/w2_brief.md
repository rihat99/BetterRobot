# Wave 2 — BetterHuman items and the documented roadmap

Read `context.md`, `../PLAN.md` (owner decisions — binding) and the wave-1 reports `w1_*.md` in this folder (what
changed in BetterRobot; BR is at the state on disk, uncommitted). Same rules as `w1_brief.md` (tests + docs with every
public change, no commits, report ≤ 150 lines in `subagents/w2_<slice>.md`, final message ≤ 15 lines).

## F `bh` — BetterHuman worktree `/home/rikhat.akizhanov/better/dev/BetterHuman` (all of it), run its gates
(`CUDA_VISIBLE_DEVICES="" BETTERHUMAN_MODELS_DIR=$PWD/models uv run pytest tests -q -p no:cacheprovider`,
`uv run ruff check src tests`, `uv run mkdocs build --strict`). Detail in
`../../2026-10-08_13-02_br_bh_release_prep/subagents/{p1_human.md,p2_bh.md,p2_bh2.md,p2_studio.md,p3c_aux_findings.md}`.
1. Call-site fallout from wave 1 (keyword-only ids in BR kinematics/dynamics/terms/collision, `Viewer.on_pointer`,
   `CameraLayer` import path, picking helper moved to BR `viewer/tools/picking.py`): make BH's src/tests/examples/docs
   use the new BR API; HumanStudio registers pointer callbacks through `Viewer.on_pointer`/`clear_pointer` and uses
   BR's picking helper instead of its own `_nearest_on_ray`.
2. `Rig.corrective_features(spec, model, q, *, identity=None)` (keyword-only identity input so STAR/SUPR can read
   shape coefficients): update the `Protocol`, all three implementations and `Body`; invalidate the feature memo at
   `core/body.py:~285` when identity changes; test that a rig receiving `identity` sees the same tensor `Body.bake`
   used. Record in `docs/contracts/bh-addendum.md` that this is breaking for third-party Rigs.
3. `FamilySpec` docstring lists every `meta` key that core reads (grep `meta[` in `core/` and `families/`), with
   shape/type; a test asserts the documented set equals the keys core reads (keep it simple: a frozen set in the
   module, test compares to a grep-like scan or to the loaders' output).
4. `docs/contracts/bh-addendum.md`: time axis `(B, T, nq)` already at line ~92 — add `fps`/`times` semantics in line
   with BR's viewer time model (see `w1_br_viewer.md`).
5. `HumanState.vertices()` and `frame_poses()` become properties (cached like `joints`); update every call site,
   docs snippets and the studio; test that `state.vertices.shape` works.
6. `tests/core/test_reduction.py::test_corrective_slice_is_exact_when_dropped_joint_is_at_rest`: compare at float32
   one-ulp tolerance (`rtol` ~1e-6, exact in float64) — owner's decision; keep the float64 exactness assertion if the
   test has one, otherwise add a float64 variant that stays exact.
7. Studio: `frames.active` resets on model switch (left item from `p2_studio.md`); `.numpy()` → `.detach().cpu().numpy()`
   in the editors (`p2_bh2.md` item 1); `mhr_rig.build_model` device comparison (`torch.device("cuda")` vs `cuda:0`)
   compares `device.type`/index properly so the MHR tree is cached (`p2_bh2.md` item 2); test for the cache hit.
8. Document: third-party families are in-tree only during alpha (`register_family` not offered); retargeting and
   fitting are not BH's job (one line each in `docs/concepts/architecture.md` or the addendum).
Verify the studio on a live server once (127.0.0.1:8082, `drive_studio_p2.py` from the previous task folder,
screenshots under `vis/studio/`), then stop it.

## G `roadmap` — `docs/reference/roadmap.md` in both repos (+ mkdocs nav if needed), `docs/concepts/architecture.md` "where next" lines only
Source: the site `../../2026-10-08_13-02_br_bh_release_prep/site/{index,optim,kindyn,collision,viewer,human,toolbox}.html`
(roadmap tables, now/next/later/maybe) and the owner's decisions in `../PLAN.md`, which override the site:
drop fitting; retargeting is a separate package; third-party families in-tree only; sparse = pure torch first,
CHOLMOD maybe; DDP/MPC in scope; contacts = contact Jacobians only, constrained dynamics far future; Warp reserved;
biomechanics joints in BH; capsules next; viser only; npz export; BVR later; STAR next; mocap import (C3D/BVH/AMASS)
next, marker sets later, muscles maybe; MJCF experimental; batch-first target.
Write the roadmap as a reader-facing page: a short intro (what "now/next/later/maybe" mean, that `now` items landed
in this version — list them from the `w1_*.md`/`w2_bh.md` reports with their contract/ADR links), then one table per
theme: brick | why | prerequisite | when. Keep BR's existing "explicit-raise inventory" block untouched (a contract
test checks it). BH's `docs/reference/roadmap.md` is currently the "Vision" page: keep the vision text, add the
roadmap section below it. `uv run mkdocs build --strict` clean in both repos; `tests/docs` pass.
