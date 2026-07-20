# Order 04 — Warp: FK for every model, RNEA kernel, honest fallback

Read `plan/README.md` and `DESIGN_RULES.md` first. Warp code is invisible to
users; it is still held to full polish (DESIGN_RULES rule 12). Every kernel
change lands with parity + gradcheck + timing evidence, measured on a pinned
idle GPU (`CUDA_VISIBLE_DEVICES=2|3|7`), median of ≥30 reps after warmup,
`torch.cuda.synchronize` bracketed.

## Audited baseline (2026-07-20)

- Warp FK forward-only kernel + torch-recompute VJP works and is well-tested
  (15 CUDA tests: parity, first/second-order gradcheck, CUDA-graph capture,
  stream interop). SMPL-like 25-joint model: **0.72 ms vs 17.2 ms eager
  torch (~24×)** at B=1; essentially flat to B=4096.
- **Panda never runs the Warp lane**: `try_warp_forward_kinematics` returns
  `None` for any model with mimic joints (the finger coupling), and the
  fallback to torch is silent. The flagship robot silently misses the fast
  path.
- Dead ABI: the op takes `nqs` (`_warp_bridge.py:95,157,198`) that the kernel
  never reads.
- ~200 of 310 kernel lines are f32/f64 twins (`_compose_*`,
  `_joint_transform_*`, `fk_frames_*`).
- Inside kernels, quaternion/SE(3) math is **native Warp** (`wp.quat*`,
  `wp.transform*`, hand-written compose) — the torch `lie/` package is never
  called inside a kernel; it appears only in the backward recompute. A Warp
  RNEA needs no ported lie library: `wp.spatial_vector`, `wp.spatial_matrix`,
  `wp.spatial_cross`, `wp.spatial_cross_dual`, `wp.transform_twist`,
  `wp.transform_wrench` cover the spatial algebra 1:1 with `rnea.py`'s
  helpers.
- The eager torch pipeline is kernel-launch-bound (Panda IK B=1024: 46k
  launches, 200 ms CPU dispatch vs 66 ms GPU compute) — this is why fused
  kernels matter more than dense-algebra tuning.

## T1 — Mimic joints in the Warp FK kernel

Support mimic models in-kernel: apply the packed q expansion
(`mimic_multiplier`/`mimic_offset`, already device-resident on
`ModelStructure`) when reading q, so Panda stops falling back. Update
`try_warp_forward_kinematics` eligibility, delete the mimic fallback branch,
and extend `tests/warp/` with Panda forward+gradient parity vs torch
(fp32/fp64) and a mimic-specific gradcheck. The existing mimic-fallback test
becomes a mimic-runs-on-warp test.

## T2 — Fallback honesty and ABI cleanup

1. When a caller passes `use_warp=True` and the lane declines (remaining
   reasons: unsupported dtype/joint kind, layout, batch-shape mismatch),
   emit a one-shot `warnings.warn` naming the reason — mirroring Order 01's
   AD-fallback warning; hard error during CUDA-graph capture stays.
2. Remove the dead `nqs` parameter from the op signature, fake, forward
   call, and test `_direct_inputs`.
3. f32/f64 duplication: attempt a collapse (Warp generic funcs / template
   codegen). Keep only if the result is at least as readable and parity +
   timing are unchanged; otherwise report the attempt and keep the twins.
   Do not sacrifice kernel clarity for LOC.

## T3 — Warp RNEA

`rnea.py:90-244` is a two-pass Featherstone over `topo_order` with per-joint
lists + stack — the same shape as FK before its kernel. Port it following the
FK lane's proven pattern:

1. One kernel (per dtype policy from T2): FK sweep producing `liMi`/`oMi`
   (reuse the existing kernel bodies), then the forward velocity/acceleration
   pass and backward force pass in the same serial per-thread sweep.
   Spatial ops via the `wp.spatial_*` builtins listed above; motion subspace
   per joint kind mirrors `_joint_transform_*` dispatch.
2. Forward-only, `record_tape=False`; backward = torch recompute VJP exactly
   like `_WarpFKFunction` (`_warp_bridge.py:31-82`) so Warp's off-manifold
   adjoint is never used.
3. Opt-in seam identical to FK: `rnea(..., use_warp=True)` via a
   `try_warp_rnea` in a new `dynamics/_warp_bridge.py` (or extend the
   kinematics bridge if the code shares >half its plumbing — your call,
   justify it in the results file). External torques/`fext`, gravity, mimic
   (after T1) must all be supported or decline with the T2 warning.
4. Tests in `tests/warp/`: parity vs torch RNEA (Panda + SMPL-like,
   fp32/fp64, B=1 and B=4096, with and without fext), gradcheck through the
   recompute VJP, CUDA-graph capture if the FK harness generalizes cheaply.
5. Timing table in the results file: warp vs eager torch vs
   `torch.compile`d torch RNEA at B=1/4096 on both models. If
   `torch.compile` matches Warp within ~2×, say so plainly — the owner
   decides lane policy on evidence, not enthusiasm.

## T4 — Small GPU-truth items

1. `examples/03_batched_ik.py`: its toy 1-DOF model runs ~13× slower on GPU
   than CPU (launch-bound). Keep auto-selection but print an honest note when
   the model is tiny, or default the example to CPU with a `--device` flag —
   pick one, document why.
2. `docs/concepts/the_compute_seam.md`: update for RNEA's Warp lane and the
   fallback warning; state the launch-bound reality and when GPU pays off
   (large batches / trajopt-sized problems).

## Acceptance

- `uv run pytest tests/ -q -m cuda` and `tests/warp/` fully green, including
  the new Panda-mimic and RNEA suites; full CPU gate green.
- Panda FK runs the Warp lane (prove with the fallback-probe pattern from the
  audit: warp-vs-torch output diff nonzero path taken, or an eligibility
  assert).
- Recorded timing tables for FK (now including Panda) and RNEA.
- No torch call inside any Warp kernel; no silent fallback under
  `use_warp=True`.
- Contract files you may touch: hot-path lint watch list (new bridge module),
  public-import contracts if `rnea` gains a parameter.
