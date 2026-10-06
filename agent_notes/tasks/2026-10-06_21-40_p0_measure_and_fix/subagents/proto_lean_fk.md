# E3: lean batched FK prototype

Script: `proto_lean_fk.py` (same folder). Run: `CUDA_VISIBLE_DEVICES="" uv run python <script>`. No library source changed.

## Method
- Plan (built once, not counted): q gather columns; one dense affine map `W (nfeat, nj*16)` + `bias (nj*16)` that
  turns a feature vector into every joint's local 4x4 `[R0 R(q) | p0 + R0 t(q)]` in level order.
  Features: revolute `sin q, cos q` (Rodrigues with R0 folded in: `R0 R = (R0+R0K^2) + s R0K - c R0K^2`);
  spherical / free-flyer quaternion `vec(q q^T)/|q|^2` (16 per joint; rotation is quadratic in q, R0 folded in);
  prismatic and free-flyer translation coordinates are linear features. Fixed = bias only. Mimic is folded
  into the gather (`addmm(o, q, E^T)` instead of `index_select`).
- Composition: per tree level, `parent_world[3x4] @ local[4x4]` (one gather + one matmul; gather skipped when
  the parent level has one joint). Then one cat + one inverse-permutation gather back to joint order.
- Output is `(B, nj, 3, 4)`; `R_world`, `p_world` are views of it.
- Counting: torch.profiler CPU, batch 256 float32. `top` = aten calls whose parent is not an aten op
  (what Python dispatches, views included); `leaf` = innermost aten calls minus views/allocs (kernel proxy).
  The 219/796 numbers from proto_level_fk.md were CUDA kernels; the library is re-measured here the same way as lean.
- Timing: CPU, 1 thread, float32, median of 25 after 5 warm-up.

## Parity (float64, 64 random q)
| model | nj | nq | levels | feats | max pos err | max rot err | max grad err (d sum p / dq) | (B,nj,7) err |
|---|---|---|---|---|---|---|---|---|
| panda | 14 | 8 | 12 | 16 | 3.3e-16 | 7.6e-7 * | 5.3e-15 | 3.3e-16 |
| g1 | 41 | 36 | 13 | 77 | 8.9e-16 | 8.9e-9 * | 1.1e-14 | 8.9e-16 |
| tree25 | 25 | 99 | 10 | 387 | 1.3e-15 | 1.6e-15 | 8.0e-15 | 1.3e-15 |

\* Not a lean error. Placement quaternions were rounded to float32 at load, so `R0` is off-orthonormal by ~1e-7.
The library composes with that same `R0` (positions match to 1e-16) but normalises its output quaternion, so
`to_matrix(lib quat)` differs from the raw composed R. Lean converted to (B,nj,7) with `so3.from_matrix` matches
the library pose to 3e-16. Normalising R0 in the plan instead makes positions differ by 3e-7.

## Operator counts (batch 256)
| model | lane | fwd top | fwd leaf | fwd+bwd top | fwd+bwd leaf |
|---|---|---|---|---|---|
| panda | library | 246 | 196 | 720 | 643 |
| panda | lean | 33 | 45 | 179 | 156 |
| panda | lean + (B,nj,7) | 127 | 106 | 485 | 462 |
| g1 | library | 336 | 311 | 1192 | 1054 |
| g1 | lean | 57 | 47 | 248 | 195 |
| g1 | lean + (B,nj,7) | 151 | 108 | 554 | 501 |
| tree25 | library | 298 | 260 | 978 | 870 |
| tree25 | lean | 45 | 36 | 190 | 148 |
| tree25 | lean + (B,nj,7) | 139 | 97 | 496 | 454 |

The (B,nj,7) conversion (`so3.from_matrix` + cat) alone costs about 94 top / 60 leaf forward and about 300 forward+backward.

## CPU time, ms (1 thread, float32)
| model | lane | B=1 fwd | B=1 f+b | B=256 fwd | B=256 f+b | B=4096 fwd | B=4096 f+b |
|---|---|---|---|---|---|---|---|
| panda | library | 0.662 | 2.129 | 2.257 | 5.776 | 35.2 | 73.3 |
| panda | lean | 0.080 | 0.542 | 0.418 | 1.531 | 5.2 | 15.8 |
| g1 | library | 0.904 | 3.396 | 5.544 | 15.21 | 138.4 | 234.8 |
| g1 | lean | 0.130 | 0.800 | 1.311 | 4.248 | 19.4 | 57.6 |
| tree25 | library | 0.795 | 2.828 | 3.755 | 10.14 | 83.6 | 148.7 |
| tree25 | lean | 0.115 | 0.669 | 1.672 | 4.815 | 25.5 | 97.2 |

## Where lean's remaining calls go (from the code path; not profiled per op)
- Local stage, about 8 compute calls whatever the joint count: q gather (or mimic addmm), sin, cos, quaternion
  outer mul, |q|^2 (diagonal sum), divide, feature cat, one addmm. Plus a few view calls in `top`.
- Per level: one matmul (bmm plus up to 2 contiguous copies when broadcasting a sliced/gathered parent) and one
  index_select gather when the parent level has more than one joint, plus slice views. 10-13 levels here, so the
  levels are now most of the count.
- End: one cat of levels, one inverse-permutation gather (skipped when level order equals joint order).

## What limits going lower
- Tree depth: one matmul per level is the only sequential dependency. Pointer jumping would give log2(depth) steps
  but each step composes all joints (more flops).
- The dense `W` is (nfeat, nj*16): 387 x 400 for tree25. For large trees it grows as nj^2; a block-sparse or
  per-family einsum version costs about 2 more calls per family but stays linear.
- Backward roughly 4x forward calls (autograd of matmul/addmm/cat/index_select).
- The (B,nj,7) quaternion format: converting back costs more calls than the whole lean FK.

## Not covered
Unbounded revolute (cos/sin q), planar, translation, helical, composite, mimic of a revolute joint
(mimic is folded into the gather generally, but only Panda's prismatic finger was exercised). Plan raises
`NotImplementedError` for unknown kinds. Only a single batch axis `(B, nq)`; no batched placements.
