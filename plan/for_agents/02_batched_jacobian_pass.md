# Order 02 — Batched `joint_jacobians_raw` (kill the per-joint loop)

Read `plan/README.md` first. This is the FK-round treatment applied to the
Jacobian pass: same outputs, same signature, per-joint Python loop replaced
by a fixed handful of batched ops. Order 03 builds J̇ on the helper this
order introduces — get the decomposition right here.

## Current shape (the problem)

`jacobian.py:39-102`: per joint in `topo_order` — parent copy, quat→matrix,
`hat_so3`, a **per-joint `joint_motion_subspace(q_j)` call** (allocates and
index-fills), `.to()`, three small matmuls, two in-place index writes.
~15–25 kernel launches × njoints, launch-bound on GPU, and the only
genuinely sequential-looking step (the parent copy) is not sequential at
all — the column structure is static.

## Target formulation

The world Jacobian has a purely static sparsity: column block of joint `a`
appears in row-joint `j`'s Jacobian iff `a ∈ supports(j)`, and the column
content `Ad(oM_a) @ S_a` depends only on its **owner** `a`, never on `j`.
So:

1. **Own-column table** (no sequencing): batched over all joints at once,
   - `R (B..., njoints, 3, 3)` from one batched `so3.to_matrix` on
     `joint_pose_world[..., :, 3:]`; `p`, `hat(p)` likewise batched;
   - `S (njoints, 6, max_nv)` from the constant
     `structure.joint_motion_subspaces` table (verified: every shipped
     joint's subspace is q-independent; the per-joint
     `joint_motion_subspace(q_j)` calls are redundant — delete them from
     this pass). Cast the table once to the compute device/dtype exactly
     as the current loop does per joint (`jacobian.py:89`). Note the
     dynamics passes (`dynamics/rnea.py:168,196`, `crba.py:64`) consume
     the table with **no** cast, relying on `model.to()` co-location — do
     not copy that; keep the Jacobian pass's existing promotion behavior;
   - `cols_lin = R@S_lin + hat(p)@(R@S_ang)`, `cols_ang = R@S_ang` — three
     batched matmuls over the `(B..., njoints)` axes.
2. **Scatter to full width**: place the `(B..., njoints, 6, max_nv)`
   padded blocks into a shared `(B..., 6, nv_full)` column table with a
   static index built once from `idx_vs_full`/`nvs_full` (padding columns
   drop out; any static scatter — `index_copy` on a flattened column axis
   or a precomputed selection matmul — is fine as long as it is
   `torch.func`- and compile-safe and does not break under vmap; follow
   what `_fk_matrix.py` does for its kind-scatter).
3. **Per-joint masking**: `J = shared.unsqueeze(-3) * mask` with a static
   `(njoints, 1, nv_full)` boolean/float mask derived once from
   `supports` (`support_offsets`/`support_indices`,
   `model_structure.py:141-142`). One broadcast multiply replaces the
   whole propagation loop.
4. **Reduce last**: `reduce_jacobian(structure, J)` unchanged
   (`jacobian.py:102`).

Static artifacts (owner map, scatter index, support mask) are a pure
function of `ModelStructure` — memoise on the structure instance exactly
like `_fk_matrix_plan` (`_fk_matrix.py:157-169`; `.to()` mints a fresh
structure, so a moved model rebuilds). Keep the helper **private to
`kinematics`** and shaped so order 03 can call it: it must expose the
shared full-width column table and the static owner map, not just the
final per-joint tensor.

Signature, output field, dtype/device/broadcast behavior, and mimic
handling of `joint_jacobians_raw` are all unchanged. `q` is still consumed
through `broadcast_to_execution_batch`/`expand_configuration` only insofar
as still needed (with the constant-subspace table, `q_full` may become
unnecessary in this pass — if so, drop the dead computation, but do not
change the public signature).

## Tests and bench

- Oracle: the existing suites are the spec — `tests/kinematics/test_jacobians.py`
  (raw-vs-workspace, all-references, batched, fullgraph-compile,
  finite-diff), `tests/kinematics/test_mimic.py:69-73`,
  `tests/test_pinocchio/test_frame_jacobian_matches_pinocchio.py`, and
  order 01's new joint parity file if already landed. No tolerance may
  loosen.
- Add one focused test only if a genuinely new surface appears (the
  memoised plan rebuild after `.to()`, mirroring the FK plan test if one
  exists); otherwise add nothing — behavior is unchanged.
- Bench (scratch, not committed): launches/joint and wall-clock before vs
  after, protocol of the FK round (torch.profiler `cudaLaunchKernel`
  count, median of 30 after 5 warmups, synchronized) on the free-flyer +
  spherical zoo (SMPL-25 / SMPLX-56 / MHR-203 shapes) and Panda, forward
  and forward+backward. Report numbers in `02_results.md`. Target:
  launches independent of njoints (~O(10) per pass); flag any case where
  the batched pass is slower on CPU for tiny models — small CPU
  regressions are acceptable if GPU wins are large, but they must be
  measured and reported, not guessed.

## Acceptance

- Full CPU gate green, zero tolerance changes, `ruff` clean.
- Bench table in `02_results.md` with launches/joint and ms before/after.
- The private helper's contract (returned tensors + owner map) documented
  in its docstring — order 03 consumes it.

## Out of scope

J̇ (order 03). `frame_jacobian_raw` internals (already thin post-J).
Reusing FK's world matrices (non-goal, see README). Warp.
