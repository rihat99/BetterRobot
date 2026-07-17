# Residual sparsity semantics retained for M5

This note preserves the useful information in the legacy
`optim/jacobian_spec.py::ResidualSpec`. It is design input, not an API
proposal. The class had no production consumer, so the semantics below are
the conservative guarantees M5 may rely on rather than claims that sparse
assembly already worked.

## Legacy fields

- `dim` was the residual's reported row count. `output_dim` defaulted to
  `dim` and represented the row-slot count available to an assembler.
- `tangent_dim=None` meant the full legacy flat tangent variable; otherwise
  it recorded the tangent-column extent the residual could touch. In the
  block API this must become per-`reads`-variable metadata, not one global
  integer.
- `structure` described possible nonzeros, never numerical values:

  | Kind | Preserved meaning |
  |---|---|
  | `dense` | Any residual row may depend on any tangent coordinate of the touched variable block. This is also the safe default when metadata is absent. |
  | `diagonal` | Row groups align one-to-one with tangent-coordinate groups, with no off-diagonal coupling (for example, per-knot scaled identities). It is stronger than `block`; unequal dimensions need an explicit grouping rule. |
  | `block` | Nonzeros are confined to one or more localized rectangular blocks, each of which may be internally dense. The legacy fields did not encode row-to-block incidence, so this label alone is insufficient for assembly. |
  | `banded` | Repeated temporal blocks lie within a bounded stencil about the time diagonal. The legacy type encoded neither offsets nor bandwidth, so M5 must add those explicitly. |

- `time_coupling` refined trajectory locality:
  - `single`: a row group observes one knot. This was the default even for
    non-trajectory residuals, where it carried no useful temporal claim.
  - `5-point`: the intended fixed five-knot finite-difference neighbourhood.
    It did **not** specify centering, boundary handling, or relative offsets,
    and no shipped residual produced it; M5 must not infer those details.
  - `custom`: arbitrary coupling whose exact pattern needs an explicit M5
    declaration.
- `affected_knots` held absolute knot indices in the evaluated trajectory.
  The only implemented producer used `(t_idx,)`. It did not represent a
  reusable relative-offset stencil; M5 needs a distinct offset convention
  for repeated temporal rows.
- `affected_joints` and `affected_frames` were unordered model-index sets
  bounding possible kinematic support. Joints identify tangent inputs that
  may affect the rows; frames identify observed frame outputs from which
  joint support may be derived using the model topology. They are static,
  conservative supersets, not row order, row-to-index incidence, or a
  promise that every listed index is numerically nonzero. Empty tuples mean
  "not declared" and therefore provide no zero-support guarantee.
- `dynamic_dim=True` was inconsistent across legacy sources: the class said
  row length may change, while the collision stub reserved a fixed candidate
  row per pair and changed only active support. M5 must conservatively treat
  either evaluation-varying rows or activity as non-static structure and
  reject/fall back to dense unless a fixed padded layout is explicitly
  proven.
- Deprecated `input_indices` named flat legacy input coordinates.
  `is_diagonal=True` rewrote the default `dense` kind to `diagonal` in
  `__post_init__`. Neither alias should return in the M5 design.

The implemented evidence was limited to `TimeIndexedResidual`
(`block`, `single`, one absolute knot) and a collision stub (`block`, dynamic,
joint indices). Velocity and acceleration had useful two-/three-knot
Jacobians and matrix-free transpose products but never declared a `.spec`;
older documentation calling them `banded` or `5-point` was aspirational.

## Deterministic M5 implications

M5 should extend the same M2a `Problem`; it must not introduce a parallel
problem type. A future declaration must be static and per variable named in
`reads`, with exact time offsets and row/block incidence. It is a
conservative support claim: under-declaring a possible nonzero is an error,
while an absent declaration remains dense within that read variable.

Symbolic analysis runs once from shapes and declarations, never tensor
values. Normalize index sets deterministically (validated, unique, sorted),
but do not let metadata reorder numerics. Dense row order remains residual
item order then the residual's own row order; dense column order remains
`Problem.vars` order, reduced free tangent-coordinate order, and knot-major
order inside a trajectory block. Batch axes never affect the pattern.

For a temporal row stencil with relative offsets `O`, Jacobian blocks occur
at those offsets and normal-matrix offsets are all differences
`{a - b | a, b in O}`. Thus a three-knot stencil gives a
block-pentadiagonal normal matrix. M5 must parity-check every declaration by
densifying it and proving both declared values and the exactly-zero
complement against the dense oracle. Dynamic or unsupported declarations
must have an explicit dense fallback or honest rejection.

## Milestone boundary

M2a v1 provides deterministic **dense** assembly of per-(residual,
variable) Jacobian blocks. Its `reads` declaration establishes only coarse
inter-variable zeros; a `(T, nq)` trajectory may still be one large dense
block. M2a ships no symbolic temporal pattern, banded storage/factorization,
Schur elimination, or sparse/matrix-free routing, and this note does not add
one.

M5 owns the reviewed declaration surface, symbolic intra-block analysis,
structured assembly and operators, banded solvers, and routing. It should
revive the information above in M2a's final vocabulary—not resurrect
`ResidualSpec` or its aliases—and retain M2a dense assembly as the default
and parity oracle.
