# M5 sparse trajectory structure — Phase A design

**Status:** Phase A approved for implementation under the owner's delegated
unattended-work authorization, 2026-07-17. This document is based on the `dev`
tree after M2a–M3.5 and does not claim that the structured path exists yet.

## Authorization and hard boundaries

The owner's explicit standing instruction for unattended work was to accept
recommended decisions and defer decisions that still require evidence or
owner input. That instruction delegates the T5.2 decision gate for the
evidence-backed recommendations recorded here; it is not inferred from owner
silence. The following choices are therefore approved for Phase B. Items that
still lack evidence remain deferred and cannot be implemented under this
authorization:

| Decision | Phase A disposition |
|---|---|
| Trajectory encoding | **Proceed:** one `VarSpec` with event shape `(T, nq)`, `RobotConfig(model)`, and an explicit leading `time_axis=0`. |
| Compatibility | **Proceed:** extend the existing M2a `Problem`; do not add a second problem type and do not change existing dense semantics. |
| Eligibility | **Proceed:** temporal structure and structured numeric blocks/operators are opt-in. An undeclared residual, an unsupported mask, or a mixed temporal/shared-variable system routes the whole solve to the dense oracle. |
| Direct representation | **Proceed:** lower-triangle block-banded storage with dense within-knot tangent blocks. |
| Solvers | **Proceed:** a batched block-banded Cholesky direct path plus an explicit `NormalOperator` and fixed-work batched CG. |
| Schur elimination | **Defer:** there is no second production in-tree temporal-trajectory-plus-shared-variable caller. The T=1 `q + T_cam` acceptance test is not such a caller. |
| Robot B-splines | **Defer re-enabling in `solve_trajopt`:** the present `BSplineTrajectory` is a component-space Euclidean utility and is known to violate quaternion, bounds, and replacement semantics. This document defines its future sparsity/chain-rule interaction, but M5 must not silently advertise it as manifold-safe. |
| GPU/capture certification | **Defer to M6:** M5 is CPU-only. The proposed tensor programs are fixed-shape candidates, not CUDA-graph evidence. |

No external consumer repository was accessed for this design. BVR is a
prospective Schur caller, not in-tree evidence, and cannot authorize T5.6.
This paragraph and the table above are the durable approval record required by
the Phase A hard stop; Phase B must not broaden them silently.

## Verified starting point

### Current public contracts

The landed M2a/M2b implementation, rather than the old architecture sketch, is
the authority. The relevant signatures on this tree are:

```python
@dataclass(frozen=True)
class VarSpec:
    name: str
    shape: tuple[int, ...]
    manifold: Manifold = Euclidean()
    bounds: Bounds | None = None
    scale: torch.Tensor | None = None
    mask: torch.Tensor | None = None

Values = dict[str, torch.Tensor]

class Problem:
    def __init__(
        self,
        *,
        vars: Sequence[VarSpec],
        residuals: Sequence[ResidualItem] = (),
        objectives: Sequence[ObjectiveItem] = (),
        providers: Sequence[Provider] = (),
        parameters: Mapping[str, torch.Tensor] | None = None,
        differentiable_parameters: Sequence[str] = (),
    ) -> None: ...

    def residual(self, values, *, weights=None) -> torch.Tensor: ...
    def jacobian_blocks(
        self,
        values,
        *,
        weights=None,
        strategy="auto",
        create_graph=False,
        fd_eps=1e-4,
    ) -> dict[tuple[str, str], torch.Tensor]: ...
    def dense_jacobian(
        self,
        values,
        *,
        weights=None,
        strategy="auto",
        create_graph=False,
    ) -> torch.Tensor: ...

@runtime_checkable
class LinearSolver(Protocol):
    def solve(
        self,
        A: torch.Tensor,
        b: torch.Tensor,
        ridge: torch.Tensor | float | None = None,
    ) -> torch.Tensor: ...

@dataclass(frozen=True)
class LevenbergMarquardt:
    def init_state(self, values, problem, *, create_graph=False) -> LMState: ...
    def update(self, values, state, problem, *, create_graph=False) -> tuple[Values, LMState]: ...
    def run(self, values, problem, state=None) -> tuple[Values, LMState]: ...
```

Evidence: `optim/blocks/variables.py` defines trajectory event shapes and
flattens `RobotConfig(shape=(T, nq))` to `T * nv`; `optim/blocks/problem.py`
derives coarse variable dependencies through the provider DAG and assembles
only present `(residual, variable)` blocks; `optim/blocks/solver_lm.py`
currently materializes `J` and `J.T @ J` in `_evaluate_model`; and
`optim/solvers/base.py` still annotates `A` as a dense tensor even though the
roadmap reserved a callable seam.

The current custom-residual contract is deliberately duck typed:

```python
class Residual(Protocol):
    name: str
    reads: tuple[str, ...]
    dim: int
    def __call__(self, ctx: Mapping[str, Any]) -> torch.Tensor: ...
```

An optional `jacobian_blocks(ctx) -> dict[str, Tensor]` returns already
mask-reduced blocks of shape `(B..., dim, free_dim)`. `create_graph=False`
detaches numeric blocks; `create_graph=True` must remain connected or raise.
The M5 path preserves this contract.

### Plan-versus-tree findings

1. `solve_ik` is on the named-block `Problem`, but `solve_trajopt` still uses
   legacy `LeastSquaresProblem`, `CostStack`, and flat optimizers. M5 T5.7 must
   perform the trajectory task rebase; it cannot merely switch a solver flag.
2. `VelocityResidual`, `AccelerationResidual`, `TimeIndexedResidual`, and
   `ContactConsistencyResidual` still consume legacy `ResidualState`.
   Velocity/acceleration mutate `dim` on first evaluation and reject leading
   batch axes, so they cannot be registered in the current named-block
   `Problem` without a static-horizon adaptation.
3. `RestResidual` supports the named-block mapping protocol for a single
   configuration, but fixes `dim=model.nv`; using it directly on `(T, nq)`
   would return `(B..., T, nv)` rather than `(B..., T * nv)`. A static-horizon
   reference-trajectory form is required.
4. M2c deliberately rejects every non-`KnotTrajectory` robot parameterization.
   The recorded probe found quaternion norms `[0.151953, 1.145826]`, discarded
   finite bounds, and a `dataclasses.replace` failure. That rejection remains
   correct until a separate manifold-spline design is implemented.
5. The current named-block `Problem` requires a positive static `dim`, so the
   old collision-style dynamic-output case cannot enter this path. A padded,
   fixed-row residual may opt in; an evaluation-varying row count cannot.
6. Raw SMPL-like `model.lower_pos_limit` / `upper_pos_limit` cannot be attached
   directly to a `RobotConfig` `VarSpec`: 92 finite `[-1, 1]` entries are
   quaternion coordinates, while `RobotConfig.box_mask` contains only the
   three free-flyer translations. `tasks/ik.py:172-185` is the authoritative
   sanitization: keep model limits only where `box_mask` is true and replace
   every other entry by `(-inf, +inf)`. The existing `JointPositionLimit` is
   also not a trajectory residual: at T=50 it returns `(50,198)` rather than
   one static flat vector, and its SMPL-like `dq/dv` projection has zero
   nonzeros because every movable joint has `nq != nv`. It cannot be used to
   make the M5 benchmark appear to exercise limits.

### Recovered structural inputs

`plan/design_notes/residual_sparsity.md` and git history at `ae6a422` and
`3b39197` establish the useful mathematics:

| Residual | Row-time convention | Jacobian time offsets | Matrix-free transpose |
|---|---:|---:|---|
| central velocity | centers `t=1..T-2` | `{-1, +1}` with blocks `[-I,+I]/(2dt)` | two aligned slice accumulations |
| acceleration | centers `t=1..T-2` | `{-1,0,+1}` with `[I,-2I,I]/dt^2` | three aligned slice accumulations |
| time-indexed | one row group at `t_idx` | `{0}` from that absolute origin | one knot scatter |
| reference trajectory | one row group per knot | `{0}` | block diagonal |
| contact consistency | interval origins `t=0..T-2` | `{0,+1}` | two endpoint accumulations |

The old `ResidualSpec` labels (`dense`, `diagonal`, `block`, `banded`) were
insufficient because they did not encode row-to-time incidence or exact
relative offsets. They are not resurrected.

The required local consumer-scale evidence is intentionally documentary; no
consumer tree was opened or changed. `plan/01_assessment.md:236` records the
BVR-scale trajectory as `T x 211` variables, and
`plan/research/audit_optim_stack.md` §2.10 records the old dense allocation
pattern and a 30-DOF, T=240 case with roughly `7k x 7k` matrices per residual
per LM iteration. The operator seam follows
`references/design/jaxopt.md` §5.4 and §6 (JVP/VJP, callable solves, ridge
wrapping, diagonal recovery, and warm starts). The comparison with per-knot
variable identities follows `references/design/pyroki.md` §4. These are
design inputs, not claims that either external implementation is copied.

### Encoding prototype at T=500

The repository's current `Problem` was used to price the Python graph surface
of the two candidate encodings. Both cases used `make_smpl_like_model()`
(`nq=99`, `nv=75`), two smoothness families, and eight fixed keyframes. The
aggregate-block case had one current `VarSpec`, ten residual items, ten
`reads` edges, one `Values` entry, and a separate constant-size prototype time
annotation. The per-knot case had 500 `VarSpec`s, 1,004 residual items, 2,498
`reads` edges, and 500 `Values` entries.

Hardware and process: `robotics2-ESC8000-E11`, Intel Xeon Platinum 8570,
Python 3.11.15, PyTorch 2.13.0+cu126, CPU fp32, one Torch intra-op and one
interop thread. Each result is the median and inclusive IQR of 31 samples
after five warmups; Python allocation peaks use `tracemalloc`.

| Probe | One aggregate block | 500 per-knot blocks | Ratio |
|---|---:|---:|---:|
| static graph construction | 0.246 ms `[0.243, 0.252]` | 14.666 ms `[14.617, 14.734]` | 59.6x |
| current `_validate_values` | 1.387 ms `[1.379, 1.395]` | 282.689 ms `[258.585, 446.911]` | 203.9x |
| graph-construction Python peak | 12.1 KiB | 1,041.6 KiB | 86.3x |

This probe measures public graph construction, value-boundary validation, and
Python allocation only. It does **not** measure numeric Jacobian assembly,
`TemporalAnalysis`, or the not-yet-implemented `time_axis` validation; Appendix
A deliberately returns the prototype annotation beside, rather than inside,
the current `VarSpec`. The aggregate design adds one optional integer field and
an `O(number of residual offset pairs)` cached analysis, whereas the rejected
surface creates Python objects and dependency edges proportional to T. The
measured 59.6x construction and 203.9x validation penalties are therefore
evidence about API/graph scaling, not a claimed structured-assembly speedup.
T5.8, not this probe, owns numeric assembly and solve performance. The full
reproducer is in Appendix A.

## 1. Declaration surface: extend `reads`, do not fork `Problem`

### Variable annotation

Add one backward-compatible field at the end of `VarSpec`:

```python
@dataclass(frozen=True)
class VarSpec:
    # existing fields unchanged
    time_axis: int | None = None
```

M5 v1 accepts only `None` or `0`. `0` means the first event axis in `shape` is
time; leading axes in the actual tensor remain independent batch axes. Thus:

```python
q_spec = VarSpec(
    "q",
    shape=(T, model.nq),
    manifold=RobotConfig(model),
    bounds=robot_bounds,
    time_axis=0,
)
# values["q"] has shape (B..., T, nq)
# its full tangent has shape (B..., T, nv)
```

No `TrajectoryVarSpec` subtype is introduced. `time_axis=None` preserves every
M2a behavior. Allowing arbitrary event axes adds layout complexity without a
caller; non-leading time axes fail with an error that says to transpose the
event layout at construction.

For a structured path, a static `mask` must be absent or separable over time:
after reshaping to `(T, tangent_width)`, every row must be identical. This gives
one constant reduced block width `d`. Arbitrary time-varying masks remain valid
for dense M2a solves but make the problem ineligible for M5 structured routing.
`scale` may vary by knot because it changes numeric diagonal scaling, not the
symbolic block width.

### Exact residual pattern

Add one small public immutable value and one optional duck-typed hook:

```python
@dataclass(frozen=True)
class TemporalPattern:
    rows: int
    row_width: int
    row_origin: int
    offsets: tuple[int, ...]

class StructuredResidual(Protocol):
    def temporal_structure(
        self,
        variable_name: str,
    ) -> TemporalPattern | None: ...
```

For row group `r in range(rows)`, offset `o` names variable knot
`r + row_origin + o`. Validation requires:

- `rows > 0`, `row_width > 0`, and `rows * row_width == residual.dim`;
- sorted, unique, non-empty integer offsets;
- every resulting knot index lies in `[0, T)`;
- the variable is in the residual's derived variable dependencies (including
  provider dependencies); and
- the variable has `time_axis=0`.

This one form covers repeated banded stencils, diagonal trajectories, interval
residuals, and fixed columns:

```python
# velocity
TemporalPattern(rows=T - 2, row_width=nv, row_origin=1, offsets=(-1, +1))
# acceleration
TemporalPattern(rows=T - 2, row_width=nv, row_origin=1, offsets=(-1, 0, +1))
# reference trajectory
TemporalPattern(rows=T, row_width=nv, row_origin=0, offsets=(0,))
# one time-indexed 6-vector at t_idx
TemporalPattern(rows=1, row_width=6, row_origin=t_idx, offsets=(0,))
# contact interval rows
TemporalPattern(rows=T - 1, row_width=3 * contacts, row_origin=0, offsets=(0, +1))
```

Boundary formulas with different stencils are separate residual items. That is
more checkable than a `custom` escape hatch. Missing
`temporal_structure(variable_name)`, a returned `None`, or a residual with a
non-static output is **dense**, never inferred from numerical zeros.

Static row counts cannot be zero. Constructors therefore enforce horizon
semantics before `Problem` construction: the recovered central
`VelocityResidual` and `AccelerationResidual` both require `T >= 3` and raise
`ValueError` for `T < 3` rather than changing their stencil; interval/contact
residuals require `T >= 2` and raise for `T < 2`; per-knot reference/envelope
residuals require `T >= 1`; and `TimeIndexedResidual` requires
`0 <= t_idx < T`. A task at a shorter horizon may simply omit an inapplicable
term. No constructor silently installs an empty residual or an endpoint formula
with different mathematics.

### Numeric structured blocks

A residual may also supply:

```python
def temporal_jacobian_blocks(
    self,
    ctx: Mapping[str, Any],
    variable_name: str,
) -> Mapping[int, torch.Tensor]:
    """offset -> (B..., rows, row_width, free_tangent_width_per_knot)."""
```

The keys must exactly equal `TemporalPattern.offsets`. The blocks contain the
residual's own derivative but not `ResidualItem.weight` or robust IRLS scaling;
`Problem` applies those once, consistently with dense assembly. The evaluation
context gains a structural `temporal_free_indices(variable_name)` accessor for
the repeated per-knot mask. As with current analytic blocks,
`create_graph=True` must remain connected or fail clearly.

The declaration and numeric hook are separate on purpose. A declaration is a
checkable support claim. Numeric blocks enable direct assembly. A declaring
residual without numeric blocks can still use autograd JVP/VJP, but it does not
make the direct banded route eligible.

### Eligibility and fallback

Eligibility is static. For this analysis, **potentially active** means every
`ResidualItem` in `Problem.residuals` whose derived dependencies include the
trajectory variable, including an item whose Python or tensor weight currently
equals zero. A zero weight never grants sparsity eligibility because phase or
runtime weight overrides can reactivate the item. Callers that want a residual
removed from the symbolic problem must omit it from the `Problem`; neither
`TemporalAnalysis` nor routing inspects tensor values.

M5 defines two nested eligibility levels:

1. **Operator eligible:** exactly one `VarSpec` has nonzero `free_dim`, it has
   `time_axis=0`, its static mask is absent or time-separable, every potentially
   active residual declares a valid `TemporalPattern`, no such residual also
   depends on another optimized variable, and the per-knot reduced width is
   constant. External parameters and providers remain allowed.
2. **Direct eligible:** operator eligibility holds and every potentially active
   residual also supplies `temporal_jacobian_blocks` with the exact declared
   offsets.

Additional optimized variables, including independent dense variables, make
M5 v1 ineligible: there is no band-plus-dense or disconnected-system container.
Fully masked zero-free-dimension `VarSpec`s do not participate. The immutable
`TemporalAnalysis` stores both eligibility booleans plus the first stable
reason code and an actionable detail string naming the offending variable or
residual.

Stable reason codes are:

```python
LinearizationReason = Literal[
    "forced_dense",
    "eligible_banded",
    "explicit_matrix_free",
    "no_time_variable",
    "multiple_optimized_variables",
    "nonseparable_mask",
    "undeclared_temporal_residual",
    "missing_temporal_blocks",
    "mixed_optimized_dependency",
    "explicit_dense_solver",
    "incompatible_solver",
]
```

The dense route remains the correctness oracle. Missing declarations or blocks
are never inferred from numerical zeros.

### Exact owner of routing

`LevenbergMarquardt`, not `Problem` or `solve_trajopt`, owns the requested
linearization and solver selection:

```python
LinearizationMode = Literal["auto", "dense", "structured", "matrix_free"]
LinearSystemKind = Literal["dense", "banded", "operator"]

@dataclass(frozen=True)
class LinearizationDecision:
    requested: LinearizationMode
    used: Literal["dense", "banded", "matrix_free"]
    reason: LinearizationReason
    detail: str

@dataclass(frozen=True)
class LevenbergMarquardt:
    # existing fields retain their order and meaning
    linear_solver: LinearSolver | None = None
    linearization: LinearizationMode = "auto"

    def resolve_linearization(self, problem: Problem) -> LinearizationDecision: ...
```

`linear_solver=None` is the route-aware default. It preserves current dense
behavior by choosing `Cholesky` for dense systems, while choosing
`BandedCholesky` for banded systems and `NormalCG` for operator systems. An
explicit solver may expose a static
`supported_systems: frozenset[LinearSystemKind]`; an existing/custom solver
without that attribute is treated as dense-only, preserving the current
extension contract. Shipped `Cholesky`/`LSTSQ`, `BandedCholesky`, and `NormalCG`
advertise `{"dense"}`, `{"banded"}`, and `{"operator"}` respectively.

The complete routing matrix is:

| Request | Eligibility | Representation/default solver | Failure/fallback |
|---|---|---|---|
| `dense` | irrelevant | dense tensor / `Cholesky` | explicit non-dense solver raises `ValueError` with `incompatible_solver` |
| `structured` | direct eligible | `BlockBandedMatrix` / `BandedCholesky` | any failed condition raises `ValueError` with the cached reason/detail; no fallback |
| `matrix_free` | operator eligible | `NormalOperator` / `NormalCG` | any failed condition raises; missing numeric blocks use the explicit autograd fallback |
| `auto`, solver `None` | direct eligible | banded / `BandedCholesky` | otherwise dense / `Cholesky`, recording the first eligibility reason |
| `auto`, explicit solver | choose the first compatible route in priority `banded`, `dense`, `operator` | the explicit solver | if none is eligible and supported, raise with `incompatible_solver` |

The `auto` default never selects the basis-vmap autograd matrix-free fallback:
if direct blocks are absent it goes dense. Selecting `NormalCG` explicitly (or
requesting `matrix_free`) is the opt-in for that cost. An explicit dense-only
solver under `auto` deliberately keeps the solve dense even when banded
structure is available and records `explicit_dense_solver` as the fallback
reason rather than silently ignoring the user's solver.

`solve_trajopt` accepts `optimizer: LevenbergMarquardt | None = None`; `None`
constructs `LevenbergMarquardt(max_iter=max_iter, linearization="auto")`.
Forced dense remains available as
`optimizer=LevenbergMarquardt(linearization="dense")`. The rebased
`TrajOptResult` adds the exact fields `linearization_requested`,
`linearization_used`, `linearization_reason`, and `linearization_detail`, copied
from `optimizer.resolve_linearization(problem)`. Other tasks need not expose a
duplicate mode argument.

## 2. What a trajectory variable is

The selected encoding is one annotated variable:

```python
VarSpec("q", (T, nq), manifold=RobotConfig(model), time_axis=0)
```

Column order remains the M2a order: `Problem.vars`, then mask-reduced tangent
coordinates, with knot-major order inside `q`. The annotation only supplies a
checked reshape from `(T * d,)` to `(T, d)`; it does not change values,
retraction, provider reads, or dense Jacobian columns.

The rejected alternative is one named variable per knot. On the current API it
turns two smoothness terms and eight keyframes at T=500 into 500 variable
objects, 1,004 residual objects, and 2,498 dependency edges. The measured
construction, validation, and allocation overhead above is already large
before evaluating FK or one residual. It also makes phase masks, provider
declarations, diagnostics, and user code scale in Python with T. Pyroki/jaxls
can afford per-knot identities because its solver owns a purpose-built analyzed
factor graph; copying that surface onto the existing M2a `Problem` would be a
second redesign.

## 3. Symbolic block sparsity

`Problem.__init__` computes and stores an immutable `TemporalAnalysis` after it
has derived provider-to-variable dependencies. It contains only Python/static
shape data and is reused for every evaluation. It is never keyed on tensor
values or recomputed per iteration.

For a row stencil `O`, the normal-matrix offsets are

```text
H_offsets = {a - b | a in O, b in O}
```

The union over residuals determines half-bandwidth
`w = max(abs(offset))`. Velocity alone has Hessian offsets
`{-2, 0, +2}`; acceleration has `{-2,-1,0,+1,+2}`; their union is
block-pentadiagonal with `w=2`.

The numeric normal matrix uses padded lower-band storage:

```python
@dataclass(frozen=True)
class BlockBandedMatrix:
    bands: torch.Tensor  # (B..., T, w + 1, d, d)
    bandwidth: int       # bands[..., t, k] == H[t, t-k], t >= k
```

Invalid edge slots (`t < k`) are zero. Lower-only storage avoids duplicating
symmetry and feeds factorization without a conversion. A `densify()` testing
helper is not used by production solves.

For T=500, d=75, fp32, w=2:

- dense `H`: 5,625,000,000 bytes = 5.239 GiB;
- one dense acceleration `J`: 5,602,500,000 bytes = 5.218 GiB;
- padded lower bands: 33,750,000 bytes = 32.187 MiB;
- a duplicated five-band layout would be 53.644 MiB.

These are exact element counts, not runtime measurements.

## 4. Structured assembly and JVP/VJP operators

### Direct assembly

For each residual item, reshape the weighted raw residual into
`(B..., rows, row_width)`. Compute its current grouped robust weights with the
same normalized `kernel.weight` convention as dense LM, expand the square-root
row scale, and apply it to both residual rows and temporal Jacobian blocks.
Lower-band orientation is canonical and never uses `abs(a-b)`. For each row
origin `k = r + row_origin`, iterate each unordered offset pair exactly once as
`a >= b` and accumulate

```text
# once for every offset a
gradient[..., k+a] += J_a.T @ residual_row

# once for every unordered pair a >= b
bands[..., k+a, a-b] += J_a.T @ J_b   # H[k+a, k+b]
```

The diagonal pair `a == b` is also visited once. The unstored upper block is
defined as the transpose of this lower block. If an implementation starts from
an arbitrary ordered pair, it must swap the offsets and transpose the product
before storage; accumulating both orders would double-count. Contractions are
vectorized over row origins and batch axes. The implementation loops only over
the small static offset tuple; it never loops over T in Python and never
materializes `(dim, T*d)`.

The stored physical-tangent bands are unscaled. The LM system builder applies
the static variable scale to both block axes as `S H S`, matching the current
`scaled_jacobian = J * scale` path. A dynamic per-element active-bound mask
then restricts the system as `M S H S M`; fixed coordinates receive the same
identity equation and zero RHS as the dense active-set path. Keeping physical
blocks also lets prediction and projected-gradient safeguards use the same
operators as the dense oracle. Scaling and masking change numeric entries, not
symbolic width.

### Operators

Every route supplies the same physical-tangent operator surface. Its Jacobian
already includes the frozen square-root robust row scale, exactly like current
`_evaluate_model`; it does not include variable scale or the active mask:

```python
class JacobianOperators(Protocol):
    def jvp(self, tangent_flat: torch.Tensor) -> torch.Tensor: ...
    def vjp(self, cotangent: torch.Tensor) -> torch.Tensor: ...
    def normal_matvec(self, tangent_flat: torch.Tensor) -> torch.Tensor: ...
    def normal_diagonal(self) -> torch.Tensor: ...
```

LM-facing tangents are flat `(B..., T*d)` to preserve all existing retraction,
mask, and step-limit helpers; a structured implementation views them as
`(B..., T, d)` internally. Residual cotangents are `(B..., dim_total)`.
`normal_matvec(v)` is exactly `vjp(jvp(v))`, and `normal_diagonal()` returns
`diag(J_w.T @ J_w)` in physical tangent coordinates. Velocity and acceleration
use the recovered aligned two- and three-slice math.

The dense-only `_ModelEvaluation` is replaced, not conditionally overloaded,
by one representation-independent evaluation object:

```python
@dataclass(frozen=True)
class _LinearizedLeastSquares:
    residual: torch.Tensor
    robust_weights: torch.Tensor
    cost: torch.Tensor
    gradient: torch.Tensor
    normal_diagonal: torch.Tensor
    operators: JacobianOperators
    normal: torch.Tensor | BlockBandedMatrix | None
    grad_norm: torch.Tensor
    projected_gradient: torch.Tensor
    projected_grad_norm: torch.Tensor
    active_mask: torch.Tensor
    finite: torch.Tensor
```

For the dense route, `normal` is the current dense `J_w.T @ J_w` and the
operators close over the evaluation-local dense `J_w`. For the banded route,
`normal` is the physical-tangent `BlockBandedMatrix` and the operators close
over the same temporal blocks used to assemble it. For the matrix-free route,
`normal is None`; gradient and normal products are obtained through VJP/JVP.
The object and its closures live for one `init_state`, `update`, or `finalize`
call only; no autograd graph is cached on `Problem` or solver state.

The current dense uses in LM are replaced exhaustively as follows:

| Current dense operation | Route-independent replacement |
|---|---|
| damping initialization from `(J * scale).T @ (J * scale)` diagonal | `(normal_diagonal * scale.square()).amax(-1)` |
| `_solve_step` dense normal construction | route-specific system builder described below |
| LM prediction `model.jacobian @ lm_actual_step` | `operators.jvp(lm_actual_step)` |
| projected-gradient denominator `model.normal_matrix @ pg1_step` | `operators.normal_matvec(pg1_step)` |
| projected-gradient prediction `model.jacobian @ pg_actual_step` | `operators.jvp(pg_actual_step)` |
| final-point `_evaluate_model` | route-aware `_linearize_model` with the same resolved decision |

Residual/candidate evaluation, robust cost decrease, projected-gradient KKT
logic, retraction, acceptance, and per-element damping updates remain common.
This table is a completeness requirement for T5.7 tests; a structured update
must not retain a hidden dense-J fallback.

When a residual declares support but omits structured blocks, a matrix-free
request derives its JVP and VJP with `torch.func.jvp` and `torch.func.vjp`
through the M2a tangent retraction. The evaluation context is transform-local;
no graph is cached. Its normal diagonal uses the exact JAXopt-style basis-vmap
fallback, so this route is intentionally explicit and unsuitable for T=500
unless the custom residual supplies blocks or an exact diagonal hook. The
automatic T=500 route requires blocks.

IRLS weights are frozen at the current LM linearization. Operators apply the
same square-root row scaling as direct assembly; they do not differentiate the
weight function inside the linear solve.

### Matrix-free solver seam

M5 widens the accepted linear-system representation while preserving the
existing `solve(A, b, ridge) -> Tensor` entry point for custom solvers:

```python
MatVec = Callable[[torch.Tensor], torch.Tensor]

@dataclass(frozen=True)
class NormalOperator:
    size: int
    matvec: MatVec
    preconditioner: MatVec | None = None
    block_shape: tuple[int, int] | None = None  # (T, d) when temporal

    def __call__(self, vector: torch.Tensor) -> torch.Tensor:
        return self.matvec(vector)

class LinearSolver(Protocol):
    def solve(
        self,
        A: torch.Tensor | BlockBandedMatrix | NormalOperator,
        b: torch.Tensor,
        ridge: torch.Tensor | float | None = None,
    ) -> torch.Tensor: ...
```

Dense Cholesky/LSTSQ continue to reject non-tensors; `BandedCholesky` accepts
`BlockBandedMatrix`; `NormalCG` accepts `NormalOperator`. `size` must equal
`b.shape[-1]`; `block_shape`, when present, must multiply to `size`;
`preconditioner(v)` applies an approximate inverse to a vector of the same
shape, or `None` means identity. These static fields remove shape guessing and
carry block-Jacobi/diagonal preconditioning without hiding it in a closure.
Damping remains owned by LM, not hidden in a solver. For a scaled solve
coordinate `y`, physical step
`delta = S M y`, physical normal `H`, active mask `M`, and damping `mu`, every
route solves the identical system

```text
(M S H S M + diag(mu*M + (1-M))) y = -M S g.
```

Dense builds that tensor. Banded scales/masks its lower blocks and adds the
diagonal in band zero. Matrix-free wraps
`y -> M*S*H(S*M*y) + (mu*M + 1-M)*y`. Because active restriction is already
baked into these representations, each solver receives `ridge=None` from LM.
Direct calls to a solver outside LM may still pass `ridge`, which adds `ridge I`
with the current broadcast semantics.

The tensor-returning protocol is supplemented, not broken, by an optional
diagnostic hook:

```python
class LinearSolveStatus(IntEnum):
    SUCCESS = 0
    MAX_ITER = 1
    NONFINITE = 2
    BREAKDOWN = 3
    NOT_SPD = 4

class LinearSolveResult(NamedTuple):
    solution: torch.Tensor
    converged: torch.Tensor
    finite: torch.Tensor
    ok: torch.Tensor
    iterations: torch.Tensor
    residual_norm: torch.Tensor
    relative_residual: torch.Tensor
    status: torch.Tensor

class InformativeLinearSolver(Protocol):
    def solve_with_info(
        self,
        A,
        b,
        ridge=None,
        *,
        initial: torch.Tensor | None = None,
    ) -> LinearSolveResult: ...

@dataclass(frozen=True)
class NormalCG:
    max_iter: int = 100
    rtol: float = 1e-5
    atol: float = 0.0
```

Every implementation validates `ok == (converged & finite &
(status == LinearSolveStatus.SUCCESS))` elementwise. All diagnostic fields have
the independent batch shape `B...`; `solution` has `(B..., n)`, `iterations`
is int64, statuses are int8, and residual metrics use the working floating
dtype/device. `initial`, when provided, must exactly match `b`; direct solvers
accept only `None`, while `NormalCG` uses it as `x0`. The public
`solve(A,b,ridge)` signature remains unchanged and uses a zero CG initial value;
LM alone opts into `solve_with_info(..., initial=state.previous_linear_step)`.

`BandedCholesky` and `NormalCG` implement `solve_with_info`; their ordinary
`solve` returns only `.solution` for protocol compatibility. LM calls the
informative hook when present and otherwise wraps an existing solver result
with `converged=finite`, `iterations=0`, an explicitly computed residual norm,
and `SUCCESS`/`NONFINITE`. The shipped dense Cholesky keeps its existing
`cholesky_ex` info mask and maps nonzero info to `NOT_SPD`. `LMState` retains
`factorization_ok` as the `result.ok` compatibility field and adds fixed-shape
`linear_solve_iterations`, `linear_solve_residual_norm`,
`linear_solve_relative_residual`, and `linear_solve_status`. A non-successful
element gets a zero candidate step and damping escalation; after an attempt at
`mu_max`, the existing LM failure rule marks only that element
`LMStatus.FAILED`. The diagnostic status preserves whether the cause was CG
iteration exhaustion, nonfinite arithmetic, breakdown, or a non-SPD block
instead of conflating all failures with Cholesky.

`NormalCG.solve_with_info` has exactly the `LinearSolveResult` contract above.
It uses a fixed Python/static iteration cap and tensor masks for per-element
convergence, with no tensor-dependent Python branch. It records the first
converged iteration per element, continues dummy masked iterations to the
common cap, and reports both `residual_norm = ||A x-b||` and
`relative_residual = residual_norm / max(||b||, finfo.tiny)`. `finite` covers
the solution, recurrence scalars, and both residual metrics. A
batch element converges when
`residual_norm <= atol + rtol * ||b||`; `iterations=0` when its initial value
already satisfies that test and otherwise records the number of CG updates
actually used before masking. A
nonpositive/nonfinite `p.T @ A p` is `BREAKDOWN`; nonfinite state is
`NONFINITE`; a finite element above tolerance at the cap is `MAX_ITER` with
`converged=False`. Only `SUCCESS` with `converged & finite` is usable by LM;
failure is signaled by fields, not by manufacturing NaNs. The preconditioner is
block-Jacobi when direct diagonal blocks are available and scalar diagonal
otherwise. LM state carries the previous **successful attempted linear
solution** as a fixed-shape warm start; failed elements retain their prior warm
start, and a route or shape change starts from zero.

### Differentiation

The default `run` remains detached. `create_graph=True` on the small explicit
unrolled oracle must flow through temporal blocks/operators and pure Torch
factorization operations or raise at the unsupported residual. No custom
structured hook may silently detach. M5 promises no new second-order paths;
the existing engineering contract limits numerical grad-grad guarantees to
named tests and excludes true nonsmooth points.

M6 implicit differentiation recomputes the final robustified, active-free
optimality system. This design does not cache a stale factorization and does
not hide external parameters, so it does not preclude that backward.

## 5. Batched block-banded Cholesky

For `bands[..., t, k] = H[t,t-k]`, factor lower blocks in increasing t:

1. subtract previously formed `L[t,j] @ L[t,j].T` terms from the diagonal;
2. call `torch.linalg.cholesky_ex` on each `(B..., d, d)` diagonal block;
3. form the next at most `w` off-diagonal blocks with batched matmul and
   triangular solves; and
4. solve with one forward and one reverse static sweep.

The complexity is `O(B*T*w^2*d^3)` time and `O(B*T*w*d^2)` storage, linear in
T for fixed w and d. The outer T loop is inherent and static; all inner work is
batched Torch. M5 runs it eagerly on CPU. It is structurally capture-eligible
(fixed shapes and no host decisions) but is neither CUDA-certified nor
`torch.compile`-default: unrolling 500 steps can create an excessive compiled
graph, and M6 owns real GPU/capture evidence.

`ridge` is broadcast to the leading independent batch shape and added to every
diagonal block before factorization. Each `cholesky_ex` info tensor updates a
per-element `ok` mask. Failed elements use identity dummy factors so they
cannot contaminate siblings, then return NaN solutions; successful siblings
remain on their original batched operations. This composes with the current
LM zero-step/info policy without a batch-wide exception.

M5 follows the engineering dtype contract: factorization stays in the input
working dtype, fp32 or fp64, and the result preserves it. There is no hidden
fp32-to-fp64 promotion in v1. A future mixed-precision factorization option
requires its own accuracy/performance evidence and documented return cast.

## 6. Schur elimination: specified, deferred

The future arrow analysis would detect:

- one `time_axis=0` variable q whose q-only terms are block-banded;
- one or more small variables without a time axis;
- joint residuals that declare temporal q support and also read a shared
  variable; and
- no other undeclared temporal coupling.

With system

```text
[Hqq Hqs] [dq] = [bq]
[Hsq Hss] [ds]   [bs]
```

it would factor `Hqq` once, solve against `bq` and the s columns of `Hqs`,
form `S = Hss - Hsq Hqq^-1 Hqs`, solve the small dense S, and
back-substitute. The cost model is band factorization plus s banded triangular
RHS solves, storage `O(T*d*s)`, and dense `O(s^3)`. It is considered only for
small `s` and when the predicted arrow storage is below the full dense system.
The previously sketched `s <= 32` value is **provisional, not an approved
threshold**: there is no second caller or benchmark supporting it. A future
T5.6 review must measure at least the two real callers, name their `(T,d,s)`
shapes, and commit a crossover rule before any numeric cutoff is shipped.

No production in-tree task currently has this form. The
`tests/optim/test_accept_q_extrinsics.py` problem has q plus camera extrinsics
but T=1; `ProjectionResidual` is a residual building block, not a second
trajectory task caller. T5.6 therefore remains design only. Once two concrete
in-tree callers exist, this section is the starting contract, not permission
to implement speculatively.

## 7. Per-element batching

Symbolic analysis is batch-independent and cached once. Every numeric tensor
carries arbitrary leading batch axes:

- residual blocks: `(B..., rows, row_width, d)`;
- bands/factors: `(B..., T, w+1, d, d)`;
- RHS/steps: `(B..., T, d)`;
- damping, success, acceptance, and convergence: `(B...)`.

Every batch element gets its own ridge, factorization status, CG convergence
mask, candidate step, and LM accept/reject result. There is no shared damping
or shared factorization. Batched-versus-sequential tests use stated tolerances
and compare per-element statuses; they do not require bitwise equality near
acceptance thresholds.

## 8. What stays dense

The following are deliberately dense:

- each within-knot tangent block (`75 x 75` for the SMPL-like model);
- T=1 IK and every variable without `time_axis`;
- small nuisance/shared blocks;
- any problem containing an undeclared temporal residual;
- any temporal problem with a non-separable static mask;
- any mixed temporal/shared optimized-variable residual until Schur has two
  callers;
- the complete existing `jacobian_blocks`/`dense_jacobian` path, retained
  unchanged as the parity oracle; and
- B-spline control systems that fail the cost threshold below.

No numerical-zero detection, generic COO/CSR solver, or resurrected
`SparseCholesky` class is part of M5.

## 9. B-spline interaction and dense threshold

For any fixed basis B, let `S_t = {c | B[t,c] != 0}` be the statically computed
control support of sample t. A residual row group centered at r with knot
offsets O touches controls

```text
U_r = union(S_(r + row_origin + o) for o in O).
```

The exact control half-bandwidth is

```text
w_z = max_r(max(U_r) - min(U_r)).
```

For contiguous degree-p support, an equivalent bound is

```text
w_z <= p + max_r,o1,o2 |start(r+o1) - start(r+o2)|.
```

The current cubic open-clamped basis has at most four nonzero controls per
sample (`p=3`). Symbolic analysis must inspect the actual static nonzero basis
mask, including endpoint behavior, rather than rely on an approximate T/C
ratio.

The chain rule is a window contraction, never a Kronecker product:

```text
J_z[r,c] = sum over touched t of J_q[r,t] @ D_expand[t,c], c in S_t.
```

For a Euclidean basis `D_expand[t,c] = B[t,c] I`. A future manifold-safe
spline supplies local `d x d` expansion Jacobians on the same support. JVP and
VJP gather/scatter these windows directly; `torch.kron(B, I)` is forbidden.

For C controls, lower-band block count is

```text
band_blocks = C*(w_z + 1) - w_z*(w_z + 1)/2
dense_lower_blocks = C*(C + 1)/2.
```

The factorization work proxy is `C*(w_z+1)^2*d^3` versus
`C^3*d^3/3`. Route to dense if either `band_blocks >= dense_lower_blocks` or
`3*(w_z+1)^2 >= C^2`; otherwise the structured control path is eligible.
This is a deterministic no-claimed-win threshold derived from storage and
leading-order work, not a tuned speed ratio. T5.8 may later tighten it with
committed evidence.

The existing `BSplineTrajectory.expand(z) = B @ z` mixes raw configuration
coordinates and is not a robot-manifold map. M5 Phase B must keep
`solve_trajopt`'s rejection for it. Re-enabling requires a separately reviewed
joint-wise manifold spline or another map with explicit local Jacobians,
feasible bound semantics, and replacement-safe solver state. This is an
intentional plan deviation, not a claim that the sparse chain rule alone fixes
M2c's three correctness failures.

## 10. Manifold and tangent bookkeeping

The state at each knot is nq-dimensional and every band block is in the
per-knot reduced tangent of width d (nv=75 before masking for the SMPL-like
model). `VarSpec.retract` and `difference` already reshape
`RobotConfig(shape=(T,nq))` to `(T,nv)` and call vectorized
`Model.integrate/difference`; structured assembly must never index nq-sized
columns.

Velocity and acceleration retain the current identity-right-Jacobian
approximation. It is accurate for the small inter-knot motion regime and is
part of dense-versus-structured parity. Upgrading to exact right Jacobians is a
separate residual-math change and cannot be hidden in the sparsity port.

`Model.difference` uses relative transforms and the principal log, so a pure
quaternion sign flip does not itself create a 2-pi residual. At task ingress,
`VarSpec` first validates unit norms under the existing "validate, do not
silently normalize" contract. `solve_trajopt` then clones the valid input and
hemisphere-aligns each free-flyer and spherical quaternion slice successively:
flip `q[t]` only when `dot(q[t-1], q[t]) < 0`; an exact zero dot keeps the input
sign. Ground-truth/reference trajectories pass through the same clone-and-align
helper.

M5 does **not** canonicalize after every retraction. Current residuals consume
rotations through manifold `difference`/FK and are sign invariant; inserting a
nonsmooth representative-changing `where` into generic `Problem.retract` would
change the M2a contract without a numerical need. After `finalize`, the task
hemisphere-aligns a clone once for the returned `Trajectory`; solver state,
cost, and residual diagnostics remain those of the equivalent pre-aligned
rotations. A directed test supplies alternating quaternion signs, checks that
dense and structured smoothness stay bounded, and checks ingress/egress
representatives are aligned. At exactly pi, the documented principal-log axis
ambiguity remains and no derivative continuity is promised.

## 11. Phase C benchmark definition

### Machine and process

- Host: `robotics2-ESC8000-E11`.
- CPU: Intel Xeon Platinum 8570, two sockets, 56 cores/socket,
  2 threads/core; benchmark pins logical CPU 0 with `taskset -c 0`.
- Environment: `OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`,
  `OPENBLAS_NUM_THREADS=1`; the benchmark also calls
  `torch.set_num_threads(1)` and `torch.set_num_interop_threads(1)`.
- Device/dtype: CPU fp32; TF32 is irrelevant and no GPU result is inferred.
- Record Python, PyTorch, BetterRobot commit, kernel, and CPU model in JSON.

### Model and deterministic problem

Use exactly:

```python
model = make_smpl_like_model(
    height=1.75,
    mass=70.0,
    preserve_joint_order=False,
    device=torch.device("cpu"),
    dtype=torch.float32,
)
```

Assert `(model.nq, model.nv, model.njoints) == (99, 75, 25)`. Set
`seed=20260717`, `dt=1/30`, and T in `{50, 125, 250, 500}`. Set
`cpu_generator=torch.Generator(device="cpu").manual_seed(20260717)`, draw
`directions=torch.randn(3, 75, generator=cpu_generator,
dtype=torch.float32)`, then normalize exactly as
`directions /= torch.linalg.vector_norm(directions, dim=1, keepdim=True)`.
Set `u=torch.linspace(0,1,T,dtype=torch.float32)` and define

```python
tangent_amplitude = torch.tensor((0.04, 0.02, 0.01), dtype=torch.float32)
tangent_phase = torch.tensor((0.0, 0.4, 0.8), dtype=torch.float32)
tangent = sum(
    tangent_amplitude[k]
    * torch.sin(2 * torch.pi * (k + 1) * u + tangent_phase[k])[:, None]
    * directions[k]
    for k in range(3)
)
q_ground_truth = model.integrate(model.q_neutral.expand(T, -1), tangent)
q_initial = model.q_neutral.expand(T, -1).clone()
```

Both trajectories pass through the ingress hemisphere helper from §10.

Construct `RobotConfig` bounds with the exact `solve_ik` sanitization, never
with raw model limits:

```python
manifold = RobotConfig(model)
box = manifold.box_mask
raw_lower = model.lower_pos_limit.to(dtype=torch.float32, device="cpu")
raw_upper = model.upper_pos_limit.to(dtype=torch.float32, device="cpu")
robot_bounds = Bounds(
    lower=torch.where(box, raw_lower, torch.full_like(raw_lower, -torch.inf)),
    upper=torch.where(box, raw_upper, torch.full_like(raw_upper, torch.inf)),
)
assert torch.isneginf(robot_bounds.lower[~box]).all()
assert torch.isposinf(robot_bounds.upper[~box]).all()
q_spec = VarSpec(
    "q", (T, model.nq), manifold=manifold,
    bounds=robot_bounds, time_axis=0,
)
```

This is the implementation in `src/better_robot/tasks/ik.py:172-185`. The
SMPL-like model has no usable scalar joint-position boxes, so the current
`JointPositionLimit` is explicitly **not** in this benchmark.

Use eight keyframe indices
`torch.linspace(0, T - 1, 8, dtype=torch.float64).round().to(torch.int64).tolist()`
and assert they are unique. At each keyframe target the poses from the ground
truth trajectory for
`body_head`, `body_left_wrist`, `body_right_wrist`, `body_left_ankle`, and
`body_right_ankle`. The residual stack, in this fixed order, is:

| Term | ResidualItem weight | Kernel/group |
|---|---:|---|
| 40 time-indexed pose targets | 1.0; pose position scale 10.0, orientation scale 2.0 | L2, group 6 |
| central velocity | 0.05 | L2, group 75 |
| acceleration | 0.005 | L2, group 75 |
| reference-to-neutral at every knot | 0.01 | L2, group 75 |
| tangent-reference envelope hinge | 0.10 | L2, group 1 |

The final term is a new named-block `TrajectoryTangentEnvelopeResidual`, not an
adapter around `JointPositionLimit`. Let

```python
delta = model.difference(model.q_neutral.expand(T, -1), q)
half_width = torch.full((75,), 0.010, dtype=torch.float32)
half_width[:3] = 0.003
lower_hinge = torch.clamp(-half_width - delta, min=0.0)
upper_hinge = torch.clamp(delta - half_width, min=0.0)
output = torch.cat((lower_hinge, upper_hinge), dim=-1).reshape(
    *delta.shape[:-2], 2 * T * 75
)
```

For arbitrary leading batch axes the reshape preserves those axes. The
residual has `dim=2*T*75`, reads only `q`, and declares
`TemporalPattern(rows=T, row_width=150, row_origin=0, offsets=(0,))`. Its one
temporal block is the concatenation of diagonal `-1` indicators for
`delta < -half_width` and diagonal `+1` indicators for
`delta > half_width`, using the same identity-right-Jacobian approximation as
the reference/smoothness residuals. Thus it has nonzero SMPL tangent support
and both dense and structured routes use identical math. With the fixed seed,
the ground-truth trajectory has respectively 127, 323, 652, and 1,305 active
envelope coordinates for T in `{50,125,250,500}`; the benchmark asserts these
counts as
`int((model.difference(model.q_neutral.expand(T,-1), q_ground_truth).abs() > half_width).sum())`
against `{50:127, 125:323, 250:652, 500:1305}`, so the term cannot regress into
a dead residual. This is a deliberate,
recorded replacement for the plan's impossible raw joint-position-limit term.
All residuals return one flat static vector and no static mask is used.

Use `LevenbergMarquardt(max_iter=5, damping_parameter=1e-4, gtol=0,
xtol=0, ftol=0, linear_solver=None, linearization=PATH)`, with `PATH` equal to
`"dense"` or `"structured"`. The benchmark driver calls `init_state`, exactly
five `update` calls without a host-side early break, and `finalize`; it does not
use the eager `run` early-exit loop. Both paths use the same initial values and
residual order. The structured result must first pass the small-T parity suite;
the benchmark is not a correctness test.

### Measurement

- Each `(path,T)` case runs in a fresh subprocess so peak RSS is independent.
- The subprocess constructs the model, targets, residuals, `Problem`, and
  optimizer once. One separately timed cold solve is followed by two complete
  warmup solves and seven measured solves. Before every solve, call
  `gc.collect()` outside the timer, clone `q_initial` into a new `Values`, and
  call a fresh `init_state`; neither values, damping state, CG warm start, nor
  terminal status is reused across repetitions.
- The measured interval starts immediately before `init_state` and ends after
  `finalize`, so it includes exactly one initial linearization, five `update`
  calls, and one final-point evaluation. Report that elapsed time divided by
  five as time/update. Store median, inclusive Q1/Q3 via
  `statistics.quantiles(samples, n=4, method="inclusive")`, and all seven raw
  samples in seconds.
- Record cold construction/first-call time separately; do not mix it into the
  warm statistic.
- On this Linux host, convert `resource.getrusage(RUSAGE_SELF).ru_maxrss` from
  KiB to bytes by multiplying by 1024. Read construction-time current RSS as
  the resident-page field of `/proc/self/statm` times
  `os.sysconf("SC_PAGE_SIZE")`. Store both byte values and
  `incremental_peak_rss_bytes = peak_rss_bytes - construction_rss_bytes`; JSON
  presentation may additionally divide by `2**20` for MiB but slopes use
  bytes. Peak RSS is the process peak after the seven measured solves.
- A case has a 600-second wall cap and 16-GiB address-space cap. Record
  `DNF_TIMEOUT` or `DNF_OOM` with the cap and stderr; never replace a DNF with
  an extrapolated number. Set `RLIMIT_AS` immediately after imports and before
  model construction; record a harness error if current virtual size already
  exceeds the cap. Thread environment variables are set before Python starts.
- GPU memory fields exist in the JSON schema but contain `null` and
  `status="pending_m6"`.

### Acceptance rule

Fit ordinary least-squares slopes to `log(metric)` versus `log(T)` over every
successful point for each path, requiring at least three dense points. For
both median time/update and incremental peak RSS:

```text
structured_slope + 0.15 <= dense_slope
```

A repetition is successful only when final cost, residual, gradient, and
linear-solve metrics are finite; its final LM status is exactly one of
`{LMStatus.RUNNING, LMStatus.CONVERGED, LMStatus.STALLED_AT_BOUNDS}`; every
recorded linear solve has `LinearSolveStatus.SUCCESS`; and the reported route
is `dense` for the dense case or `banded` for the structured case.
`LMStatus.FAILED` and `LinearSolveStatus.MAX_ITER` are failures even though
their integer encodings are finite. Fixed-budget `RUNNING` is allowed and is
not relabeled as convergence.

Additionally, every structured T=500 repetition must be successful, and its
time and memory slopes must each be at most 1.35. If fewer than three dense
points complete, the milestone benchmark is inconclusive rather than passing
from analytic estimates. The committed JSON contains raw values, IQRs, slopes,
per-repetition LM and linear-solve statuses, DNF records, exact command, and
commit hash.

Suggested files for Phase C are
`tests/bench/bench_trajopt_sparse.py` and
`tests/bench/baselines/trajopt_sparse_cpu.json`, plus a normal-suite T=50,
one-update smoke test.

## Phase B implementation and verification order

1. Add `time_axis`, `TemporalPattern`, validation, and cached symbolic
   analysis; prove dense M2a behavior is unchanged.
2. Adapt velocity, acceleration, time-indexed, and reference-trajectory
   residuals to the named-block static-horizon protocol without changing their
   math, and add the explicitly specified `TrajectoryTangentEnvelopeResidual`
   used by the benchmark. Add temporal blocks, short-horizon constructor tests,
   zero-weight structure tests, and structure-versus-dense-complement tests.
3. Add lower-band assembly and operator parity, including robust row weights,
   masks, scales, arbitrary leading batch axes, fixed-base and free-flyer
   models.
4. Add batched banded Cholesky and `NormalCG`; compare with dense solves, test
   `NormalOperator` metadata and warm starts, and test each diagnostic failure
   beside successful batch siblings.
5. Rebase knot `solve_trajopt` on named blocks. Default to structured only when
   eligibility passes, expose forced dense, and record route/reason in the
   result.
6. Keep robot B-spline rejection and Schur deferral visible in docs/results.
7. Land the benchmark definition, then measurements. CPU evidence only.

Load-bearing tolerances follow the milestone order: structured blocks,
`Jv`, `J.Tu`, normal diagonal, `J.TJ`, and `J.Tr` versus dense at fp32
`rtol <= 1e-5`; banded/CG solve versus dense Cholesky at
`rtol <= 1e-4`; end-to-end trajectory cost parity at `1e-5` or q tangent
`rtol <= 1e-3`, with per-element statuses compared explicitly.

## Deferred items and known deviations

1. **Schur is not built.** No second production in-tree caller exists.
2. **Robot B-spline solve support remains rejected.** The required sparse
   support formula is designed, but the current component-space basis is not a
   manifold-safe parameterization. This deviates from T5.7's optimistic
   B-spline routing language and preserves M2c's correctness gate.
3. **Trajectory rebase is larger than the original prerequisite implies.** The
   live `solve_trajopt` and temporal residuals are still legacy surfaces; Phase
   B must migrate them before routing can work.
4. **Mixed declared/undeclared problems fall fully dense.** M5 v1 does not
   create a band-plus-dense container.
5. **GPU and CUDA graph claims remain M6-only.** The M5 benchmark is pinned
   CPU evidence.
6. **Mixed-precision factorization is not implemented.** Users select an
   fp64 problem when fp64 factorization is required.
7. **The Phase C raw joint-position-limit term is replaced.** On the required
   free-flyer-plus-spherical SMPL-like model, current `JointPositionLimit` has
   no tangent support and is incompatible with the named trajectory output
   shape. The benchmark instead uses the exact nonzero
   `TrajectoryTangentEnvelopeResidual` specified in §11. This is an explicit
   correctness-driven deviation from T5.2 item 11's shorthand "limits", not a
   silent omission or claim that quaternion components are box coordinates.

## Appendix A — reproducible encoding probe

Run from the repository root:

```bash
UV_CACHE_DIR=/tmp/betterrobot-uv-cache uv run python - <<'PY'
from dataclasses import dataclass
import gc, platform, statistics, time, tracemalloc
from typing import Any, Mapping
import torch
from better_robot.io.builders.smpl_like import make_smpl_like_model
from better_robot.optim.blocks import Problem, ResidualItem, RobotConfig, VarSpec

T = 500
KEYS = (0, 71, 142, 213, 285, 356, 427, 499)
REPEATS, WARMUPS = 31, 5
torch.set_num_threads(1)
torch.set_num_interop_threads(1)
model = make_smpl_like_model(dtype=torch.float32)
manifold = RobotConfig(model)
q_traj = model.q_neutral.expand(T, model.nq).clone()

@dataclass(frozen=True)
class PrototypeTimeAxis:
    axis: int
    length: int
    state_width: int
    tangent_width: int

@dataclass(frozen=True)
class Stub:
    name: str
    reads: tuple[str, ...]
    dim: int = 1
    def __call__(self, ctx: Mapping[str, Any]) -> torch.Tensor:
        raise AssertionError("construction-only probe")

def item(name, reads):
    return ResidualItem(name, Stub(name, reads))

def build_one():
    spec = VarSpec("q", (T, model.nq), manifold)
    residuals = [item("velocity", ("q",)), item("acceleration", ("q",))]
    residuals += [item(f"pose_{k}", ("q",)) for k in KEYS]
    return Problem(vars=(spec,), residuals=residuals), PrototypeTimeAxis(0, T, model.nq, model.nv)

def build_many():
    names = tuple(f"q_{t:03d}" for t in range(T))
    specs = tuple(VarSpec(name, (model.nq,), manifold) for name in names)
    residuals = []
    for t in range(1, T - 1):
        residuals.append(item(f"velocity_{t:03d}", (names[t-1], names[t+1])))
        residuals.append(item(f"acceleration_{t:03d}", (names[t-1], names[t], names[t+1])))
    residuals += [item(f"pose_{k:03d}", (names[k],)) for k in KEYS]
    return Problem(vars=specs, residuals=residuals)

one_problem, _ = build_one()
many_problem = build_many()
one_values = {"q": q_traj}
many_values = {f"q_{t:03d}": q_traj[t] for t in range(T)}

def measure(fn):
    for _ in range(WARMUPS): fn()
    samples = []
    gc.disable()
    try:
        for _ in range(REPEATS):
            start = time.perf_counter_ns()
            fn()
            samples.append((time.perf_counter_ns() - start) / 1e6)
    finally:
        gc.enable()
    q = statistics.quantiles(samples, n=4, method="inclusive")
    return statistics.median(samples), q[0], q[2]

def peak(fn):
    gc.collect()
    tracemalloc.start()
    value = fn()
    _ = value
    _, high = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return high / 1024

for label, fn in (
    ("build_one_ms", build_one),
    ("build_many_ms", build_many),
    ("validate_one_ms", lambda: one_problem._validate_values(one_values)),
    ("validate_many_ms", lambda: many_problem._validate_values(many_values)),
):
    median, q1, q3 = measure(fn)
    print(f"{label}: median={median:.6f} q1={q1:.6f} q3={q3:.6f}")
print(f"build_one_peak_kib={peak(build_one):.3f}")
print(f"build_many_peak_kib={peak(build_many):.3f}")
print("model", model.nq, model.nv, model.njoints)
print(platform.node(), platform.python_version(), torch.__version__)
PY
```

Observed output for the environment recorded above:

```text
build_one_ms: median=0.246017 q1=0.242800 q3=0.252337
build_many_ms: median=14.666206 q1=14.617071 q3=14.733711
validate_one_ms: median=1.386580 q1=1.379007 q3=1.395145
validate_many_ms: median=282.688571 q1=258.584780 q3=446.910672
build_one_peak_kib=12.063
build_many_peak_kib=1041.637
model 99 75 25
robotics2-ESC8000-E11 3.11.15 2.13.0+cu126
```
