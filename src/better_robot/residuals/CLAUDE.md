# `residuals/` — Object-Referenced Error Terms

## Residual protocol

A residual subclasses `Residual`, receives every variable or shared node it
reads at construction, and returns raw `(..., dim)` rows from `error()`.
`Residual` owns the stable name, fixed positive dimension, weight, robust
kernel, group size, and ordered variable references. Missing analytic blocks
use the `Problem` Jacobian strategy; finite differences remain an explicit
debug choice.

`jacobian()` returns a tuple of complete reduced-tangent blocks ordered like
the residual's trainable variable dependencies. A residual with shared work
lists its `Node` objects in `nodes`. Nodes own their input variables and cache
graph-bearing results for one evaluation epoch only. `RobotState` is the
standard lazy FK node; equivalent nodes over the same robot variable merge
when the problem freezes.

## Temporal structure

`TemporalPattern` is the optimizer-independent declaration for a residual over
one `Variable(..., time_axis=0)`. For row group `r`, offset `o` identifies knot
`r + row_origin + o`; `rows * row_width` must equal the residual's static
`dim`. Offsets are sorted, unique, non-empty integers and every referenced
knot must lie inside the declared horizon.

A structured residual implements both optional hooks:

- `temporal_structure(variable) -> TemporalPattern | None` declares support
  without inspecting tensor values;
- `temporal_jacobian_blocks(variable)` returns `offset -> Tensor` with shape
  `(B..., rows, row_width, reduced_width_per_knot)`.

Numeric blocks contain only the raw residual derivative. The residual's
`weight` and robust row scaling are applied once by `Problem`. A declaration
without numeric blocks is not direct-banded eligible. Missing declarations
make automatic LM use dense assembly; forced structured routing fails
actionably. A zero weight never grants eligibility.

## Vision and point-cloud pack

- `projection.py`: camera-thin `ProjectionResidual` over model frame-table
  rows. The complete analytic block is projection derivative × camera rotation
  × frame Jacobian. Observation tensors must already match the evaluated model
  dtype/device; no hidden hot-path transfers.
- `chamfer.py`: `MaskedChamferResidual` on fixed padded point clouds and bool
  masks. Nearest indices are detached while distances remain differentiable.
- `scene_sdf.py`: `SceneSDFState` performs one detached nearest-neighbour pass
  and feeds penetration, attraction, and clearance residuals. The shared result
  carries signed distance, nearest distance, confidence, and validity.
- `_point_cloud.py`: private chunked nearest-correspondence helpers. The public
  ragged convention is `(padded_tensor, validity_mask)`; invalid rows are finite
  zeros.

Robust kernels live on each `Residual`. Use `group_size=2` for per-point
projection kernels such as Geman–McClure.

## Existing library

- `pose.py`: pose, position, and orientation targets using `RobotState`.
- `limits.py`, `human.py`: joint and swing/twist limits.
- `regularization.py`: rest, spherical-joint prior, nullspace placeholder, and
  reference-trajectory terms.
- `smoothness.py`, `temporal.py`: trajectory differences and time indexing.
- `contact.py`: contact-consistency residual.
- Collision residuals are not exported; the separate collision package remains
  owner-gated and must not be partially implemented without reviewed scope.

## Author checklist

1. Pass every dependency at construction and give the residual a stable name
   and fixed dimension.
2. Preserve arbitrary leading execution batches and return raw residual rows.
3. Require compatible input dtype/device instead of coercing during an
   evaluation.
4. Make masks fixed-shape and explicit; never infer structure from tensor
   values.
5. If analytic blocks are advertised, provide one for every trainable
   dependency and test them against both AD directions.
6. Record detached choices such as nearest-neighbour indices in the docstring
   and test that gradients reach only selected continuous values.
7. Put shared FK, dynamics, or NN work in a `Node` and add a counting test for
   one computation per evaluation epoch.
8. For time-local support, test dense blocks, JVP, VJP, normal bands, arbitrary
   leading batches, and short-horizon constructor failures. Never infer
   support from numerical zeros.
