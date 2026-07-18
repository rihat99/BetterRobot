# residuals/ — Residual Functions and Named-Block Components

## Residual protocol

A residual is a callable over an evaluation-local mapping, declares `reads`,
returns `(..., dim)` rows, and may implement
`jacobian_blocks(ctx) -> dict[var_name, Tensor]`. Missing analytic blocks use
the `Problem` Jacobian strategy; finite differences remain an explicit debug
choice.

Providers own shared computation. Declare their static `reads` and `outputs`;
`Problem` evaluates each provider at most once per evaluation context. Residuals
must not cache graph-carrying tensors across evaluations.

## Temporal structure

`TemporalPattern` is the optimizer-independent declaration for a residual over
one `VarSpec(..., time_axis=0)`. For row group `r`, an offset `o` identifies
knot `r + row_origin + o`; `rows * row_width` must equal the residual's static
`dim`. Offsets are sorted, unique, non-empty integers and every referenced
knot must lie inside the declared horizon.

A structured residual implements both optional hooks:

- `temporal_structure(variable_name) -> TemporalPattern | None` declares
  support without inspecting tensor values;
- `temporal_jacobian_blocks(ctx, variable_name)` returns `offset -> Tensor`
  with shape `(B..., rows, row_width, reduced_width_per_knot)`.

Numeric blocks contain only the residual derivative. `ResidualItem.weight`
and robust row scaling are applied once by `Problem`. A declaration without
numeric blocks is operator-eligible but not direct-banded eligible. Missing
declarations cause automatic LM to use dense assembly; forced structured
routing fails actionably. A zero item weight never grants eligibility.

## Vision and point-cloud pack

- `projection.py`: camera-thin `ProjectionResidual` over model frame-table
  rows. The complete analytic block is projection derivative × camera rotation
  × frame Jacobian. Observation tensors must already match the evaluated model
  dtype/device; no hidden hot-path transfers.
- `chamfer.py`: `MaskedChamferResidual` on fixed padded point clouds and bool
  masks. Nearest indices are detached while distances remain differentiable.
- `scene_sdf.py`: `SceneSDFProvider` performs one detached nearest-neighbour
  pass and feeds penetration, attraction, and clearance heads. The shared
  result carries signed distance, nearest distance, confidence, and validity.
- `_point_cloud.py`: private chunked nearest-correspondence helpers. The public
  ragged convention is `(padded_tensor, validity_mask)`; invalid rows are finite
  zeros.

Robust kernels live on `optim.ResidualItem`, not inside residual math. Use
`group_size=2` for per-point projection kernels such as Geman–McClure.

## Existing library

- `pose.py`: pose, position, and orientation targets.
- `limits.py`, `human.py`: joint and swing/twist limits.
- `regularization.py`: rest, spherical-joint prior, nullspace placeholder, and
  reference-trajectory terms.
- `smoothness.py`, `temporal.py`: trajectory differences and time indexing.
- `contact.py`: contact-consistency residual.
- Collision residuals are not exported; the separate collision package remains
  owner-gated and must not be partially implemented without reviewed scope.

## Author checklist

1. Give every residual a stable non-empty `name`, fixed `dim`, and precise
   `reads` declaration.
2. Preserve arbitrary leading execution batches and return raw residual rows.
3. Require compatible input dtype/device instead of coercing tensors during an
   evaluation.
4. Make masks fixed-shape and explicit; never infer structure from tensor
   values.
5. If an analytic block is advertised, it must be complete for every variable
   read by the residual and tested against both AD directions.
6. Record detached choices such as nearest-neighbour indices in the docstring
   and test that gradients reach only the selected continuous values.
7. Put shared FK, dynamics, or NN work in a provider and add a counting test.
8. For time-local support, declare `TemporalPattern` and test dense blocks,
   JVP, VJP, normal bands, arbitrary leading batches, and short-horizon
   constructor failures. Never infer support from numerical zeros.
