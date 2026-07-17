# residuals/ — Residual Functions and Named-Block Components

## Two supported call shapes

Legacy residuals accept `ResidualState(model, data, variables)` and may expose
an analytic `jacobian(state)`. Current optimization tasks use M2 named-block
`Problem`: a residual is a callable over an evaluation-local mapping, declares
`reads`, returns `(..., dim)` rows, and may implement
`jacobian_blocks(ctx) -> dict[var_name, Tensor]`. Do not introduce a third
protocol or silently fall back to finite differences.

Providers own shared computation. Declare their static `inputs` and `outputs`;
`Problem` evaluates each provider at most once per evaluation context. Residuals
must not cache graph-carrying tensors across evaluations.

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
- `collision.py`: collision residual placeholders pending the evidence-gated
  M4 port-or-cut decision; do not partially implement this path.

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
