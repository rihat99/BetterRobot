# Order 03 results — composed mesh residuals

Status: complete on `dev` (2026-07-22).

## Delivered

- `SceneSDFState` and `MaskedChamferResidual` accept Node-, Variable-, or tensor-valued clouds, masks, and confidence. Named static inputs remain updatable through `Problem.update()`; bare tensors remain construction-time constants.
- Scene geometry supports signed point and point-to-plane distance. Penetration, attraction, and clearance own detached confidence, distance, mask, depth/band, and margin gates; inactive rows are exact zero and `mean_active` uses each head's authoritative active set.
- Scene and Chamfer use a gradient-safe square root of detached non-negative confidence, making L2 contributions linear in confidence with finite zero-confidence gradients.
- `PointProjectionResidual` reuses the existing camera convention for `(P, 3)` and trajectory `(T, P, 3)` events. `time_axis=None|0` makes event axes explicit under arbitrary execution batches. Confidence is a per-group outer coefficient, while visibility is detached activity and therefore absent from inactive diagnostic rows.
- A vertical integration test feeds one synthetic posed-body Node to penetration, attraction, Chamfer, and point projection without subclassing any shipped residual. Documentation, migration guidance, public exports, and generated API pages were updated.

## Verification

| Gate | Result |
|---|---|
| Full CPU, `pytest tests/ -q -m "not bench and not cuda"` | 1,634 passed, 2 skipped, 50 deselected |
| CUDA, GPU 2, `pytest tests/ -q -m cuda` | 49 passed, 1,637 deselected |
| Complete residual suite | 75 passed |
| Projection, integration, and public-import contract suite | 115 passed |
| Documentation content, snippets, and docstrings | 34 passed |
| Strict Sphinx HTML (`-W --keep-going -E`) | Passed |
| Sphinx doctest (`-W --keep-going -E`) | 30 passed, 0 failed |
| Ruff and Ruff format, changed Python files | Passed |
| `git diff --check` | Passed |

Source accounting by the plan's `wc -l` method: **22,724 → 22,947** lines. The source diff is 286 additions and 63 deletions, net **+223**, within the `+250` hard cap.

## Deviations and findings

- The plan required an explicit temporal discriminator but did not name it. The implementation uses `time_axis=None|0`, matching Variable semantics and preventing an unbatched `T` axis from becoming an execution batch.
- The optional analytic Jacobian for `PointProjectionResidual` was not added. Explicit `jacrev` and `jacfwd` coverage agree, preserving the warning policy without duplicating the existing camera derivatives.
- Chamfer's old L2 magnitude can be reproduced with `c_new=c_old**2`. Exact old Scene behavior generally cannot: its internal geometric confidence `g` means row equivalence would require the query-dependent value `c_new=g*c_old**2`. `MIGRATION.md` records this required semantic change rather than suggesting a false universal conversion.
- No BVR constants, thresholds, sign convention, or visibility model were imported. Gates are caller-configured, point distance remains the default with point-to-plane opt-in, and externally computed visibility enters only through mask inputs.
- Non-negative tensor confidence remains a documented domain rather than a per-evaluation value scan, preserving the project's no-host-synchronization rule on CUDA hot paths.

No other implementation deviations from Order 03 remained after the final adversarial audit.
