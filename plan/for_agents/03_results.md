# 03 results — Documentation style and examples

**Status:** complete.

## Delivered

| Task | Delivered disposition |
|---|---|
| T1 — visible snippet results | Converted every listed tutorial, guide, landing-page, Lie/spatial, and extension snippet from assertions or silent execution to deterministic `print(...)` plus `{testoutput}`. Preserved the custom-residual extraction markers and the explanatory Lie `assert_close` prose. Added an AST-backed documentation contract that rejects bare `assert` and `torch.testing.assert_close` inside published Python fences. |
| T2 — executable concepts | Added working RNEA/CRBA examples to dynamics, FK/frame-Jacobian examples to kinematics, optimizer lifecycle/TorchOptimizer/implicit examples to residuals and solvers, and raw FK/RNEA plus opt-in Warp-selection examples to the compute seam. |
| T3 — console examples | Added batched IK, differentiable IK, standing contact-force, and RNEA/ABA examples as `03`, `06`, `07`, and `08`. Each new script is 46–76 lines, needs no viewer, prints its result, and has a headless smoke test. The four existing viewer examples retain their behavior and were also run with `--no-viewer`. |
| T4 — changelog honesty | Redirected every changelog migration link to root `MIGRATION.md`, marked the v0.2.0 public list as historical, and repaired three stale Unreleased bullets that still described the pre-v2 optimizer API. Seven current `MIGRATION.md` links resolve to the root file after Parts 1/2 added two references beyond the five originally audited. |

## Per-page disposition

| Page | Result |
|---|---|
| `getting_started/installation.md` | Installation result and model dimensions are printed. |
| `getting_started/01_robot_model.md` | Configuration shape, model counts, and frame name are printed. |
| `getting_started/02_forward_kinematics.md` | Joint, frame, and hand-pose shapes are printed. |
| `getting_started/03_inverse_kinematics.md` | Convergence and solution shapes are printed. |
| `getting_started/04_floating_base.md` | Storage/tangent sizes and pose shapes are printed. |
| `getting_started/05_batched_gpu.md` | Batch shapes and device-type agreement are printed deterministically. |
| `guides/custom_residual.md` | Residual rows are printed; exact extraction markers remain intact. |
| `guides/differentiate_through_kinematics.md` | Jacobian shape and finiteness are printed. |
| `guides/load_a_robot.md` | Loaded dtype/shape and builder result are printed. |
| `guides/own_your_optimization_loop.md` | Initial and warm-updated solutions plus convergence are printed. |
| `guides/visualize.md` | Viewer state shape is printed without opening the viewer in the snippet. |
| `docs/index.md` | Landing-page convergence and shapes are printed; its extractor accepts `{testcode}`. |
| `concepts/lie_and_spatial.md` | SE(3), tangent-Jacobian, and Umeyama outputs are shown. |
| `conventions/extension.md` | Custom residual and dense-solver outputs are shown. |
| `concepts/dynamics.md` | RNEA gravity torque and CRBA matrix shape are executable. |
| `concepts/kinematics_and_jacobians.md` | FK shapes and a `(6, nv)` frame Jacobian are executable. |
| `concepts/residuals_costs_and_solvers.md` | Six snippets cover residuals, fitting, step/solve lifecycle, TorchOptimizer, and implicit differentiation. |
| `concepts/the_compute_seam.md` | Raw FK/RNEA results and explicit Warp-or-reference selection are executable. |
| `docs/CHANGELOG.md` | Current links and API-era qualifications replace misleading text. |

## Example evidence

| Example | Headless result |
|---|---|
| `01_basic_ik.py` | Converged with `0.0000 m` reported position error. |
| `02_g1_ik.py` | Converged at its reachable initial target. |
| `03_batched_ik.py` | Default CPU run converged 1,000/1,000 problems; CUDA is selected automatically when available. |
| `04_smpl_like_body.py` | Built the 25-joint model and completed FK without a viewer. |
| `05_panda_trajopt.py` | Converged in 9 iterations with sub-nanometre endpoint errors in the verification run. |
| `06_differentiable_ik.py` | Converged and produced a finite, nonzero seven-component target-pose gradient. |
| `07_contact_forces.py` | Recovered three `19.62 N` standing forces and printed zero joint-origin moments. |
| `08_dynamics.py` | Printed nonzero gravity compensation torque and zero RNEA-to-ABA round-trip error. |

## Verification

| Gate | Final result |
|---|---|
| Full repository suite (`tests/`, excluding CUDA and benchmark markers) | **1,583 passed, 2 skipped, 16 deselected**, 38 warnings. |
| Documentation and example focus | **31 passed**, 18 existing Torch JIT deprecation warnings. |
| Sphinx doctest | **30 tests, 0 failures**; four offline intersphinx DNS warnings. |
| Sphinx HTML | Fresh **158-document build succeeded**; the same four offline inventory warnings. |
| Snippet inventory | **30 `{testcode}` / 30 `{testoutput}`** blocks. |
| Snippet assertion contract | Zero bare assertions or `torch.testing.assert_close` calls in published Python fences. |
| New example smoke tests | **4 passed**; all eight example scripts were also run directly headlessly. |
| Changed Python style | Ruff check passed; all eight changed/new Python files are formatted. |
| Patch hygiene | `git diff --check` passed. |

## Deviations and environment limits

1. **Expanded T4 honesty fix.** The plan requested only the v0.2.0 historical
   note and migration-link repair. The same audit found three contradictory
   Unreleased bullets still advertising `VarSpec`, `ResidualItem`, and
   `run_first_order`; they were corrected before publishing rather than
   knowingly shipping false current documentation.
2. **Strict Sphinx is a pre-existing red gate.** The documented nitpicky
   `-W` build reports 1,145 unresolved cross-reference warnings, dominated by
   generated API annotations and unreachable external inventories. The normal
   fresh HTML build and all doctests succeed. This task did not suppress or
   mass-ignore that repository-wide debt; the successful normal HTML artifact
   is used for Pages publication.
3. **CUDA was unavailable.** The batched example contains synchronized CUDA
   timing and automatic device selection, but only its CPU path could be run
   here. No CUDA result is claimed.
