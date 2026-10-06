# Study: can optim2 express the BVR person fit? (mars_aux, read-only, 2026-10-07)

Code reading only; nothing was run. kindyn term bodies and the force parametrization were not opened.
BVR = /home/rikhat.akizhanov/better/BetterVideoReconstruction.

## Where the fit lives
- Engine: `tools/optim.py` (`Problem.add_leaf/add_term`, `Phase`, `run_phases`, :58-193).
- Fit: `tools/human_optim/optimizer.py:371` `optimize_person(...)`; leaves :724-738, terms :800-948, `run_phases` :1019.
- Callers: `tools/human_optim/stages.py:421` `run_kinematic`, `:785` `run_kinematics_2`, `kindyn.py:699` `run_kindyn`,
  `prefit.py:202` `prefit_person_round`.

## Variables (one problem per person, T <= 500)
| name | shape | parametrization | shared over time |
|---|---|---|---|
| `q` | (T,211) | raw: transl, root quat, 51 joint quats; normalized in the forward | no |
| `betas` (optional) | (10,) | Euclidean | yes |
| `f_free` (kindyn) | (T,12,3) | world force per contact frame | no |
| `gen_u` (kindyn) | (T,12,4) | pyramid-edge activations | no |
| prefit `log_s` | (1,) | log scale | yes |

## Terms (each returns one scalar = weight x mean; weight columns root/kin1/kin2/kindyn)
- `reproj_sapiens` (Geman-McClure of px^2, per-keypoint score weights, normalized by total score sum, `losses.py:38-77`).
- `reproj_init`, `pose_prior`, `orient_prior` (anchored to frozen `q_init`).
- `chamfer`: visibility mask and nearest neighbours recomputed under no_grad each evaluation (`optimizer.py:682-685`).
- `velocity/accel/jerk/snap` root and joint: finite differences of order 1-4 (windows 2-5), rows masked.
- `radial_velocity`, `penetration` (scene candidates rebuilt at phase start), `ann_*` contact terms (labels made in `on_start`).
- kindyn: base wrench, force magnitude/edge/smoothness (2-4 frame windows), torque terms.

## Phases
- root (Adam 150, root DOFs) -> full (Adam 300) -> refine (Adam 200, scene on) -> lbfgs (30).
- kinematics_2: 100 / 200 / 250, no lbfgs. kindyn: force warm start (projected Adam), then L-BFGS 150.
- Adam lr 0.01 with MultiStepLR x0.1 at 1/2 and 3/4 of a phase; torch LBFGS strong-Wolfe in one `step(closure)`.
- No stopping rules besides iteration budgets.

## Gaps against optim2 (as of commit 65b63bc)
1. Learning-rate decay inside a phase keeping Adam moments: `TorchOptim` has no scheduler hook.
2. Shared `betas` next to per-frame `q` lost the banded path (dense would be ~105k x 105k).
3. Normalizations that depend on x (chamfer visible count, score sum): scalar terms only, not LM residuals.
4. Phase-start state read by terms: `on_start` can only return x; needs writing to `problem.data`.
5. Free mask on a time variable must be (T*211) long; BVR's (211,) mask broadcasts over T.
6. L-BFGS: BVR uses strong-Wolfe; optim2 `LBFGS` is Armijo backtracking. Exact match via `TorchOptim(torch.optim.LBFGS)`.
7. No per-term diagnostics or log hook.
8. Force warm start: expressible as a first phase.
9. Non-differentiable parts per evaluation work under autograd; an analytic `jac` must treat them as constants.
- Maps cleanly: no coupling between people; no term couples frames further than 5 apart.

## Size
T <= 500, 211 unknowns per frame (+84 in kindyn) + 10 shared; up to ~148k unknowns. Runtime not found in docs.
