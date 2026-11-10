# lie/ traps

Each module opens with its conventions; the layouts other repositories read are in
`docs/contracts/coupling-standard.md` (quaternions `[qx, qy, qz, qw]`, scalar last; tangents linear first).

- A small-angle branch needs a safe input in the full formula too: `torch.where` back-propagates through both
  branches, and an infinite derivative times a zero weight is nan.
  ```python
  # Right (so3.exp_raw): the full branch sees theta² = 1 inside the Taylor region.
  theta = torch.where(use_taylor, torch.ones_like(theta2), theta2).sqrt()
  # Wrong: the gradient of exp at w = 0 is nan.
  theta = theta2.sqrt()
  ```
  The cutoffs are on `theta²` and depend on the dtype (`_taylor_theta2`); fp64 `gradcheck` and `gradgradcheck` at
  and near identity guard the public exp, log and right Jacobians (`tests/lie/test_torch_backend_gradcheck.py`).
- Every public function checks its tensor inputs with `_checks.py` (the library's one validator) and, when an
  in-library caller needs it, calls an unchecked `<name>_raw` twin. `kinematics`, `dynamics`, `spatial`, `model`
  and `io` call the `_raw` twins on inputs their own boundary checked. A new public function goes in the module's
  `__all__` and in `tests/lie/test_boundary.py`.
- `lie` and `spatial` are held at 100 % line and branch coverage by `tests/lie` and `tests/spatial` (the release
  tier's coverage step): a new branch comes with the test that reaches it.
