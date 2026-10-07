# Conventions

These rules keep the library predictable: the same names, shapes and failure modes everywhere. Read
them before you add a public function. Contract tests enforce several of them (see
[Testing](testing.md)).

## Names

| Kind | Rule | Examples |
|------|------|----------|
| Published algorithm | keep the lowercase acronym | `rnea`, `aba`, `crba`, `ccrba` |
| Compute | form a result from inputs | `compute_centroidal_map` |
| User task | a plain action phrase | `forward_kinematics_state`, `inverse_dynamics`, `solve_ik` |
| Raw tensor pass | the public name plus `_raw` | `rnea_raw`, `frame_jacobian_raw` |
| Boolean | starts with `is_`, `has_` or `should_` | |

Short mathematical names stay short where the field uses them: `q`, `v`, `a`, `tau`, `nq`, `nv`,
`njoints`. Spell everything else out. Old Pinocchio-style storage names (`oMi`, `liMi`, ...) are
banned by `tests/contract/test_naming.py`. Files in an implementation family read
`category_specific.py`; public modules use the plain name a user expects.

## Shapes and batch axes

- Event axes are trailing; any number of leading batch axes are allowed: `q (B..., nq)`,
  `joint_transform (B..., njoints, 3, 4)`, frame Jacobian `(B..., 6, nv)`.
- Leading axes follow PyTorch broadcasting. Never squeeze or add a batch axis for the caller.
- Pose: `[tx, ty, tz, qx, qy, qz, qw]`, quaternion scalar last. Twist: `[linear(3), angular(3)]`.
- `nq` and `nv` can differ. Move on the configuration manifold with `model.integrate` and
  `model.difference`, never with `q + v`.
- Frame Jacobians default to `LOCAL_WORLD_ALIGNED`.
- Write shapes in annotations and docstrings as ``q (B..., nq)``.

## Validate at public boundaries only

Public functions check shape, dtype, device and that the inputs belong to the model, once, and raise
the typed errors from `better_robot.exceptions` (`ShapeError`, `DtypeMismatchError`,
`DeviceMismatchError`, ...). Raw passes below them trust their inputs and do not re-check
(`tests/contract/test_boundary_validation_count.py`). Do not scan tensor values for finiteness as
general input validation; that forces a host sync.

## Preserve dtype and device

Outputs take the dtype and device of the model and inputs. Never silently cast, call `.to(...)`, or
move tensors between devices; a mismatch raises. Do not call `.item()`, `.cpu()` or Python `if` on
tensor values in kinematics and dynamics (`tests/contract/test_hot_path_lint.py`).

## No process-wide torch settings

Library code never changes global state: no `torch.set_default_dtype`, `torch.set_num_threads`,
`torch.manual_seed`, or anomaly-mode switches. Scripts and benchmarks may set them for themselves.

## Code style

- `ruff` with `line-length = 120`, target Python 3.10, rules `E`, `F`, `PLC`, `PLE`, `PLR`, `PLW`
  (ignoring `E501`, `PLR0913`, `PLR2004`). Run `uv run ruff check src/ tests/`.
- Type hints on every public signature. Lie and spatial operations stay direct torch formulas.
- Comments explain why, not what. Optional dependencies are imported inside the function that needs
  them.
- Add an abstraction only when at least two in-tree callers need it.

## Docstrings

Every public symbol has a docstring (`tests/contract/test_docstrings.py`). The API reference is
generated from them by mkdocstrings.

- The first line sits on the opening `"""`: one sentence, imperative or noun phrase, ending with a
  period. Then a blank line and short paragraphs.
- Shapes inline in double backticks: ``q (B, nq)``, ``(..., J, 3, 4)``; say which axes broadcast.
- State contracts as facts: dtype and device preserved, pose order, the reference frame by name
  (``LOCAL``, ``WORLD``, ``LOCAL_WORLD_ALIGNED``).
- Cross-reference with plain backticked names (``KinematicState``); no Sphinx roles.
- An example, when the call is not obvious, is a fenced `python` block of at most 8 lines that runs.

```python no-run
def frame_jacobian(model: Model, state: KinematicState, frame: int | str, *, reference: str = "local_world_aligned") -> torch.Tensor:
    """Jacobian of one frame at the configuration of ``state``. Returns ``(B..., 6, nv)``.

    Rows are ``[linear, angular]``; ``reference`` names the frame they are expressed in.
    """
```
