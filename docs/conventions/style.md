# Coding style

BetterRobot source should be unsurprising to a reader who knows Python,
PyTorch, and the equations being implemented. Prefer a direct tensor formula
and a small public boundary over clever dispatch or hidden state.

Tooling decides formatting. This page records the design choices that a
formatter cannot decide.

## Principles

1. Write boring, readable code.
2. Validate public structure once; trust internal callers.
3. State tensor shapes, units, and coordinate frames.
4. Prefer composition and protocols to deep inheritance.
5. Keep optional integrations local to the feature that needs them.
6. Match an established convention before inventing another one.

## Formatting and checks

Ruff uses a 120-character line length, four-space indentation, double-quoted
strings, sorted imports, and trailing commas in multiline constructs.

```bash
uv run ruff format .
uv run ruff check .
uv run mypy
uv run pyright
```

The type checkers are configured as useful development checks, not strict
proofs. New ignore comments need a reason.

Install the repository hooks explicitly:

```bash
uv run pre-commit install
uv run pre-commit install --hook-type pre-push
```

## Python names

| Kind | Form | Example |
|---|---|---|
| module | `lower_snake_case` | `joint_models` |
| class or exception | `CapWords` | `IKResult`, `ShapeError` |
| function or variable | `lower_snake_case` | `forward_kinematics` |
| constant | `UPPER_SNAKE_CASE` | `MAX_ITERATIONS` |
| type variable | short `CapWords` | `T`, `StateT` |
| internal name | one leading underscore | `_validate_q` |

Public storage names follow `<entity>_<quantity>_<frame>`. Standard robotics
symbols such as `q`, `v`, `tau`, `rnea`, and `SE3` remain short.
See {doc}`naming`.

Use `is_`, `has_`, or `should_` for booleans. Avoid unit suffixes:
public quantities use SI units, and docstrings state those units.

## Tensor and pose conventions

| Value | Layout |
|---|---|
| pose | `(..., 7) [tx, ty, tz, qx, qy, qz, qw]` |
| quaternion | `(..., 4) [qx, qy, qz, qw]` |
| SE(3) tangent | `(..., 6) [vx, vy, vz, wx, wy, wz]` |
| spatial Jacobian | `(..., 6, nv)`, linear rows before angular rows |

Angles are radians. Length, time, mass, and force use SI units. Convert at an
application boundary, not inside the algorithms.

## Imports

- Let Ruff group standard-library, third-party, package, and local imports.
- Do not use wildcard imports.
- Use a short relative import inside one subpackage when it improves clarity;
  otherwise use the package path.
- Import optional runtimes such as MuJoCo, viser, or Warp inside the owning
  function or behind `TYPE_CHECKING`.
- Avoid module-level work beyond defining immutable names.

Importing `better_robot` must not initialize a viewer, CUDA runtime, or
network resource.

## Types and protocols

Public signatures carry ordinary Python annotations. Tensor event shapes also
appear in docstrings because a plain `torch.Tensor` annotation cannot express
the runtime contract by itself.

Use modern syntax supported by Python 3.10:

- `list[int]` and `dict[str, float]`;
- `X | None`;
- `Protocol` for structural interfaces; and
- `Self` for fluent methods that return the same type.

Use an abstract base class only when implementations genuinely share behavior.
Do not add Pydantic or a metaclass for ordinary configuration.

## Docstrings

Public docstrings use NumPy style:

1. one-line summary;
2. `Parameters`;
3. `Returns`;
4. `Raises` for public failure modes;
5. notes only when the behavior needs more explanation; and
6. a small example when it clarifies normal use.

For every tensor, state:

- shape, including leading `B...` axes;
- dtype restrictions when relevant;
- units; and
- the frame for poses, twists, wrenches, velocities, and Jacobians.

Use double backticks for code names in docstrings. Keep examples short enough
to run in a focused test.

## Dataclasses and mutability

Match the ownership contract:

- `Model` and structural specifications are shallowly frozen;
- `Data` is mutable and local to one evaluation;
- solver updates return fresh state values;
- task configuration and result classes follow their current public
  dataclass behavior.

Contained tensor storage can still be mutable even when the dataclass is
frozen. Do not describe that object as deeply immutable.

Avoid mutable default arguments. Use `None` or a `default_factory`.

## Errors

Use the most specific exception already defined in
{doc}`contracts`. `TypeError` and `ValueError` remain appropriate for
boundaries without a dedicated BetterRobot exception.

An error message names the field, the rule, and the received value:

```text
q.shape must end in (8,), got (2, 7)
```

Use assertions only for internal states that a valid public call cannot
produce. Never use a bare `except`. A narrow `except Exception` is
acceptable at an optional-runtime, parser, or cleanup boundary when it
immediately converts or annotates the failure.

Library code does not print unsolicited diagnostics. A module that needs
logging uses `logging.getLogger(__name__)` and leaves handler configuration
to the application.

## Numerical code

- Express the readable batched formula first, then profile it.
- Compare floating values with explicit `atol` and `rtol`.
- Compare rotations through a relative rotation or matrix, not quaternion
  component equality.
- Keep raw Lie, kinematics, and dynamics math functional.
- Treat explicit `Data` cache updates as the documented mutation boundary.
- Hoist static tensors out of topology loops.

The executable hot-path rules and watched files live in
`tests/contract/test_hot_path_lint.py`; {doc}`performance` explains why
they matter.

## Files and public exports

Give a module one clear subject. Avoid vague `utils.py` and `helpers.py`
files. Tests live under the matching `tests/<area>/` folder.

A package `__init__.py` re-exports documented public symbols and lists them
in `__all__`. Internal helpers start with an underscore and stay out of
generated API pages.

Avoid:

- mutable process-wide configuration;
- runtime monkey-patching outside the package;
- decorators that hide a public signature;
- `eval` or `exec`; and
- operator overloads whose physical meaning is ambiguous.

## Public-function checklist

- [ ] descriptive name and complete type annotation
- [ ] NumPy-style docstring
- [ ] shapes, units, and frames stated
- [ ] one structural validation at the public boundary
- [ ] focused success and failure tests
- [ ] gradient test when differentiability is promised
- [ ] correct `__all__` entry for a public symbol
- [ ] changelog and user documentation updated
- [ ] Ruff, relevant type checks, tests, and docs build pass
