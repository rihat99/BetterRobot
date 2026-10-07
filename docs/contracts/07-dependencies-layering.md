# 07 Dependencies and layering

The packages under `src/better_robot/` form a directed acyclic graph (DAG): an import may only point *down*.
Packages at the same rank do not import each other. The graph is the architecture; a new package is placed
in it by an ADR (`00-index`, "When to write an ADR").

```text
viewer -> terms -> optim (optim imports only torch; knows nothing about robots)
            |
io    dynamics   collision
 \       |          /
      kinematics
          |
   data_model -> spatial -> lie
```

Ranks in `tests/contract/test_layer_dependencies.py:23` (higher imports lower; equal rank may import each
other today): `_typing` 1, `lie` 2, `spatial` 3, `optim` 3, `data_model` 4, `io` / `kinematics` / `dynamics` /
`collision` 5, `terms` 6, `viewer` 7. Splitting `dynamics` above `kinematics` is `[planned: L-3]`.

## C-DEP-1 Imports follow the DAG; no cycles, no same-rank imports (MUST)

`lie` imports torch only. `spatial` imports `lie`. `data_model` imports `spatial`. `kinematics` imports
`data_model` and below. `dynamics`, `collision`, `io` import `kinematics` and below (`dynamics` builds its
world-frame Jacobians on a kinematics helper, `dynamics/_world.py:30`; that helper is private today and
becomes a public-internal `kinematics.jacobian_columns` `[planned: L-3]`). `optim` imports only torch and
the standard library. `terms` imports `optim` and the robot packages. `viewer` imports anything.
`exceptions`, `_typing`, `_validation` are rank zero and import nothing from the package.

```python { .no-run }
# good  (illustrative; see terms/robot.py)
from better_robot.kinematics import frame_jacobian
# bad  (illustrative)
from better_robot.data_model import Model          # optim must not know robots
```

Enforcement: `tests/contract/test_layer_dependencies.py` (ast walk of every module; edges checked against the
`LAYER_RANK` table in the test).

## C-DEP-2 Optional dependencies are imported lazily at their boundary (MUST)

`viser`, `trimesh`, `yourdfpy`, `pinocchio` and any other extra are imported inside the function or module that
needs them, wrapped so the first use raises `ImportError` with the install hint (`06`). Importing
`better_robot` or any core package must succeed with none of the extras installed.

```python { .no-run }
# good  (illustrative)
def _viser():
    try:
        import viser
    except ImportError as exc:
        raise ImportError("install the viewer extra: uv add better-robot[viewer]") from exc
    return viser
# bad  (top of a core module)
import trimesh
```

Enforcement: `tests/contract/test_optional_imports.py` (subprocess with `sys.modules[name] = None` for each
extra, then `import better_robot` and the core packages); `tests/contract/test_optional_imports.py`;
Lazy imports inside functions trip `[ruff:PLC0415]`; each intended one carries `# noqa: PLC0415`.

## C-DEP-3 No process-wide torch settings (MUST)

The library never calls `torch.set_default_dtype`, `torch.set_default_device`, `torch.set_grad_enabled`,
`torch.use_deterministic_algorithms`, `torch.manual_seed` or changes thread counts. Those are the user's.

Enforcement: `[ruff:TID251]` (banned-api table in `pyproject.toml`).

## C-DEP-4 Adding a dependency needs an ADR and a manifest entry (MUST)

A new runtime dependency goes through an ADR and into `pyproject.toml` (`dependencies` for required,
`[project.optional-dependencies]` for an extra). No `pip install` in scripts, no vendored copies.

Enforcement: `[review]`.
