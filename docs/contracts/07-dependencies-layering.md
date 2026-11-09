# 07 Dependencies and layering

The packages under `src/better_robot/` import each other along one edge table, the `ALLOWED` table in
`tests/contract/test_layer_dependencies.py`: one row per package, naming the packages it may import. The table has no
cycle, and a new package gets its row through an ADR (`00-index`, "When to write an ADR"). The root
`src/better_robot/__init__.py` re-exports the packages and has no row.

| Package | May import |
|---|---|
| `exceptions`, `_version` | nothing from the package |
| `lie` | `exceptions` |
| `spatial` | `exceptions`, `lie` |
| `model` | `exceptions`, `lie`, `spatial` |
| `kinematics` | `exceptions`, `lie`, `model` |
| `dynamics` | `exceptions`, `lie`, `spatial`, `model`, `kinematics` |
| `collision` | `exceptions`, `lie`, `model`, `kinematics` |
| `io` | `exceptions`, `lie`, `model` |
| `optim` | `exceptions` |
| `terms` | `exceptions`, `lie`, `model`, `kinematics`, `collision`, `optim` |
| `viewer` | `exceptions`, `lie`, `model`, `kinematics`, `collision` |
| `testing` | `exceptions`, `model` |

## C-DEP-1 Imports follow the edge table, which has no cycle (MUST)

A package imports only the packages its row names. No row names `viewer`, `terms` or `testing`: no other package
depends on them. `dynamics` and `collision` import `kinematics`: `dynamics` reads the state's world column table
and geometry (`KinematicState.jacobian_columns`, `KinematicState.world_geometry`, record 0041) and the Jacobian
plan of `src/better_robot/kinematics/_plans.py`. Outside the package, `lie`, `spatial` and `optim` import torch and
nothing else: `optim` knows nothing about robots, and `terms` adapts robots to it.

```python { .no-run }
# good  (illustrative; see src/better_robot/terms/robot.py)
from better_robot.kinematics import frame_jacobian
# bad  (illustrative)
from better_robot.model import Model          # optim must not know robots
```

Enforcement: `tests/contract/test_layer_dependencies.py` (every import of every module, function-level and
`TYPE_CHECKING` ones included, checked against the `ALLOWED` table through
`src/better_robot/testing/import_edges.py::imported_packages`); `tests/contract/test_optional_imports.py` (`lie`,
`spatial` and `optim` import no outside package but torch).

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

No compute-backend registry exists today; an optional compiled backend would sit behind the same public
signatures and report its missing package with `better_robot.exceptions.OptionalDependencyError`.

Enforcement: `tests/contract/test_optional_imports.py` (every outside import of every unit, through the same
walker, against `BOUNDARY`: each package `pyproject.toml` declares is allowed in the units its `IMPORTERS` row names,
an extra's packages in one unit, an undeclared package nowhere). Lazy imports inside functions trip
`[ruff:PLC0415]`; each intended one carries `# noqa: PLC0415`.

## C-DEP-3 No process-wide torch settings (MUST)

The library never calls `torch.set_default_dtype`, `torch.set_default_device`, `torch.set_grad_enabled`,
`torch.use_deterministic_algorithms`, `torch.manual_seed` or changes thread counts. Those are the user's.

Enforcement: `[ruff:TID251]` (banned-api table in `pyproject.toml`).

## C-DEP-4 Adding a dependency needs an ADR and a manifest entry (MUST)

A new runtime dependency goes through an ADR and into `pyproject.toml` (`dependencies` for required,
`[project.optional-dependencies]` for an extra). No `pip install` in scripts, no vendored copies.

Enforcement: `[review]`.
