# Order 02 results — node composition and scalar costs

Status: complete on `dev` (2026-07-22).

## Delivered

- `Node` accepts direct child nodes while retaining an order-stable, deduplicated tuple of transitive leaf variables. `Problem` recursively scopes and merges the complete graph, detects named cycles, and evaluates a shared child once per epoch.
- Non-trainable `Variable` values may use bool and integer dtypes. They remain tangent-free and unretractable; trainable variables remain floating-only. Problems require one device for every value and one dtype for floating values.
- Constructor documentation distinguishes explicitly named, updatable static variables from raw tensors captured at construction time.
- `ScalarCost(fn, *reads, weight=..., name=...)` accepts variables and nodes and contributes exactly `weight * fn(...)` for its documented non-negative domain. Its safe square root has finite backward behavior at zero, and implicit differentiation rejects it actionably.
- Public exports, generated API reference, concepts, guides, contracts, extension notes, changelog, and focused tests were updated.

## Verification

| Gate | Result |
|---|---|
| Full CPU, `pytest tests/ -q -m "not bench and not cuda"` | 1,618 passed, 2 skipped, 50 deselected |
| CUDA, GPU 2, `pytest tests/ -q -m cuda` | 49 passed, 1,621 deselected |
| Focused node/static/scalar/problem/implicit suite | 62 passed |
| Documentation contracts and snippets | 26 passed |
| Strict Sphinx HTML (`-W --keep-going -E`) | Passed |
| Sphinx doctest (`-W --keep-going -E`) | 30 passed, 0 failed |
| Ruff and Ruff format, changed Python files | Passed |
| `git diff --check` | Passed |

Source accounting by the plan's `wc -l` method: **22,622 → 22,724** lines. The source diff is 174 additions and 72 deletions, net **+102**, within the `+150` hard cap.

## Deviations and findings

- `ScalarCost` is re-exported from `better_robot.optim` as well as the required `better_robot.residuals` surface, matching the package's existing convenience exports for residual primitives.
- The graph boundary requires actual `Node` instances rather than structural lookalikes, which follows the accepted public contract and gives recursive traversal a reliable type boundary.
- Repository-wide Ruff still has 112 pre-existing violations in unchanged files; every changed Python file passes. Optional nitpicky Sphinx mode still exposes the pre-existing unresolved-reference baseline, while the required strict HTML and doctest builds pass.

No other implementation deviations from Order 02 were found in the final audit.
