# Backends

BetterRobot's hot paths route through a `Backend` Protocol object. The
default is `torch_native` — pure PyTorch, no extra dependencies. A
future Warp backend (P11) will plug in via the same Protocol without
touching call sites.

## Why a backend Protocol?

Three reasons:

1. **Single dispatch surface.** SE(3) ops, FK, and dynamics all flow
   through one object. Swapping the backend swaps every kernel at once.
2. **Optional native deps.** Warp / CUDA-graph capture is opt-in.
   Importing `better_robot` never imports `warp-lang`.
3. **Compile compatibility.** `default_backend()` returns the same
   instance per process, so `torch.compile` keys on a stable identity.

## What's in a backend?

```python
from better_robot.backends import default_backend, Backend

b: Backend = default_backend()
b.lie         # SE3/SO3 ops (exp, log, compose, …)
b.kinematics  # forward_kinematics, jacobian assembly
b.dynamics    # rnea, aba, crba (P11 — torch-native today)
```

Each sub-Protocol (`LieOps`, `KinematicsOps`, `DynamicsOps`) is
runtime-checkable, so user code can query "does this backend implement
ABA?" without importing internal modules.

## Switching backends

Globally:

```python
from better_robot.backends import set_backend
set_backend("torch_native")   # the default
set_backend("warp")           # raises until P11 ships
```

Per call (no global mutation):

```python
from better_robot.backends import get_backend
backend = get_backend("torch_native")
data = br.forward_kinematics(model, q, backend=backend)
```

## CUDA graph capture

```python
from better_robot.backends import graph_capture

@graph_capture
def step(model, q):
    return br.forward_kinematics(model, q)
```

Today this is a no-op on `torch_native`; the decorator becomes a real
graph capture once the Warp backend lands. Writing user code against
the seam now means no source diff later.

```{seealso}
{doc}`/design/10_BATCHING_AND_BACKENDS` for the full Protocol
definition and the Warp bridge plan.
```
