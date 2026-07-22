# How-to guides

Use these recipes when you know the outcome you want and need the shortest
supported path to it. The getting-started tutorials introduce the vocabulary;
the concept chapters explain the design choices behind these APIs.

```{toctree}
:maxdepth: 1

custom_residual
load_a_robot
visualize
differentiate_through_kinematics
own_your_optimization_loop
staged_fit
```

## Available guides

| Guide | Use it when |
|-------|-------------|
| {doc}`custom_residual` | Add a new least-squares error term. |
| {doc}`load_a_robot` | Load URDF or MJCF, or construct a model in Python. |
| {doc}`visualize` | Inspect a pose or trajectory in the browser viewer. |
| {doc}`differentiate_through_kinematics` | Compute gradients or a representation Jacobian with PyTorch. |
| {doc}`own_your_optimization_loop` | Step LM yourself or reuse its damping state. |
| {doc}`staged_fit` | Preserve Adam state while changing a fit's observations and active terms. |
