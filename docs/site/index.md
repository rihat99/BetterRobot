# BetterRobot

PyTorch-native robot kinematics, IK, and trajectory optimisation. One
code path for fixed-base and floating-base robots; everything is
batched, differentiable, and built on plain PyTorch tensors.

```python
import better_robot as br

model = br.load("panda.urdf")
result = br.solve_ik(model, {"panda_hand": target_pose})
result.q             # (nq,) joint solution
result.frame_pose("panda_hand")
```

## Get started

::::{grid} 1 2 2 2
:gutter: 3

:::{grid-item-card} Tutorials
:link: tutorials/index
:link-type: doc
Step-by-step walkthroughs that get a working solution on the screen
in 10 minutes or less.
:::

:::{grid-item-card} How-to guides
:link: guides/index
:link-type: doc
Recipes for specific tasks once you already know what you're doing.
:::

:::{grid-item-card} Concepts
:link: concepts/index
:link-type: doc
Why BetterRobot is shaped the way it is — frozen `Model`, mutable
`Data`, single-pass FK, layered backends.
:::

:::{grid-item-card} Reference
:link: reference/index
:link-type: doc
Auto-generated API per public symbol.
:::
::::

## Status

The package is at v0.x — a public API freeze is in flight. See the
[changelog](reference/changelog) for the latest release notes.

```{toctree}
:hidden:
:maxdepth: 2

tutorials/index
guides/index
concepts/index
reference/index
```
