# {py:mod}`better_robot.optim.autograd`

```{py:module} better_robot.optim.autograd
```

```{autodoc2-docstring} better_robot.optim.autograd
:allowtitles:
```

## Module Contents

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`perturb_values <better_robot.optim.autograd.perturb_values>`
  - ```{autodoc2-docstring} better_robot.optim.autograd.perturb_values
    :summary:
    ```
* - {py:obj}`tangent_grad <better_robot.optim.autograd.tangent_grad>`
  - ```{autodoc2-docstring} better_robot.optim.autograd.tangent_grad
    :summary:
    ```
````

### API

````{py:function} perturb_values(specs: tuple[better_robot.optim.variables.VarSpec, ...], values: collections.abc.Mapping[str, torch.Tensor], deltas: collections.abc.Mapping[str, torch.Tensor]) -> better_robot.optim.variables.Values
:canonical: better_robot.optim.autograd.perturb_values

```{autodoc2-docstring} better_robot.optim.autograd.perturb_values
```
````

````{py:function} tangent_grad(f: collections.abc.Callable[[better_robot.optim.variables.Values], torch.Tensor], specs: tuple[better_robot.optim.variables.VarSpec, ...], values: collections.abc.Mapping[str, torch.Tensor], *, create_graph: bool = False, graph_inputs: collections.abc.Sequence[torch.Tensor] = ()) -> better_robot.optim.variables.Values
:canonical: better_robot.optim.autograd.tangent_grad

```{autodoc2-docstring} better_robot.optim.autograd.tangent_grad
```
````
