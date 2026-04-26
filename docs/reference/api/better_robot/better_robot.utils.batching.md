# {py:mod}`better_robot.utils.batching`

```{py:module} better_robot.utils.batching
```

```{autodoc2-docstring} better_robot.utils.batching
:allowtitles:
```

## Module Contents

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`batch_shape <better_robot.utils.batching.batch_shape>`
  - ```{autodoc2-docstring} better_robot.utils.batching.batch_shape
    :summary:
    ```
* - {py:obj}`flatten_batch <better_robot.utils.batching.flatten_batch>`
  - ```{autodoc2-docstring} better_robot.utils.batching.flatten_batch
    :summary:
    ```
````

### API

````{py:function} batch_shape(t: torch.Tensor, feature_dims: int) -> tuple[int, ...]
:canonical: better_robot.utils.batching.batch_shape

```{autodoc2-docstring} better_robot.utils.batching.batch_shape
```
````

````{py:function} flatten_batch(t: torch.Tensor, feature_dims: int) -> tuple[torch.Tensor, tuple[int, ...]]
:canonical: better_robot.utils.batching.flatten_batch

```{autodoc2-docstring} better_robot.utils.batching.flatten_batch
```
````
