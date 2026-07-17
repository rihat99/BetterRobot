# {py:mod}`better_robot.data_model.execution_batch`

```{py:module} better_robot.data_model.execution_batch
```

```{autodoc2-docstring} better_robot.data_model.execution_batch
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`ExecutionInput <better_robot.data_model.execution_batch.ExecutionInput>`
  - ```{autodoc2-docstring} better_robot.data_model.execution_batch.ExecutionInput
    :summary:
    ```
* - {py:obj}`ExecutionBatch <better_robot.data_model.execution_batch.ExecutionBatch>`
  - ```{autodoc2-docstring} better_robot.data_model.execution_batch.ExecutionBatch
    :summary:
    ```
````

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`flatten_execution_batch <better_robot.data_model.execution_batch.flatten_execution_batch>`
  - ```{autodoc2-docstring} better_robot.data_model.execution_batch.flatten_execution_batch
    :summary:
    ```
````

### API

`````{py:class} ExecutionInput
:canonical: better_robot.data_model.execution_batch.ExecutionInput

```{autodoc2-docstring} better_robot.data_model.execution_batch.ExecutionInput
```

````{py:attribute} tensor
:canonical: better_robot.data_model.execution_batch.ExecutionInput.tensor
:type: torch.Tensor
:value: >
   None

```{autodoc2-docstring} better_robot.data_model.execution_batch.ExecutionInput.tensor
```

````

````{py:attribute} batch_shape
:canonical: better_robot.data_model.execution_batch.ExecutionInput.batch_shape
:type: tuple[int, ...]
:value: >
   None

```{autodoc2-docstring} better_robot.data_model.execution_batch.ExecutionInput.batch_shape
```

````

````{py:attribute} event_shape
:canonical: better_robot.data_model.execution_batch.ExecutionInput.event_shape
:type: tuple[int, ...]
:value: >
   None

```{autodoc2-docstring} better_robot.data_model.execution_batch.ExecutionInput.event_shape
```

````

````{py:attribute} batch_indices
:canonical: better_robot.data_model.execution_batch.ExecutionInput.batch_indices
:type: torch.Tensor
:value: >
   None

```{autodoc2-docstring} better_robot.data_model.execution_batch.ExecutionInput.batch_indices
```

````

````{py:method} gather() -> torch.Tensor
:canonical: better_robot.data_model.execution_batch.ExecutionInput.gather

```{autodoc2-docstring} better_robot.data_model.execution_batch.ExecutionInput.gather
```

````

````{py:method} reduce_gradient(execution_gradient: torch.Tensor) -> torch.Tensor
:canonical: better_robot.data_model.execution_batch.ExecutionInput.reduce_gradient

```{autodoc2-docstring} better_robot.data_model.execution_batch.ExecutionInput.reduce_gradient
```

````

`````

`````{py:class} ExecutionBatch
:canonical: better_robot.data_model.execution_batch.ExecutionBatch

```{autodoc2-docstring} better_robot.data_model.execution_batch.ExecutionBatch
```

````{py:attribute} batch_shape
:canonical: better_robot.data_model.execution_batch.ExecutionBatch.batch_shape
:type: tuple[int, ...]
:value: >
   None

```{autodoc2-docstring} better_robot.data_model.execution_batch.ExecutionBatch.batch_shape
```

````

````{py:attribute} size
:canonical: better_robot.data_model.execution_batch.ExecutionBatch.size
:type: int
:value: >
   None

```{autodoc2-docstring} better_robot.data_model.execution_batch.ExecutionBatch.size
```

````

````{py:attribute} q
:canonical: better_robot.data_model.execution_batch.ExecutionBatch.q
:type: better_robot.data_model.execution_batch.ExecutionInput
:value: >
   None

```{autodoc2-docstring} better_robot.data_model.execution_batch.ExecutionBatch.q
```

````

````{py:attribute} values
:canonical: better_robot.data_model.execution_batch.ExecutionBatch.values
:type: tuple[better_robot.data_model.execution_batch.ExecutionInput, ...]
:value: >
   None

```{autodoc2-docstring} better_robot.data_model.execution_batch.ExecutionBatch.values
```

````

````{py:method} unflatten(tensor: torch.Tensor) -> torch.Tensor
:canonical: better_robot.data_model.execution_batch.ExecutionBatch.unflatten

```{autodoc2-docstring} better_robot.data_model.execution_batch.ExecutionBatch.unflatten
```

````

`````

````{py:function} flatten_execution_batch(q: torch.Tensor, value_tensors: typing.Sequence[torch.Tensor] = (), *, value_event_ndims: typing.Sequence[int] = ()) -> better_robot.data_model.execution_batch.ExecutionBatch
:canonical: better_robot.data_model.execution_batch.flatten_execution_batch

```{autodoc2-docstring} better_robot.data_model.execution_batch.flatten_execution_batch
```
````
