# {py:mod}`better_robot.residuals.chamfer`

```{py:module} better_robot.residuals.chamfer
```

```{autodoc2-docstring} better_robot.residuals.chamfer
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`MaskedChamferResidual <better_robot.residuals.chamfer.MaskedChamferResidual>`
  - ```{autodoc2-docstring} better_robot.residuals.chamfer.MaskedChamferResidual
    :summary:
    ```
````

### API

`````{py:class} MaskedChamferResidual(source: better_robot.residuals.utils.VariableLike | torch.Tensor, target: better_robot.residuals.utils.VariableLike | torch.Tensor, source_validity: better_robot.residuals.utils.VariableLike | torch.Tensor, target_validity: better_robot.residuals.utils.VariableLike | torch.Tensor, *, vertex_weights: better_robot.residuals.utils.VariableLike | torch.Tensor | None = None, bidirectional: bool = True, chunk_size: int = 4096, weight: numbers.Real | torch.Tensor = 1.0, row_weight: better_robot.residuals.base.Weight | numbers.Real | torch.Tensor = 1.0, kernel: object | None = None, name: str = 'masked_chamfer')
:canonical: better_robot.residuals.chamfer.MaskedChamferResidual

Bases: {py:obj}`better_robot.residuals.base.Residual`

```{autodoc2-docstring} better_robot.residuals.chamfer.MaskedChamferResidual
```

````{py:method} error() -> torch.Tensor
:canonical: better_robot.residuals.chamfer.MaskedChamferResidual.error

````

`````
