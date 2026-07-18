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

````{py:class} MaskedChamferResidual(frames: int, source_count: int, target_count: int, *, source: str = 'points', target: str = 'target_points', source_validity: str = 'point_validity', target_validity: str = 'target_validity', vertex_weights: str | None = None, bidirectional: bool = True, chunk_size: int = 4096, name: str = 'masked_chamfer')
:canonical: better_robot.residuals.chamfer.MaskedChamferResidual

```{autodoc2-docstring} better_robot.residuals.chamfer.MaskedChamferResidual
```

````
