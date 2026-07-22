# {py:mod}`better_robot.residuals.scene_sdf`

```{py:module} better_robot.residuals.scene_sdf
```

```{autodoc2-docstring} better_robot.residuals.scene_sdf
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`SceneSDFResult <better_robot.residuals.scene_sdf.SceneSDFResult>`
  - ```{autodoc2-docstring} better_robot.residuals.scene_sdf.SceneSDFResult
    :summary:
    ```
* - {py:obj}`SceneSDFState <better_robot.residuals.scene_sdf.SceneSDFState>`
  - ```{autodoc2-docstring} better_robot.residuals.scene_sdf.SceneSDFState
    :summary:
    ```
* - {py:obj}`ScenePenetrationResidual <better_robot.residuals.scene_sdf.ScenePenetrationResidual>`
  - ```{autodoc2-docstring} better_robot.residuals.scene_sdf.ScenePenetrationResidual
    :summary:
    ```
* - {py:obj}`SceneAttractionResidual <better_robot.residuals.scene_sdf.SceneAttractionResidual>`
  - ```{autodoc2-docstring} better_robot.residuals.scene_sdf.SceneAttractionResidual
    :summary:
    ```
* - {py:obj}`SceneClearanceResidual <better_robot.residuals.scene_sdf.SceneClearanceResidual>`
  - ```{autodoc2-docstring} better_robot.residuals.scene_sdf.SceneClearanceResidual
    :summary:
    ```
````

### API

`````{py:class} SceneSDFResult
:canonical: better_robot.residuals.scene_sdf.SceneSDFResult

```{autodoc2-docstring} better_robot.residuals.scene_sdf.SceneSDFResult
```

````{py:attribute} signed_distance
:canonical: better_robot.residuals.scene_sdf.SceneSDFResult.signed_distance
:type: torch.Tensor
:value: >
   None

```{autodoc2-docstring} better_robot.residuals.scene_sdf.SceneSDFResult.signed_distance
```

````

````{py:attribute} dmin
:canonical: better_robot.residuals.scene_sdf.SceneSDFResult.dmin
:type: torch.Tensor
:value: >
   None

```{autodoc2-docstring} better_robot.residuals.scene_sdf.SceneSDFResult.dmin
```

````

````{py:attribute} confidence
:canonical: better_robot.residuals.scene_sdf.SceneSDFResult.confidence
:type: torch.Tensor
:value: >
   None

```{autodoc2-docstring} better_robot.residuals.scene_sdf.SceneSDFResult.confidence
```

````

````{py:attribute} has_point
:canonical: better_robot.residuals.scene_sdf.SceneSDFResult.has_point
:type: torch.Tensor
:value: >
   None

```{autodoc2-docstring} better_robot.residuals.scene_sdf.SceneSDFResult.has_point
```

````

`````

`````{py:class} SceneSDFState(query_points: better_robot.residuals.utils.VariableLike | better_robot.residuals.nodes.Node | torch.Tensor, query_validity: better_robot.residuals.utils.VariableLike | better_robot.residuals.nodes.Node | torch.Tensor, scene_points: better_robot.residuals.utils.VariableLike | better_robot.residuals.nodes.Node | torch.Tensor, scene_normals: better_robot.residuals.utils.VariableLike | better_robot.residuals.nodes.Node | torch.Tensor, scene_validity: better_robot.residuals.utils.VariableLike | better_robot.residuals.nodes.Node | torch.Tensor, *, scene_confidence: better_robot.residuals.utils.VariableLike | better_robot.residuals.nodes.Node | torch.Tensor | None = None, distance: typing.Literal[point, plane] = 'point', chunk_size: int = 4096, eps: float = 1e-08)
:canonical: better_robot.residuals.scene_sdf.SceneSDFState

Bases: {py:obj}`better_robot.residuals.nodes.Node`

```{autodoc2-docstring} better_robot.residuals.scene_sdf.SceneSDFState
```

````{py:method} compute() -> better_robot.residuals.scene_sdf.SceneSDFResult
:canonical: better_robot.residuals.scene_sdf.SceneSDFState.compute

````

`````

`````{py:class} ScenePenetrationResidual(state: better_robot.residuals.scene_sdf.SceneSDFState, *, min_confidence: float | None = None, max_distance: float | None = None, mask: better_robot.residuals.utils.VariableLike | better_robot.residuals.nodes.Node | torch.Tensor | None = None, max_penetration: float | None = None, margin: float | None = None, weight: numbers.Real | torch.Tensor = 1.0, row_weight: better_robot.residuals.base.Weight | numbers.Real | torch.Tensor = 1.0, reduce: typing.Literal[sum, mean, mean_active] = 'sum', kernel: object | None = None, name: str = 'scene_penetration', enabled: bool = True)
:canonical: better_robot.residuals.scene_sdf.ScenePenetrationResidual

Bases: {py:obj}`better_robot.residuals.scene_sdf._ScenePenaltyResidual`

```{autodoc2-docstring} better_robot.residuals.scene_sdf.ScenePenetrationResidual
```

````{py:method} error() -> torch.Tensor
:canonical: better_robot.residuals.scene_sdf.ScenePenetrationResidual.error

````

`````

`````{py:class} SceneAttractionResidual(state: better_robot.residuals.scene_sdf.SceneSDFState, *, target_distance: float = 0.0, min_confidence: float | None = None, max_distance: float | None = None, mask: better_robot.residuals.utils.VariableLike | better_robot.residuals.nodes.Node | torch.Tensor | None = None, band: float | None = None, weight: numbers.Real | torch.Tensor = 1.0, row_weight: better_robot.residuals.base.Weight | numbers.Real | torch.Tensor = 1.0, reduce: typing.Literal[sum, mean, mean_active] = 'sum', kernel: object | None = None, name: str = 'scene_attraction', enabled: bool = True)
:canonical: better_robot.residuals.scene_sdf.SceneAttractionResidual

Bases: {py:obj}`better_robot.residuals.scene_sdf._ScenePenaltyResidual`

```{autodoc2-docstring} better_robot.residuals.scene_sdf.SceneAttractionResidual
```

````{py:method} error() -> torch.Tensor
:canonical: better_robot.residuals.scene_sdf.SceneAttractionResidual.error

````

`````

`````{py:class} SceneClearanceResidual(state: better_robot.residuals.scene_sdf.SceneSDFState, *, clearance: float, min_confidence: float | None = None, max_distance: float | None = None, mask: better_robot.residuals.utils.VariableLike | better_robot.residuals.nodes.Node | torch.Tensor | None = None, margin: float | None = None, weight: numbers.Real | torch.Tensor = 1.0, row_weight: better_robot.residuals.base.Weight | numbers.Real | torch.Tensor = 1.0, reduce: typing.Literal[sum, mean, mean_active] = 'sum', kernel: object | None = None, name: str = 'scene_clearance', enabled: bool = True)
:canonical: better_robot.residuals.scene_sdf.SceneClearanceResidual

Bases: {py:obj}`better_robot.residuals.scene_sdf._ScenePenaltyResidual`

```{autodoc2-docstring} better_robot.residuals.scene_sdf.SceneClearanceResidual
```

````{py:method} error() -> torch.Tensor
:canonical: better_robot.residuals.scene_sdf.SceneClearanceResidual.error

````

`````
