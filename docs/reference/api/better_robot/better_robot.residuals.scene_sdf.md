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
* - {py:obj}`SceneSDFProvider <better_robot.residuals.scene_sdf.SceneSDFProvider>`
  - ```{autodoc2-docstring} better_robot.residuals.scene_sdf.SceneSDFProvider
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

`````{py:class} SceneSDFProvider
:canonical: better_robot.residuals.scene_sdf.SceneSDFProvider

```{autodoc2-docstring} better_robot.residuals.scene_sdf.SceneSDFProvider
```

````{py:attribute} query_points
:canonical: better_robot.residuals.scene_sdf.SceneSDFProvider.query_points
:type: str
:value: >
   'scene_query_points'

```{autodoc2-docstring} better_robot.residuals.scene_sdf.SceneSDFProvider.query_points
```

````

````{py:attribute} query_validity
:canonical: better_robot.residuals.scene_sdf.SceneSDFProvider.query_validity
:type: str
:value: >
   'scene_query_validity'

```{autodoc2-docstring} better_robot.residuals.scene_sdf.SceneSDFProvider.query_validity
```

````

````{py:attribute} scene_points
:canonical: better_robot.residuals.scene_sdf.SceneSDFProvider.scene_points
:type: str
:value: >
   'scene_points'

```{autodoc2-docstring} better_robot.residuals.scene_sdf.SceneSDFProvider.scene_points
```

````

````{py:attribute} scene_normals
:canonical: better_robot.residuals.scene_sdf.SceneSDFProvider.scene_normals
:type: str
:value: >
   'scene_normals'

```{autodoc2-docstring} better_robot.residuals.scene_sdf.SceneSDFProvider.scene_normals
```

````

````{py:attribute} scene_validity
:canonical: better_robot.residuals.scene_sdf.SceneSDFProvider.scene_validity
:type: str
:value: >
   'scene_validity'

```{autodoc2-docstring} better_robot.residuals.scene_sdf.SceneSDFProvider.scene_validity
```

````

````{py:attribute} scene_confidence
:canonical: better_robot.residuals.scene_sdf.SceneSDFProvider.scene_confidence
:type: str | None
:value: >
   None

```{autodoc2-docstring} better_robot.residuals.scene_sdf.SceneSDFProvider.scene_confidence
```

````

````{py:attribute} output
:canonical: better_robot.residuals.scene_sdf.SceneSDFProvider.output
:type: str
:value: >
   'scene_sdf'

```{autodoc2-docstring} better_robot.residuals.scene_sdf.SceneSDFProvider.output
```

````

````{py:attribute} name
:canonical: better_robot.residuals.scene_sdf.SceneSDFProvider.name
:type: str
:value: >
   'scene_sdf_provider'

```{autodoc2-docstring} better_robot.residuals.scene_sdf.SceneSDFProvider.name
```

````

````{py:attribute} chunk_size
:canonical: better_robot.residuals.scene_sdf.SceneSDFProvider.chunk_size
:type: int
:value: >
   4096

```{autodoc2-docstring} better_robot.residuals.scene_sdf.SceneSDFProvider.chunk_size
```

````

````{py:attribute} eps
:canonical: better_robot.residuals.scene_sdf.SceneSDFProvider.eps
:type: float
:value: >
   1e-08

```{autodoc2-docstring} better_robot.residuals.scene_sdf.SceneSDFProvider.eps
```

````

````{py:property} inputs
:canonical: better_robot.residuals.scene_sdf.SceneSDFProvider.inputs
:type: tuple[str, ...]

```{autodoc2-docstring} better_robot.residuals.scene_sdf.SceneSDFProvider.inputs
```

````

````{py:property} outputs
:canonical: better_robot.residuals.scene_sdf.SceneSDFProvider.outputs
:type: tuple[str, ...]

```{autodoc2-docstring} better_robot.residuals.scene_sdf.SceneSDFProvider.outputs
```

````

`````

````{py:class} ScenePenetrationResidual(frames: int, points: int, *, scene_sdf: str = 'scene_sdf', name: str = 'scene_penetration')
:canonical: better_robot.residuals.scene_sdf.ScenePenetrationResidual

Bases: {py:obj}`better_robot.residuals.scene_sdf._ScenePenaltyResidual`

```{autodoc2-docstring} better_robot.residuals.scene_sdf.ScenePenetrationResidual
```

````

````{py:class} SceneAttractionResidual(frames: int, points: int, *, target_distance: float = 0.0, scene_sdf: str = 'scene_sdf', name: str = 'scene_attraction')
:canonical: better_robot.residuals.scene_sdf.SceneAttractionResidual

Bases: {py:obj}`better_robot.residuals.scene_sdf._ScenePenaltyResidual`

```{autodoc2-docstring} better_robot.residuals.scene_sdf.SceneAttractionResidual
```

````

````{py:class} SceneClearanceResidual(frames: int, points: int, *, clearance: float, scene_sdf: str = 'scene_sdf', name: str = 'scene_clearance')
:canonical: better_robot.residuals.scene_sdf.SceneClearanceResidual

Bases: {py:obj}`better_robot.residuals.scene_sdf._ScenePenaltyResidual`

```{autodoc2-docstring} better_robot.residuals.scene_sdf.SceneClearanceResidual
```

````
