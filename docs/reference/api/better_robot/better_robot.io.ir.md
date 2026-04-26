# {py:mod}`better_robot.io.ir`

```{py:module} better_robot.io.ir
```

```{autodoc2-docstring} better_robot.io.ir
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`IRJoint <better_robot.io.ir.IRJoint>`
  - ```{autodoc2-docstring} better_robot.io.ir.IRJoint
    :summary:
    ```
* - {py:obj}`IRGeom <better_robot.io.ir.IRGeom>`
  - ```{autodoc2-docstring} better_robot.io.ir.IRGeom
    :summary:
    ```
* - {py:obj}`IRBody <better_robot.io.ir.IRBody>`
  - ```{autodoc2-docstring} better_robot.io.ir.IRBody
    :summary:
    ```
* - {py:obj}`IRFrame <better_robot.io.ir.IRFrame>`
  - ```{autodoc2-docstring} better_robot.io.ir.IRFrame
    :summary:
    ```
* - {py:obj}`IRModel <better_robot.io.ir.IRModel>`
  - ```{autodoc2-docstring} better_robot.io.ir.IRModel
    :summary:
    ```
````

### Data

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`IR_SCHEMA_VERSION <better_robot.io.ir.IR_SCHEMA_VERSION>`
  - ```{autodoc2-docstring} better_robot.io.ir.IR_SCHEMA_VERSION
    :summary:
    ```
````

### API

`````{py:class} IRJoint
:canonical: better_robot.io.ir.IRJoint

```{autodoc2-docstring} better_robot.io.ir.IRJoint
```

````{py:attribute} name
:canonical: better_robot.io.ir.IRJoint.name
:type: str
:value: >
   None

```{autodoc2-docstring} better_robot.io.ir.IRJoint.name
```

````

````{py:attribute} parent_body
:canonical: better_robot.io.ir.IRJoint.parent_body
:type: str
:value: >
   None

```{autodoc2-docstring} better_robot.io.ir.IRJoint.parent_body
```

````

````{py:attribute} child_body
:canonical: better_robot.io.ir.IRJoint.child_body
:type: str
:value: >
   None

```{autodoc2-docstring} better_robot.io.ir.IRJoint.child_body
```

````

````{py:attribute} kind
:canonical: better_robot.io.ir.IRJoint.kind
:type: str
:value: >
   None

```{autodoc2-docstring} better_robot.io.ir.IRJoint.kind
```

````

````{py:attribute} axis
:canonical: better_robot.io.ir.IRJoint.axis
:type: typing.Optional[torch.Tensor]
:value: >
   None

```{autodoc2-docstring} better_robot.io.ir.IRJoint.axis
```

````

````{py:attribute} origin
:canonical: better_robot.io.ir.IRJoint.origin
:type: torch.Tensor
:value: >
   'field(...)'

```{autodoc2-docstring} better_robot.io.ir.IRJoint.origin
```

````

````{py:attribute} lower
:canonical: better_robot.io.ir.IRJoint.lower
:type: typing.Optional[float]
:value: >
   None

```{autodoc2-docstring} better_robot.io.ir.IRJoint.lower
```

````

````{py:attribute} upper
:canonical: better_robot.io.ir.IRJoint.upper
:type: typing.Optional[float]
:value: >
   None

```{autodoc2-docstring} better_robot.io.ir.IRJoint.upper
```

````

````{py:attribute} velocity_limit
:canonical: better_robot.io.ir.IRJoint.velocity_limit
:type: typing.Optional[float]
:value: >
   None

```{autodoc2-docstring} better_robot.io.ir.IRJoint.velocity_limit
```

````

````{py:attribute} effort_limit
:canonical: better_robot.io.ir.IRJoint.effort_limit
:type: typing.Optional[float]
:value: >
   None

```{autodoc2-docstring} better_robot.io.ir.IRJoint.effort_limit
```

````

````{py:attribute} mimic_source
:canonical: better_robot.io.ir.IRJoint.mimic_source
:type: typing.Optional[str]
:value: >
   None

```{autodoc2-docstring} better_robot.io.ir.IRJoint.mimic_source
```

````

````{py:attribute} mimic_multiplier
:canonical: better_robot.io.ir.IRJoint.mimic_multiplier
:type: float
:value: >
   1.0

```{autodoc2-docstring} better_robot.io.ir.IRJoint.mimic_multiplier
```

````

````{py:attribute} mimic_offset
:canonical: better_robot.io.ir.IRJoint.mimic_offset
:type: float
:value: >
   0.0

```{autodoc2-docstring} better_robot.io.ir.IRJoint.mimic_offset
```

````

`````

`````{py:class} IRGeom
:canonical: better_robot.io.ir.IRGeom

```{autodoc2-docstring} better_robot.io.ir.IRGeom
```

````{py:attribute} kind
:canonical: better_robot.io.ir.IRGeom.kind
:type: str
:value: >
   None

```{autodoc2-docstring} better_robot.io.ir.IRGeom.kind
```

````

````{py:attribute} params
:canonical: better_robot.io.ir.IRGeom.params
:type: dict
:value: >
   None

```{autodoc2-docstring} better_robot.io.ir.IRGeom.params
```

````

````{py:attribute} origin
:canonical: better_robot.io.ir.IRGeom.origin
:type: torch.Tensor
:value: >
   None

```{autodoc2-docstring} better_robot.io.ir.IRGeom.origin
```

````

````{py:attribute} rgba
:canonical: better_robot.io.ir.IRGeom.rgba
:type: typing.Optional[tuple[float, float, float, float]]
:value: >
   None

```{autodoc2-docstring} better_robot.io.ir.IRGeom.rgba
```

````

`````

`````{py:class} IRBody
:canonical: better_robot.io.ir.IRBody

```{autodoc2-docstring} better_robot.io.ir.IRBody
```

````{py:attribute} name
:canonical: better_robot.io.ir.IRBody.name
:type: str
:value: >
   None

```{autodoc2-docstring} better_robot.io.ir.IRBody.name
```

````

````{py:attribute} mass
:canonical: better_robot.io.ir.IRBody.mass
:type: float
:value: >
   0.0

```{autodoc2-docstring} better_robot.io.ir.IRBody.mass
```

````

````{py:attribute} com
:canonical: better_robot.io.ir.IRBody.com
:type: torch.Tensor
:value: >
   'field(...)'

```{autodoc2-docstring} better_robot.io.ir.IRBody.com
```

````

````{py:attribute} inertia
:canonical: better_robot.io.ir.IRBody.inertia
:type: torch.Tensor
:value: >
   'field(...)'

```{autodoc2-docstring} better_robot.io.ir.IRBody.inertia
```

````

````{py:attribute} visual_geoms
:canonical: better_robot.io.ir.IRBody.visual_geoms
:type: list[better_robot.io.ir.IRGeom]
:value: >
   'field(...)'

```{autodoc2-docstring} better_robot.io.ir.IRBody.visual_geoms
```

````

````{py:attribute} collision_geoms
:canonical: better_robot.io.ir.IRBody.collision_geoms
:type: list[better_robot.io.ir.IRGeom]
:value: >
   'field(...)'

```{autodoc2-docstring} better_robot.io.ir.IRBody.collision_geoms
```

````

`````

`````{py:class} IRFrame
:canonical: better_robot.io.ir.IRFrame

```{autodoc2-docstring} better_robot.io.ir.IRFrame
```

````{py:attribute} name
:canonical: better_robot.io.ir.IRFrame.name
:type: str
:value: >
   None

```{autodoc2-docstring} better_robot.io.ir.IRFrame.name
```

````

````{py:attribute} parent_body
:canonical: better_robot.io.ir.IRFrame.parent_body
:type: str
:value: >
   None

```{autodoc2-docstring} better_robot.io.ir.IRFrame.parent_body
```

````

````{py:attribute} placement
:canonical: better_robot.io.ir.IRFrame.placement
:type: torch.Tensor
:value: >
   None

```{autodoc2-docstring} better_robot.io.ir.IRFrame.placement
```

````

````{py:attribute} frame_type
:canonical: better_robot.io.ir.IRFrame.frame_type
:type: str
:value: >
   'op'

```{autodoc2-docstring} better_robot.io.ir.IRFrame.frame_type
```

````

`````

````{py:data} IR_SCHEMA_VERSION
:canonical: better_robot.io.ir.IR_SCHEMA_VERSION
:type: int
:value: >
   1

```{autodoc2-docstring} better_robot.io.ir.IR_SCHEMA_VERSION
```

````

`````{py:class} IRModel
:canonical: better_robot.io.ir.IRModel

```{autodoc2-docstring} better_robot.io.ir.IRModel
```

````{py:attribute} name
:canonical: better_robot.io.ir.IRModel.name
:type: str
:value: >
   None

```{autodoc2-docstring} better_robot.io.ir.IRModel.name
```

````

````{py:attribute} bodies
:canonical: better_robot.io.ir.IRModel.bodies
:type: list[better_robot.io.ir.IRBody]
:value: >
   None

```{autodoc2-docstring} better_robot.io.ir.IRModel.bodies
```

````

````{py:attribute} joints
:canonical: better_robot.io.ir.IRModel.joints
:type: list[better_robot.io.ir.IRJoint]
:value: >
   None

```{autodoc2-docstring} better_robot.io.ir.IRModel.joints
```

````

````{py:attribute} frames
:canonical: better_robot.io.ir.IRModel.frames
:type: list[better_robot.io.ir.IRFrame]
:value: >
   'field(...)'

```{autodoc2-docstring} better_robot.io.ir.IRModel.frames
```

````

````{py:attribute} root_body
:canonical: better_robot.io.ir.IRModel.root_body
:type: str
:value: <Multiline-String>

```{autodoc2-docstring} better_robot.io.ir.IRModel.root_body
```

````

````{py:attribute} gravity
:canonical: better_robot.io.ir.IRModel.gravity
:type: torch.Tensor
:value: >
   'field(...)'

```{autodoc2-docstring} better_robot.io.ir.IRModel.gravity
```

````

````{py:attribute} schema_version
:canonical: better_robot.io.ir.IRModel.schema_version
:type: int
:value: >
   None

```{autodoc2-docstring} better_robot.io.ir.IRModel.schema_version
```

````

````{py:attribute} meta
:canonical: better_robot.io.ir.IRModel.meta
:type: dict
:value: >
   'field(...)'

```{autodoc2-docstring} better_robot.io.ir.IRModel.meta
```

````

`````

````{py:exception} IRError()
:canonical: better_robot.io.ir.IRError

Bases: {py:obj}`ValueError`

```{autodoc2-docstring} better_robot.io.ir.IRError
```

````
