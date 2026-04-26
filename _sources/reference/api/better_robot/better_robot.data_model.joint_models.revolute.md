# {py:mod}`better_robot.data_model.joint_models.revolute`

```{py:module} better_robot.data_model.joint_models.revolute
```

```{autodoc2-docstring} better_robot.data_model.joint_models.revolute
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`JointRX <better_robot.data_model.joint_models.revolute.JointRX>`
  - ```{autodoc2-docstring} better_robot.data_model.joint_models.revolute.JointRX
    :summary:
    ```
* - {py:obj}`JointRY <better_robot.data_model.joint_models.revolute.JointRY>`
  - ```{autodoc2-docstring} better_robot.data_model.joint_models.revolute.JointRY
    :summary:
    ```
* - {py:obj}`JointRZ <better_robot.data_model.joint_models.revolute.JointRZ>`
  - ```{autodoc2-docstring} better_robot.data_model.joint_models.revolute.JointRZ
    :summary:
    ```
* - {py:obj}`JointRevoluteUnaligned <better_robot.data_model.joint_models.revolute.JointRevoluteUnaligned>`
  - ```{autodoc2-docstring} better_robot.data_model.joint_models.revolute.JointRevoluteUnaligned
    :summary:
    ```
* - {py:obj}`JointRevoluteUnbounded <better_robot.data_model.joint_models.revolute.JointRevoluteUnbounded>`
  - ```{autodoc2-docstring} better_robot.data_model.joint_models.revolute.JointRevoluteUnbounded
    :summary:
    ```
````

### API

`````{py:class} JointRX
:canonical: better_robot.data_model.joint_models.revolute.JointRX

```{autodoc2-docstring} better_robot.data_model.joint_models.revolute.JointRX
```

````{py:attribute} kind
:canonical: better_robot.data_model.joint_models.revolute.JointRX.kind
:type: str
:value: >
   'revolute_rx'

```{autodoc2-docstring} better_robot.data_model.joint_models.revolute.JointRX.kind
```

````

````{py:attribute} nq
:canonical: better_robot.data_model.joint_models.revolute.JointRX.nq
:type: int
:value: >
   1

```{autodoc2-docstring} better_robot.data_model.joint_models.revolute.JointRX.nq
```

````

````{py:attribute} nv
:canonical: better_robot.data_model.joint_models.revolute.JointRX.nv
:type: int
:value: >
   1

```{autodoc2-docstring} better_robot.data_model.joint_models.revolute.JointRX.nv
```

````

````{py:attribute} axis
:canonical: better_robot.data_model.joint_models.revolute.JointRX.axis
:type: torch.Tensor | None
:value: >
   None

```{autodoc2-docstring} better_robot.data_model.joint_models.revolute.JointRX.axis
```

````

````{py:method} joint_transform(q_slice)
:canonical: better_robot.data_model.joint_models.revolute.JointRX.joint_transform

```{autodoc2-docstring} better_robot.data_model.joint_models.revolute.JointRX.joint_transform
```

````

````{py:method} joint_motion_subspace(q_slice)
:canonical: better_robot.data_model.joint_models.revolute.JointRX.joint_motion_subspace

```{autodoc2-docstring} better_robot.data_model.joint_models.revolute.JointRX.joint_motion_subspace
```

````

````{py:method} joint_velocity(q_slice, v_slice)
:canonical: better_robot.data_model.joint_models.revolute.JointRX.joint_velocity

```{autodoc2-docstring} better_robot.data_model.joint_models.revolute.JointRX.joint_velocity
```

````

````{py:method} integrate(q_slice, v_slice)
:canonical: better_robot.data_model.joint_models.revolute.JointRX.integrate

```{autodoc2-docstring} better_robot.data_model.joint_models.revolute.JointRX.integrate
```

````

````{py:method} difference(q0_slice, q1_slice)
:canonical: better_robot.data_model.joint_models.revolute.JointRX.difference

```{autodoc2-docstring} better_robot.data_model.joint_models.revolute.JointRX.difference
```

````

````{py:method} random_configuration(generator, lower, upper)
:canonical: better_robot.data_model.joint_models.revolute.JointRX.random_configuration

```{autodoc2-docstring} better_robot.data_model.joint_models.revolute.JointRX.random_configuration
```

````

````{py:method} neutral()
:canonical: better_robot.data_model.joint_models.revolute.JointRX.neutral

```{autodoc2-docstring} better_robot.data_model.joint_models.revolute.JointRX.neutral
```

````

`````

`````{py:class} JointRY
:canonical: better_robot.data_model.joint_models.revolute.JointRY

```{autodoc2-docstring} better_robot.data_model.joint_models.revolute.JointRY
```

````{py:attribute} kind
:canonical: better_robot.data_model.joint_models.revolute.JointRY.kind
:type: str
:value: >
   'revolute_ry'

```{autodoc2-docstring} better_robot.data_model.joint_models.revolute.JointRY.kind
```

````

````{py:attribute} nq
:canonical: better_robot.data_model.joint_models.revolute.JointRY.nq
:type: int
:value: >
   1

```{autodoc2-docstring} better_robot.data_model.joint_models.revolute.JointRY.nq
```

````

````{py:attribute} nv
:canonical: better_robot.data_model.joint_models.revolute.JointRY.nv
:type: int
:value: >
   1

```{autodoc2-docstring} better_robot.data_model.joint_models.revolute.JointRY.nv
```

````

````{py:attribute} axis
:canonical: better_robot.data_model.joint_models.revolute.JointRY.axis
:type: torch.Tensor | None
:value: >
   None

```{autodoc2-docstring} better_robot.data_model.joint_models.revolute.JointRY.axis
```

````

````{py:method} joint_transform(q_slice)
:canonical: better_robot.data_model.joint_models.revolute.JointRY.joint_transform

```{autodoc2-docstring} better_robot.data_model.joint_models.revolute.JointRY.joint_transform
```

````

````{py:method} joint_motion_subspace(q_slice)
:canonical: better_robot.data_model.joint_models.revolute.JointRY.joint_motion_subspace

```{autodoc2-docstring} better_robot.data_model.joint_models.revolute.JointRY.joint_motion_subspace
```

````

````{py:method} joint_velocity(q_slice, v_slice)
:canonical: better_robot.data_model.joint_models.revolute.JointRY.joint_velocity

```{autodoc2-docstring} better_robot.data_model.joint_models.revolute.JointRY.joint_velocity
```

````

````{py:method} integrate(q_slice, v_slice)
:canonical: better_robot.data_model.joint_models.revolute.JointRY.integrate

```{autodoc2-docstring} better_robot.data_model.joint_models.revolute.JointRY.integrate
```

````

````{py:method} difference(q0_slice, q1_slice)
:canonical: better_robot.data_model.joint_models.revolute.JointRY.difference

```{autodoc2-docstring} better_robot.data_model.joint_models.revolute.JointRY.difference
```

````

````{py:method} random_configuration(generator, lower, upper)
:canonical: better_robot.data_model.joint_models.revolute.JointRY.random_configuration

```{autodoc2-docstring} better_robot.data_model.joint_models.revolute.JointRY.random_configuration
```

````

````{py:method} neutral()
:canonical: better_robot.data_model.joint_models.revolute.JointRY.neutral

```{autodoc2-docstring} better_robot.data_model.joint_models.revolute.JointRY.neutral
```

````

`````

`````{py:class} JointRZ
:canonical: better_robot.data_model.joint_models.revolute.JointRZ

```{autodoc2-docstring} better_robot.data_model.joint_models.revolute.JointRZ
```

````{py:attribute} kind
:canonical: better_robot.data_model.joint_models.revolute.JointRZ.kind
:type: str
:value: >
   'revolute_rz'

```{autodoc2-docstring} better_robot.data_model.joint_models.revolute.JointRZ.kind
```

````

````{py:attribute} nq
:canonical: better_robot.data_model.joint_models.revolute.JointRZ.nq
:type: int
:value: >
   1

```{autodoc2-docstring} better_robot.data_model.joint_models.revolute.JointRZ.nq
```

````

````{py:attribute} nv
:canonical: better_robot.data_model.joint_models.revolute.JointRZ.nv
:type: int
:value: >
   1

```{autodoc2-docstring} better_robot.data_model.joint_models.revolute.JointRZ.nv
```

````

````{py:attribute} axis
:canonical: better_robot.data_model.joint_models.revolute.JointRZ.axis
:type: torch.Tensor | None
:value: >
   None

```{autodoc2-docstring} better_robot.data_model.joint_models.revolute.JointRZ.axis
```

````

````{py:method} joint_transform(q_slice)
:canonical: better_robot.data_model.joint_models.revolute.JointRZ.joint_transform

```{autodoc2-docstring} better_robot.data_model.joint_models.revolute.JointRZ.joint_transform
```

````

````{py:method} joint_motion_subspace(q_slice)
:canonical: better_robot.data_model.joint_models.revolute.JointRZ.joint_motion_subspace

```{autodoc2-docstring} better_robot.data_model.joint_models.revolute.JointRZ.joint_motion_subspace
```

````

````{py:method} joint_velocity(q_slice, v_slice)
:canonical: better_robot.data_model.joint_models.revolute.JointRZ.joint_velocity

```{autodoc2-docstring} better_robot.data_model.joint_models.revolute.JointRZ.joint_velocity
```

````

````{py:method} integrate(q_slice, v_slice)
:canonical: better_robot.data_model.joint_models.revolute.JointRZ.integrate

```{autodoc2-docstring} better_robot.data_model.joint_models.revolute.JointRZ.integrate
```

````

````{py:method} difference(q0_slice, q1_slice)
:canonical: better_robot.data_model.joint_models.revolute.JointRZ.difference

```{autodoc2-docstring} better_robot.data_model.joint_models.revolute.JointRZ.difference
```

````

````{py:method} random_configuration(generator, lower, upper)
:canonical: better_robot.data_model.joint_models.revolute.JointRZ.random_configuration

```{autodoc2-docstring} better_robot.data_model.joint_models.revolute.JointRZ.random_configuration
```

````

````{py:method} neutral()
:canonical: better_robot.data_model.joint_models.revolute.JointRZ.neutral

```{autodoc2-docstring} better_robot.data_model.joint_models.revolute.JointRZ.neutral
```

````

`````

`````{py:class} JointRevoluteUnaligned
:canonical: better_robot.data_model.joint_models.revolute.JointRevoluteUnaligned

```{autodoc2-docstring} better_robot.data_model.joint_models.revolute.JointRevoluteUnaligned
```

````{py:attribute} axis
:canonical: better_robot.data_model.joint_models.revolute.JointRevoluteUnaligned.axis
:type: torch.Tensor
:value: >
   'field(...)'

```{autodoc2-docstring} better_robot.data_model.joint_models.revolute.JointRevoluteUnaligned.axis
```

````

````{py:attribute} kind
:canonical: better_robot.data_model.joint_models.revolute.JointRevoluteUnaligned.kind
:type: str
:value: >
   'revolute_unaligned'

```{autodoc2-docstring} better_robot.data_model.joint_models.revolute.JointRevoluteUnaligned.kind
```

````

````{py:attribute} nq
:canonical: better_robot.data_model.joint_models.revolute.JointRevoluteUnaligned.nq
:type: int
:value: >
   1

```{autodoc2-docstring} better_robot.data_model.joint_models.revolute.JointRevoluteUnaligned.nq
```

````

````{py:attribute} nv
:canonical: better_robot.data_model.joint_models.revolute.JointRevoluteUnaligned.nv
:type: int
:value: >
   1

```{autodoc2-docstring} better_robot.data_model.joint_models.revolute.JointRevoluteUnaligned.nv
```

````

````{py:method} joint_transform(q_slice)
:canonical: better_robot.data_model.joint_models.revolute.JointRevoluteUnaligned.joint_transform

```{autodoc2-docstring} better_robot.data_model.joint_models.revolute.JointRevoluteUnaligned.joint_transform
```

````

````{py:method} joint_motion_subspace(q_slice)
:canonical: better_robot.data_model.joint_models.revolute.JointRevoluteUnaligned.joint_motion_subspace

```{autodoc2-docstring} better_robot.data_model.joint_models.revolute.JointRevoluteUnaligned.joint_motion_subspace
```

````

````{py:method} joint_velocity(q_slice, v_slice)
:canonical: better_robot.data_model.joint_models.revolute.JointRevoluteUnaligned.joint_velocity

```{autodoc2-docstring} better_robot.data_model.joint_models.revolute.JointRevoluteUnaligned.joint_velocity
```

````

````{py:method} integrate(q_slice, v_slice)
:canonical: better_robot.data_model.joint_models.revolute.JointRevoluteUnaligned.integrate

```{autodoc2-docstring} better_robot.data_model.joint_models.revolute.JointRevoluteUnaligned.integrate
```

````

````{py:method} difference(q0_slice, q1_slice)
:canonical: better_robot.data_model.joint_models.revolute.JointRevoluteUnaligned.difference

```{autodoc2-docstring} better_robot.data_model.joint_models.revolute.JointRevoluteUnaligned.difference
```

````

````{py:method} random_configuration(generator, lower, upper)
:canonical: better_robot.data_model.joint_models.revolute.JointRevoluteUnaligned.random_configuration

```{autodoc2-docstring} better_robot.data_model.joint_models.revolute.JointRevoluteUnaligned.random_configuration
```

````

````{py:method} neutral()
:canonical: better_robot.data_model.joint_models.revolute.JointRevoluteUnaligned.neutral

```{autodoc2-docstring} better_robot.data_model.joint_models.revolute.JointRevoluteUnaligned.neutral
```

````

`````

`````{py:class} JointRevoluteUnbounded
:canonical: better_robot.data_model.joint_models.revolute.JointRevoluteUnbounded

```{autodoc2-docstring} better_robot.data_model.joint_models.revolute.JointRevoluteUnbounded
```

````{py:attribute} axis
:canonical: better_robot.data_model.joint_models.revolute.JointRevoluteUnbounded.axis
:type: torch.Tensor
:value: >
   'field(...)'

```{autodoc2-docstring} better_robot.data_model.joint_models.revolute.JointRevoluteUnbounded.axis
```

````

````{py:attribute} kind
:canonical: better_robot.data_model.joint_models.revolute.JointRevoluteUnbounded.kind
:type: str
:value: >
   'revolute_unbounded'

```{autodoc2-docstring} better_robot.data_model.joint_models.revolute.JointRevoluteUnbounded.kind
```

````

````{py:attribute} nq
:canonical: better_robot.data_model.joint_models.revolute.JointRevoluteUnbounded.nq
:type: int
:value: >
   2

```{autodoc2-docstring} better_robot.data_model.joint_models.revolute.JointRevoluteUnbounded.nq
```

````

````{py:attribute} nv
:canonical: better_robot.data_model.joint_models.revolute.JointRevoluteUnbounded.nv
:type: int
:value: >
   1

```{autodoc2-docstring} better_robot.data_model.joint_models.revolute.JointRevoluteUnbounded.nv
```

````

````{py:method} joint_transform(q_slice: torch.Tensor) -> torch.Tensor
:canonical: better_robot.data_model.joint_models.revolute.JointRevoluteUnbounded.joint_transform

```{autodoc2-docstring} better_robot.data_model.joint_models.revolute.JointRevoluteUnbounded.joint_transform
```

````

````{py:method} joint_motion_subspace(q_slice)
:canonical: better_robot.data_model.joint_models.revolute.JointRevoluteUnbounded.joint_motion_subspace

```{autodoc2-docstring} better_robot.data_model.joint_models.revolute.JointRevoluteUnbounded.joint_motion_subspace
```

````

````{py:method} joint_velocity(q_slice, v_slice)
:canonical: better_robot.data_model.joint_models.revolute.JointRevoluteUnbounded.joint_velocity

```{autodoc2-docstring} better_robot.data_model.joint_models.revolute.JointRevoluteUnbounded.joint_velocity
```

````

````{py:method} integrate(q_slice: torch.Tensor, v_slice: torch.Tensor) -> torch.Tensor
:canonical: better_robot.data_model.joint_models.revolute.JointRevoluteUnbounded.integrate

```{autodoc2-docstring} better_robot.data_model.joint_models.revolute.JointRevoluteUnbounded.integrate
```

````

````{py:method} difference(q0_slice: torch.Tensor, q1_slice: torch.Tensor) -> torch.Tensor
:canonical: better_robot.data_model.joint_models.revolute.JointRevoluteUnbounded.difference

```{autodoc2-docstring} better_robot.data_model.joint_models.revolute.JointRevoluteUnbounded.difference
```

````

````{py:method} random_configuration(generator, lower, upper) -> torch.Tensor
:canonical: better_robot.data_model.joint_models.revolute.JointRevoluteUnbounded.random_configuration

```{autodoc2-docstring} better_robot.data_model.joint_models.revolute.JointRevoluteUnbounded.random_configuration
```

````

````{py:method} neutral() -> torch.Tensor
:canonical: better_robot.data_model.joint_models.revolute.JointRevoluteUnbounded.neutral

```{autodoc2-docstring} better_robot.data_model.joint_models.revolute.JointRevoluteUnbounded.neutral
```

````

`````
