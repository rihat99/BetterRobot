# {py:mod}`better_robot.backends.protocol`

```{py:module} better_robot.backends.protocol
```

```{autodoc2-docstring} better_robot.backends.protocol
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`LieOps <better_robot.backends.protocol.LieOps>`
  - ```{autodoc2-docstring} better_robot.backends.protocol.LieOps
    :summary:
    ```
* - {py:obj}`KinematicsOps <better_robot.backends.protocol.KinematicsOps>`
  - ```{autodoc2-docstring} better_robot.backends.protocol.KinematicsOps
    :summary:
    ```
* - {py:obj}`DynamicsOps <better_robot.backends.protocol.DynamicsOps>`
  - ```{autodoc2-docstring} better_robot.backends.protocol.DynamicsOps
    :summary:
    ```
* - {py:obj}`Backend <better_robot.backends.protocol.Backend>`
  - ```{autodoc2-docstring} better_robot.backends.protocol.Backend
    :summary:
    ```
````

### API

`````{py:class} LieOps
:canonical: better_robot.backends.protocol.LieOps

Bases: {py:obj}`typing.Protocol`

```{autodoc2-docstring} better_robot.backends.protocol.LieOps
```

````{py:method} se3_compose(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor
:canonical: better_robot.backends.protocol.LieOps.se3_compose

```{autodoc2-docstring} better_robot.backends.protocol.LieOps.se3_compose
```

````

````{py:method} se3_inverse(t: torch.Tensor) -> torch.Tensor
:canonical: better_robot.backends.protocol.LieOps.se3_inverse

```{autodoc2-docstring} better_robot.backends.protocol.LieOps.se3_inverse
```

````

````{py:method} se3_log(t: torch.Tensor) -> torch.Tensor
:canonical: better_robot.backends.protocol.LieOps.se3_log

```{autodoc2-docstring} better_robot.backends.protocol.LieOps.se3_log
```

````

````{py:method} se3_exp(v: torch.Tensor) -> torch.Tensor
:canonical: better_robot.backends.protocol.LieOps.se3_exp

```{autodoc2-docstring} better_robot.backends.protocol.LieOps.se3_exp
```

````

````{py:method} se3_act(t: torch.Tensor, p: torch.Tensor) -> torch.Tensor
:canonical: better_robot.backends.protocol.LieOps.se3_act

```{autodoc2-docstring} better_robot.backends.protocol.LieOps.se3_act
```

````

````{py:method} se3_adjoint(t: torch.Tensor) -> torch.Tensor
:canonical: better_robot.backends.protocol.LieOps.se3_adjoint

```{autodoc2-docstring} better_robot.backends.protocol.LieOps.se3_adjoint
```

````

````{py:method} se3_adjoint_inv(t: torch.Tensor) -> torch.Tensor
:canonical: better_robot.backends.protocol.LieOps.se3_adjoint_inv

```{autodoc2-docstring} better_robot.backends.protocol.LieOps.se3_adjoint_inv
```

````

````{py:method} se3_normalize(t: torch.Tensor) -> torch.Tensor
:canonical: better_robot.backends.protocol.LieOps.se3_normalize

```{autodoc2-docstring} better_robot.backends.protocol.LieOps.se3_normalize
```

````

````{py:method} so3_compose(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor
:canonical: better_robot.backends.protocol.LieOps.so3_compose

```{autodoc2-docstring} better_robot.backends.protocol.LieOps.so3_compose
```

````

````{py:method} so3_inverse(q: torch.Tensor) -> torch.Tensor
:canonical: better_robot.backends.protocol.LieOps.so3_inverse

```{autodoc2-docstring} better_robot.backends.protocol.LieOps.so3_inverse
```

````

````{py:method} so3_log(q: torch.Tensor) -> torch.Tensor
:canonical: better_robot.backends.protocol.LieOps.so3_log

```{autodoc2-docstring} better_robot.backends.protocol.LieOps.so3_log
```

````

````{py:method} so3_exp(w: torch.Tensor) -> torch.Tensor
:canonical: better_robot.backends.protocol.LieOps.so3_exp

```{autodoc2-docstring} better_robot.backends.protocol.LieOps.so3_exp
```

````

````{py:method} so3_act(q: torch.Tensor, p: torch.Tensor) -> torch.Tensor
:canonical: better_robot.backends.protocol.LieOps.so3_act

```{autodoc2-docstring} better_robot.backends.protocol.LieOps.so3_act
```

````

````{py:method} so3_to_matrix(q: torch.Tensor) -> torch.Tensor
:canonical: better_robot.backends.protocol.LieOps.so3_to_matrix

```{autodoc2-docstring} better_robot.backends.protocol.LieOps.so3_to_matrix
```

````

````{py:method} so3_from_matrix(R: torch.Tensor) -> torch.Tensor
:canonical: better_robot.backends.protocol.LieOps.so3_from_matrix

```{autodoc2-docstring} better_robot.backends.protocol.LieOps.so3_from_matrix
```

````

````{py:method} so3_normalize(q: torch.Tensor) -> torch.Tensor
:canonical: better_robot.backends.protocol.LieOps.so3_normalize

```{autodoc2-docstring} better_robot.backends.protocol.LieOps.so3_normalize
```

````

`````

`````{py:class} KinematicsOps
:canonical: better_robot.backends.protocol.KinematicsOps

Bases: {py:obj}`typing.Protocol`

```{autodoc2-docstring} better_robot.backends.protocol.KinematicsOps
```

````{py:method} forward_kinematics(model: better_robot.data_model.model.Model, q: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]
:canonical: better_robot.backends.protocol.KinematicsOps.forward_kinematics

```{autodoc2-docstring} better_robot.backends.protocol.KinematicsOps.forward_kinematics
```

````

````{py:method} compute_joint_jacobians(model: better_robot.data_model.model.Model, data: better_robot.data_model.data.Data) -> torch.Tensor
:canonical: better_robot.backends.protocol.KinematicsOps.compute_joint_jacobians

```{autodoc2-docstring} better_robot.backends.protocol.KinematicsOps.compute_joint_jacobians
```

````

`````

`````{py:class} DynamicsOps
:canonical: better_robot.backends.protocol.DynamicsOps

Bases: {py:obj}`typing.Protocol`

```{autodoc2-docstring} better_robot.backends.protocol.DynamicsOps
```

````{py:method} rnea(model: better_robot.data_model.model.Model, data: better_robot.data_model.data.Data, q: torch.Tensor, v: torch.Tensor, a: torch.Tensor) -> torch.Tensor
:canonical: better_robot.backends.protocol.DynamicsOps.rnea

```{autodoc2-docstring} better_robot.backends.protocol.DynamicsOps.rnea
```

````

````{py:method} aba(model: better_robot.data_model.model.Model, data: better_robot.data_model.data.Data, q: torch.Tensor, v: torch.Tensor, tau: torch.Tensor) -> torch.Tensor
:canonical: better_robot.backends.protocol.DynamicsOps.aba

```{autodoc2-docstring} better_robot.backends.protocol.DynamicsOps.aba
```

````

````{py:method} crba(model: better_robot.data_model.model.Model, data: better_robot.data_model.data.Data, q: torch.Tensor) -> torch.Tensor
:canonical: better_robot.backends.protocol.DynamicsOps.crba

```{autodoc2-docstring} better_robot.backends.protocol.DynamicsOps.crba
```

````

````{py:method} center_of_mass(model: better_robot.data_model.model.Model, data: better_robot.data_model.data.Data, q: torch.Tensor, v: torch.Tensor | None = None, a: torch.Tensor | None = None) -> torch.Tensor
:canonical: better_robot.backends.protocol.DynamicsOps.center_of_mass

```{autodoc2-docstring} better_robot.backends.protocol.DynamicsOps.center_of_mass
```

````

`````

`````{py:class} Backend
:canonical: better_robot.backends.protocol.Backend

Bases: {py:obj}`typing.Protocol`

```{autodoc2-docstring} better_robot.backends.protocol.Backend
```

````{py:attribute} name
:canonical: better_robot.backends.protocol.Backend.name
:type: str
:value: >
   None

```{autodoc2-docstring} better_robot.backends.protocol.Backend.name
```

````

````{py:attribute} lie
:canonical: better_robot.backends.protocol.Backend.lie
:type: better_robot.backends.protocol.LieOps
:value: >
   None

```{autodoc2-docstring} better_robot.backends.protocol.Backend.lie
```

````

````{py:attribute} kinematics
:canonical: better_robot.backends.protocol.Backend.kinematics
:type: better_robot.backends.protocol.KinematicsOps
:value: >
   None

```{autodoc2-docstring} better_robot.backends.protocol.Backend.kinematics
```

````

````{py:attribute} dynamics
:canonical: better_robot.backends.protocol.Backend.dynamics
:type: better_robot.backends.protocol.DynamicsOps
:value: >
   None

```{autodoc2-docstring} better_robot.backends.protocol.Backend.dynamics
```

````

`````
