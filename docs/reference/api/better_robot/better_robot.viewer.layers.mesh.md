# {py:mod}`better_robot.viewer.layers.mesh`

```{py:module} better_robot.viewer.layers.mesh
```

```{autodoc2-docstring} better_robot.viewer.layers.mesh
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`MeshLayer <better_robot.viewer.layers.mesh.MeshLayer>`
  - ```{autodoc2-docstring} better_robot.viewer.layers.mesh.MeshLayer
    :summary:
    ```
* - {py:obj}`MeshSpec <better_robot.viewer.layers.mesh.MeshSpec>`
  - ```{autodoc2-docstring} better_robot.viewer.layers.mesh.MeshSpec
    :summary:
    ```
* - {py:obj}`RigidMeshLayer <better_robot.viewer.layers.mesh.RigidMeshLayer>`
  - ```{autodoc2-docstring} better_robot.viewer.layers.mesh.RigidMeshLayer
    :summary:
    ```
* - {py:obj}`SpheresLayer <better_robot.viewer.layers.mesh.SpheresLayer>`
  - ```{autodoc2-docstring} better_robot.viewer.layers.mesh.SpheresLayer
    :summary:
    ```
````

### API

`````{py:class} MeshLayer(name: str, vertices: numpy.ndarray, faces: numpy.ndarray, *, color: better_robot.viewer._math.RGB = (0.8, 0.8, 0.8), opacity: float = 1.0, flat_shading: bool = False)
:canonical: better_robot.viewer.layers.mesh.MeshLayer

Bases: {py:obj}`better_robot.viewer.layer.Layer`

```{autodoc2-docstring} better_robot.viewer.layers.mesh.MeshLayer
```

````{py:method} build(scene: typing.Any, root: str) -> None
:canonical: better_robot.viewer.layers.mesh.MeshLayer.build

````

````{py:method} geometry(frame: int) -> list[better_robot.viewer.layer.Geometry]
:canonical: better_robot.viewer.layers.mesh.MeshLayer.geometry

````

````{py:method} set_frame(frame: int) -> None
:canonical: better_robot.viewer.layers.mesh.MeshLayer.set_frame

````

`````

`````{py:class} MeshSpec
:canonical: better_robot.viewer.layers.mesh.MeshSpec

```{autodoc2-docstring} better_robot.viewer.layers.mesh.MeshSpec
```

````{py:attribute} vertices
:canonical: better_robot.viewer.layers.mesh.MeshSpec.vertices
:type: numpy.ndarray
:value: >
   None

```{autodoc2-docstring} better_robot.viewer.layers.mesh.MeshSpec.vertices
```

````

````{py:attribute} faces
:canonical: better_robot.viewer.layers.mesh.MeshSpec.faces
:type: numpy.ndarray
:value: >
   None

```{autodoc2-docstring} better_robot.viewer.layers.mesh.MeshSpec.faces
```

````

````{py:attribute} body
:canonical: better_robot.viewer.layers.mesh.MeshSpec.body
:type: int
:value: >
   None

```{autodoc2-docstring} better_robot.viewer.layers.mesh.MeshSpec.body
```

````

````{py:attribute} origin
:canonical: better_robot.viewer.layers.mesh.MeshSpec.origin
:type: numpy.ndarray
:value: >
   None

```{autodoc2-docstring} better_robot.viewer.layers.mesh.MeshSpec.origin
```

````

````{py:attribute} color
:canonical: better_robot.viewer.layers.mesh.MeshSpec.color
:type: better_robot.viewer._math.RGB
:value: >
   (0.8, 0.8, 0.8)

```{autodoc2-docstring} better_robot.viewer.layers.mesh.MeshSpec.color
```

````

````{py:attribute} trimesh
:canonical: better_robot.viewer.layers.mesh.MeshSpec.trimesh
:type: typing.Any
:value: >
   None

```{autodoc2-docstring} better_robot.viewer.layers.mesh.MeshSpec.trimesh
```

````

`````

`````{py:class} RigidMeshLayer(name: str, meshes: list[better_robot.viewer.layers.mesh.MeshSpec], body_poses: numpy.ndarray, *, opacity: float = 1.0)
:canonical: better_robot.viewer.layers.mesh.RigidMeshLayer

Bases: {py:obj}`better_robot.viewer.layer.Layer`

```{autodoc2-docstring} better_robot.viewer.layers.mesh.RigidMeshLayer
```

````{py:method} build(scene: typing.Any, root: str) -> None
:canonical: better_robot.viewer.layers.mesh.RigidMeshLayer.build

````

````{py:method} geometry(frame: int) -> list[better_robot.viewer.layer.Geometry]
:canonical: better_robot.viewer.layers.mesh.RigidMeshLayer.geometry

````

````{py:method} set_frame(frame: int) -> None
:canonical: better_robot.viewer.layers.mesh.RigidMeshLayer.set_frame

````

`````

`````{py:class} SpheresLayer(name: str, centers: numpy.ndarray, radii: numpy.ndarray, *, colors: numpy.ndarray | better_robot.viewer._math.RGB = (0.95, 0.45, 0.2), opacity: float = 0.5)
:canonical: better_robot.viewer.layers.mesh.SpheresLayer

Bases: {py:obj}`better_robot.viewer.layer.Layer`

```{autodoc2-docstring} better_robot.viewer.layers.mesh.SpheresLayer
```

````{py:method} build(scene: typing.Any, root: str) -> None
:canonical: better_robot.viewer.layers.mesh.SpheresLayer.build

````

````{py:method} geometry(frame: int) -> list[better_robot.viewer.layer.Geometry]
:canonical: better_robot.viewer.layers.mesh.SpheresLayer.geometry

````

````{py:method} set_frame(frame: int) -> None
:canonical: better_robot.viewer.layers.mesh.SpheresLayer.set_frame

````

`````
