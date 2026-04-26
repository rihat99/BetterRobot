# {py:mod}`better_robot.io`

```{py:module} better_robot.io
```

```{autodoc2-docstring} better_robot.io
:allowtitles:
```

## Subpackages

```{toctree}
:titlesonly:
:maxdepth: 3

better_robot.io.builders
better_robot.io.parsers
```

## Submodules

```{toctree}
:titlesonly:
:maxdepth: 1

better_robot.io.ir
better_robot.io.assets
better_robot.io.build_model
```

## Package Contents

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`register_parser <better_robot.io.register_parser>`
  - ```{autodoc2-docstring} better_robot.io.register_parser
    :summary:
    ```
* - {py:obj}`load <better_robot.io.load>`
  - ```{autodoc2-docstring} better_robot.io.load
    :summary:
    ```
````

### API

````{py:function} register_parser(suffix: str, fn: typing.Callable[..., better_robot.io.ir.IRModel]) -> None
:canonical: better_robot.io.register_parser

```{autodoc2-docstring} better_robot.io.register_parser
```
````

````{py:function} load(source: str | pathlib.Path | typing.Any | typing.Callable[[], better_robot.io.ir.IRModel], *, format: typing.Literal[auto, urdf, mjcf, builder] = 'auto', root_joint: better_robot.data_model.joint_models.base.JointModel | None = None, free_flyer: bool = False, device: torch.device | None = None, dtype: torch.dtype = torch.float32) -> better_robot.data_model.model.Model
:canonical: better_robot.io.load

```{autodoc2-docstring} better_robot.io.load
```
````
