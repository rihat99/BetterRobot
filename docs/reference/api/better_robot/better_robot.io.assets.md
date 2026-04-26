# {py:mod}`better_robot.io.assets`

```{py:module} better_robot.io.assets
```

```{autodoc2-docstring} better_robot.io.assets
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`AssetResolver <better_robot.io.assets.AssetResolver>`
  - ```{autodoc2-docstring} better_robot.io.assets.AssetResolver
    :summary:
    ```
* - {py:obj}`FilesystemResolver <better_robot.io.assets.FilesystemResolver>`
  - ```{autodoc2-docstring} better_robot.io.assets.FilesystemResolver
    :summary:
    ```
* - {py:obj}`PackageResolver <better_robot.io.assets.PackageResolver>`
  - ```{autodoc2-docstring} better_robot.io.assets.PackageResolver
    :summary:
    ```
* - {py:obj}`CompositeResolver <better_robot.io.assets.CompositeResolver>`
  - ```{autodoc2-docstring} better_robot.io.assets.CompositeResolver
    :summary:
    ```
* - {py:obj}`CachedDownloadResolver <better_robot.io.assets.CachedDownloadResolver>`
  - ```{autodoc2-docstring} better_robot.io.assets.CachedDownloadResolver
    :summary:
    ```
````

### API

`````{py:class} AssetResolver
:canonical: better_robot.io.assets.AssetResolver

Bases: {py:obj}`typing.Protocol`

```{autodoc2-docstring} better_robot.io.assets.AssetResolver
```

````{py:method} resolve(uri: str) -> pathlib.Path
:canonical: better_robot.io.assets.AssetResolver.resolve

```{autodoc2-docstring} better_robot.io.assets.AssetResolver.resolve
```

````

`````

`````{py:class} FilesystemResolver(base_path: pathlib.Path | str)
:canonical: better_robot.io.assets.FilesystemResolver

```{autodoc2-docstring} better_robot.io.assets.FilesystemResolver
```

````{py:method} resolve(uri: str) -> pathlib.Path
:canonical: better_robot.io.assets.FilesystemResolver.resolve

```{autodoc2-docstring} better_robot.io.assets.FilesystemResolver.resolve
```

````

`````

`````{py:class} PackageResolver(packages: dict[str, pathlib.Path | str])
:canonical: better_robot.io.assets.PackageResolver

```{autodoc2-docstring} better_robot.io.assets.PackageResolver
```

````{py:method} resolve(uri: str) -> pathlib.Path
:canonical: better_robot.io.assets.PackageResolver.resolve

```{autodoc2-docstring} better_robot.io.assets.PackageResolver.resolve
```

````

`````

`````{py:class} CompositeResolver(resolvers: list[better_robot.io.assets.AssetResolver])
:canonical: better_robot.io.assets.CompositeResolver

```{autodoc2-docstring} better_robot.io.assets.CompositeResolver
```

````{py:method} resolve(uri: str) -> pathlib.Path
:canonical: better_robot.io.assets.CompositeResolver.resolve

```{autodoc2-docstring} better_robot.io.assets.CompositeResolver.resolve
```

````

`````

`````{py:class} CachedDownloadResolver(cache_dir: pathlib.Path | str)
:canonical: better_robot.io.assets.CachedDownloadResolver

```{autodoc2-docstring} better_robot.io.assets.CachedDownloadResolver
```

````{py:method} resolve(uri: str) -> pathlib.Path
:canonical: better_robot.io.assets.CachedDownloadResolver.resolve

```{autodoc2-docstring} better_robot.io.assets.CachedDownloadResolver.resolve
```

````

`````
