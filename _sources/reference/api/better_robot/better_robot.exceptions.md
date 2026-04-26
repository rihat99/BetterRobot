# {py:mod}`better_robot.exceptions`

```{py:module} better_robot.exceptions
```

```{autodoc2-docstring} better_robot.exceptions
:allowtitles:
```

## Module Contents

### API

````{py:exception} BetterRobotError()
:canonical: better_robot.exceptions.BetterRobotError

Bases: {py:obj}`Exception`

```{autodoc2-docstring} better_robot.exceptions.BetterRobotError
```

````

````{py:exception} ModelInconsistencyError()
:canonical: better_robot.exceptions.ModelInconsistencyError

Bases: {py:obj}`better_robot.exceptions.BetterRobotError`, {py:obj}`ValueError`

```{autodoc2-docstring} better_robot.exceptions.ModelInconsistencyError
```

````

````{py:exception} DeviceMismatchError()
:canonical: better_robot.exceptions.DeviceMismatchError

Bases: {py:obj}`better_robot.exceptions.BetterRobotError`, {py:obj}`ValueError`

```{autodoc2-docstring} better_robot.exceptions.DeviceMismatchError
```

````

````{py:exception} DtypeMismatchError()
:canonical: better_robot.exceptions.DtypeMismatchError

Bases: {py:obj}`better_robot.exceptions.BetterRobotError`, {py:obj}`ValueError`

```{autodoc2-docstring} better_robot.exceptions.DtypeMismatchError
```

````

````{py:exception} QuaternionNormError()
:canonical: better_robot.exceptions.QuaternionNormError

Bases: {py:obj}`better_robot.exceptions.BetterRobotError`, {py:obj}`ValueError`

```{autodoc2-docstring} better_robot.exceptions.QuaternionNormError
```

````

````{py:exception} ShapeError()
:canonical: better_robot.exceptions.ShapeError

Bases: {py:obj}`better_robot.exceptions.BetterRobotError`, {py:obj}`ValueError`

```{autodoc2-docstring} better_robot.exceptions.ShapeError
```

````

````{py:exception} ConvergenceError()
:canonical: better_robot.exceptions.ConvergenceError

Bases: {py:obj}`better_robot.exceptions.BetterRobotError`, {py:obj}`RuntimeError`

```{autodoc2-docstring} better_robot.exceptions.ConvergenceError
```

````

````{py:exception} BackendNotAvailableError()
:canonical: better_robot.exceptions.BackendNotAvailableError

Bases: {py:obj}`better_robot.exceptions.BetterRobotError`, {py:obj}`ImportError`

```{autodoc2-docstring} better_robot.exceptions.BackendNotAvailableError
```

````

````{py:exception} IRSchemaVersionError()
:canonical: better_robot.exceptions.IRSchemaVersionError

Bases: {py:obj}`better_robot.exceptions.BetterRobotError`, {py:obj}`ValueError`

```{autodoc2-docstring} better_robot.exceptions.IRSchemaVersionError
```

````

````{py:exception} StaleCacheError()
:canonical: better_robot.exceptions.StaleCacheError

Bases: {py:obj}`better_robot.exceptions.BetterRobotError`, {py:obj}`RuntimeError`

```{autodoc2-docstring} better_robot.exceptions.StaleCacheError
```

````

````{py:exception} UnsupportedJointError()
:canonical: better_robot.exceptions.UnsupportedJointError

Bases: {py:obj}`better_robot.exceptions.BetterRobotError`, {py:obj}`ValueError`

```{autodoc2-docstring} better_robot.exceptions.UnsupportedJointError
```

````

````{py:exception} SingularityWarning()
:canonical: better_robot.exceptions.SingularityWarning

Bases: {py:obj}`UserWarning`

```{autodoc2-docstring} better_robot.exceptions.SingularityWarning
```

````
