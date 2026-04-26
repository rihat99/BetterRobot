# Releasing BetterRobot

The release workflow follows `docs/conventions/20_PACKAGING.md §5`.

## 1. Pre-flight

* `git checkout main && git pull`
* Confirm `pytest tests/` is green.
* Confirm `uv run pytest tests/contract/` is green with `BR_STRICT=1`.
* Confirm bench-cpu numbers are within the 5% noise floor against the
  baseline (`tests/bench/baseline_cpu.json`).

## 2. Bump the version

* Update `src/better_robot/_version.py::__version__`. Keep it in step
  with `pyproject.toml::project.version` (single source of truth — the
  test suite imports `better_robot.__version__` to verify).
* Update `docs/CHANGELOG.md` — list the proposals in
  `docs/claude_plan/accepted/` that landed in this cycle.
* Update `docs/site/reference/changelog.md` to point at the new entry.

## 3. Tag and push

```
git commit -am "release: vX.Y.Z"
git tag vX.Y.Z
git push --tags
```

## 4. CI gates

GitHub Actions runs the full matrix:
* `contract` (blocks all) — strict mode (`BR_STRICT=1`),
* `unit` on Python 3.10/3.11/3.12,
* `cuda` on the self-hosted runner,
* `bench-cpu-advisory` (continues on error),
* `docs` — `sphinx-build -W` over `docs/site/`.

## 5. Build and publish

```
uv sync --extra dev
uv build              # produces wheel + sdist under dist/
uv publish            # to PyPI; needs PYPI_TOKEN
```

Sanity-check the wheel install before publishing:
```
pip install dist/better_robot-X.Y.Z-py3-none-any.whl
python -c "import better_robot; print(better_robot.__version__)"
```

The lean-core install must satisfy:
```
pip install better-robot           # only torch + numpy + rich
python -c "import better_robot; better_robot.load('foo.urdf')"
# → BackendNotAvailableError pointing to pip install better-robot[urdf]
```

## 6. Post-release

* Move closed work items from `docs/status/18_ROADMAP.md` to
  `CHANGELOG.md` under the landing release.
* Open a tracking issue for the next milestone.
