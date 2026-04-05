"""Registry for named solver components."""
from __future__ import annotations


class Registry:
    """Generic registry for named components.

    Usage::

        SOLVERS = Registry("solvers")

        @SOLVERS.register("lm")
        class LevenbergMarquardt(Solver):
            ...

        solver_cls = SOLVERS.get("lm")
        solver = solver_cls()
    """

    def __init__(self, name: str) -> None:
        self._name = name
        self._entries: dict[str, type] = {}

    def register(self, name: str):
        """Decorator to register a class under a name."""
        def wrapper(cls):
            self._entries[name] = cls
            return cls
        return wrapper

    def get(self, name: str) -> type:
        """Retrieve a registered class by name."""
        if name not in self._entries:
            raise KeyError(
                f"'{name}' not in {self._name} registry. "
                f"Available: {list(self._entries.keys())}"
            )
        return self._entries[name]

    def list(self) -> list[str]:
        """Return all registered names."""
        return list(self._entries.keys())

    def __contains__(self, name: str) -> bool:
        return name in self._entries


SOLVERS: Registry = Registry("solvers")
"""Global solver registry. Use SOLVERS.get('lm')() to instantiate."""
