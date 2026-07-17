"""Lazy provider DAG and evaluation-local read-only context."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Protocol, runtime_checkable

from ...data_model.model import Model
from ...kinematics.forward import forward_kinematics


@runtime_checkable
class Provider(Protocol):
    """Structural protocol for a lazily evaluated context producer."""

    name: str
    inputs: tuple[str, ...]
    outputs: tuple[str, ...]

    def __call__(self, ctx: Mapping[str, Any]) -> dict[str, Any]: ...


class EvaluationContext(Mapping[str, Any]):
    """Read-only values plus lazy provider outputs for exactly one evaluation."""

    __slots__ = (
        "_values",
        "_providers_by_output",
        "_provider_cache",
        "_running",
        "_free_indices",
        "_temporal_free_indices",
        "__weakref__",
    )

    def __init__(
        self,
        values: Mapping[str, Any],
        providers_by_output: Mapping[str, Provider],
        free_indices: Mapping[str, Any],
        temporal_free_indices: Mapping[str, Any] | None = None,
    ) -> None:
        self._values = MappingProxyType(dict(values))
        self._providers_by_output = MappingProxyType(dict(providers_by_output))
        self._provider_cache: dict[str, Any] = {}
        self._running: set[str] = set()
        self._free_indices = MappingProxyType(dict(free_indices))
        self._temporal_free_indices = MappingProxyType(dict(temporal_free_indices or {}))

    def __getitem__(self, key: str) -> Any:
        if key in self._values:
            return self._values[key]
        if key in self._provider_cache:
            return self._provider_cache[key]
        provider = self._providers_by_output.get(key)
        if provider is None:
            raise KeyError(key)
        if provider.name in self._running:  # construction rejects cycles; defensive only
            raise RuntimeError(f"provider {provider.name!r} recursively requested itself")
        self._running.add(provider.name)
        try:
            for input_name in provider.inputs:
                self[input_name]
            produced = provider(self.restrict(provider.inputs))
        finally:
            self._running.remove(provider.name)
        if set(produced) != set(provider.outputs):
            raise ValueError(
                f"Provider {provider.name!r} declared outputs {provider.outputs} but returned {tuple(produced)}"
            )
        self._provider_cache.update(produced)
        return self._provider_cache[key]

    def __iter__(self) -> Iterator[str]:
        yield from self._values
        for key in self._providers_by_output:
            if key not in self._values:
                yield key

    def __len__(self) -> int:
        return len(self._values) + len(self._providers_by_output)

    def free_indices(self, variable: str):
        """Return the reduced-column convention for an analytic block author."""
        return self._free_indices[variable]

    def temporal_free_indices(self, variable: str):
        """Return per-knot local free coordinates for a temporal block hook."""
        try:
            return self._temporal_free_indices[variable]
        except KeyError as exc:
            raise KeyError(f"variable {variable!r} has no separable temporal tangent layout") from exc

    def restrict(self, reads: tuple[str, ...]) -> _ItemContext:
        """Expose only one item's declared dependencies."""
        return _ItemContext(self, reads)


class _ItemContext(Mapping[str, Any]):
    """Read-only view that makes undeclared dependencies fail fast."""

    __slots__ = ("_context", "_reads", "__weakref__")

    def __init__(self, context: EvaluationContext, reads: tuple[str, ...]) -> None:
        self._context = context
        self._reads = reads

    def __getitem__(self, key: str) -> Any:
        if key not in self._reads:
            raise KeyError(f"context name {key!r} was not declared in this item's reads={self._reads!r}")
        return self._context[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._reads)

    def __len__(self) -> int:
        return len(self._reads)

    def free_indices(self, variable: str):
        return self._context.free_indices(variable)

    def temporal_free_indices(self, variable: str):
        return self._context.temporal_free_indices(variable)


@dataclass(frozen=True)
class RobotStateProvider:
    """Compute robot FK once when an active consumer requests its output."""

    model: Model
    var: str = "q"
    output: str = "data"
    name: str = "robot_state"

    @property
    def inputs(self) -> tuple[str, ...]:
        return (self.var,)

    @property
    def outputs(self) -> tuple[str, ...]:
        return (self.output,)

    def __call__(self, ctx: Mapping[str, Any]) -> dict[str, Any]:
        return {
            self.output: forward_kinematics(
                self.model,
                ctx[self.var],
                compute_frames=True,
            )
        }
