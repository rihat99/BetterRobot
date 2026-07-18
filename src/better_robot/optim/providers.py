"""Lazy provider helpers and an evaluation-local read-only memo."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Protocol, runtime_checkable

from ..data_model.model import Model
from ..kinematics.forward import forward_kinematics


@runtime_checkable
class Provider(Protocol):
    name: str
    reads: tuple[str, ...]
    outputs: tuple[str, ...]

    def __call__(self, ctx: Mapping[str, Any]) -> dict[str, Any]: ...


class EvaluationContext(Mapping[str, Any]):
    __slots__ = (
        "_values",
        "_providers_by_output",
        "_memo",
        "_resolving",
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
        self._memo: dict[str, Any] = {}
        self._resolving: set[str] = set()
        self._free_indices = MappingProxyType(dict(free_indices))
        self._temporal_free_indices = MappingProxyType(dict(temporal_free_indices or {}))

    def __getitem__(self, key: str) -> Any:
        if key in self._values:
            return self._values[key]
        if key in self._memo:
            return self._memo[key]
        provider = self._providers_by_output.get(key)
        if provider is None:
            raise KeyError(key)
        if provider.name in self._resolving:
            raise RuntimeError(f"provider dependency cycle while resolving {provider.name!r}")
        self._resolving.add(provider.name)
        try:
            for name in provider.reads:
                self[name]
            produced = provider(self)
        finally:
            self._resolving.remove(provider.name)
        if set(produced) != set(provider.outputs):
            raise ValueError(
                f"Provider {provider.name!r} declared outputs {provider.outputs} but returned {tuple(produced)}"
            )
        self._memo.update(produced)
        return self._memo[key]

    def __iter__(self) -> Iterator[str]:
        yield from self._values
        yield from (key for key in self._providers_by_output if key not in self._values)

    def __len__(self) -> int:
        return len(self._values) + len(self._providers_by_output)

    def free_indices(self, variable: str):
        return self._free_indices[variable]

    def temporal_free_indices(self, variable: str):
        try:
            return self._temporal_free_indices[variable]
        except KeyError as exc:
            raise KeyError(f"variable {variable!r} has no separable temporal tangent layout") from exc


@dataclass(frozen=True)
class RobotStateProvider:
    model: Model
    var: str = "q"
    output: str = "data"
    name: str = "robot_state"

    @property
    def reads(self) -> tuple[str, ...]:
        return (self.var,)

    @property
    def outputs(self) -> tuple[str, ...]:
        return (self.output,)

    def __call__(self, ctx: Mapping[str, Any]) -> dict[str, Any]:
        return {self.output: forward_kinematics(self.model, ctx[self.var], compute_frames=True)}
