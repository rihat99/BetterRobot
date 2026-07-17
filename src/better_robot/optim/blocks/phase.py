"""Functional phase orchestration for immutable named-block problems.

Each phase rebuilds a cheap static :class:`Problem` view with weight and
mask overrides, then discards it after the solve.  The caller's problem is
therefore unchanged even when a phase raises; no mutable snapshot protocol or
second provider cache is needed.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, fields, is_dataclass, replace
from numbers import Real
from types import MappingProxyType
from typing import Any, NamedTuple

import torch

from .problem import Problem, Weight
from .variables import Values


class PhaseResult(NamedTuple):
    """Final values plus one fresh solver state per phase.

    A zero-iteration phase contributes ``None`` after still running its
    ``on_start`` hook.
    """

    values: Values
    states: tuple[Any | None, ...]


@dataclass(frozen=True)
class Phase:
    """One solver segment with functional weight and free-DOF overrides.

    Mask entries follow :class:`VarSpec`: nonzero means free.  A phase mask is
    intersected with the base mask, so staging can never unfreeze a coordinate
    that the original problem fixed permanently.  ``on_start`` is a host-side
    zero-argument hook and runs exactly once, including for ``iters=0``.
    """

    name: str
    iters: int
    optimizer: Any
    weight_overrides: Mapping[str, Weight] = field(default_factory=dict)
    mask_overrides: Mapping[str, torch.Tensor] = field(default_factory=dict)
    on_start: Callable[[], None] | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("Phase name must be a non-empty string")
        if isinstance(self.iters, bool) or not isinstance(self.iters, int) or self.iters < 0:
            raise ValueError("Phase iters must be a non-negative int")
        if not is_dataclass(self.optimizer) or "max_iter" not in {item.name for item in fields(self.optimizer)}:
            raise TypeError("Phase optimizer must be a dataclass solver with a max_iter field")
        for method in ("init_state", "update", "run"):
            if not callable(getattr(self.optimizer, method, None)):
                raise TypeError(f"Phase optimizer must provide callable {method}()")
        if self.on_start is not None and not callable(self.on_start):
            raise TypeError("Phase on_start must be callable or None")

        weights = dict(self.weight_overrides)
        if any(not isinstance(name, str) or not name for name in weights):
            raise TypeError("Phase weight override names must be non-empty strings")
        for name, weight in weights.items():
            if not isinstance(weight, (Real, torch.Tensor)):
                raise TypeError(f"Phase weight override {name!r} must be a real number or tensor")
        masks = dict(self.mask_overrides)
        if any(not isinstance(name, str) or not name for name in masks):
            raise TypeError("Phase mask override names must be non-empty strings")
        if any(not isinstance(mask, torch.Tensor) for mask in masks.values()):
            raise TypeError("Phase mask overrides must be tensors")
        object.__setattr__(self, "weight_overrides", MappingProxyType(weights))
        object.__setattr__(self, "mask_overrides", MappingProxyType(masks))


def _phase_problem(problem: Problem, phase: Phase) -> Problem:
    item_names = {item.name for item in (*problem.residuals, *problem.objectives)}
    unknown_weights = set(phase.weight_overrides) - item_names
    if unknown_weights:
        raise ValueError(f"Phase {phase.name!r} has unknown weight overrides {sorted(unknown_weights)}")
    variable_names = {spec.name for spec in problem.vars}
    unknown_masks = set(phase.mask_overrides) - variable_names
    if unknown_masks:
        raise ValueError(f"Phase {phase.name!r} has unknown mask overrides {sorted(unknown_masks)}")

    phase_vars = []
    for spec in problem.vars:
        override = phase.mask_overrides.get(spec.name)
        if override is None:
            phase_vars.append(spec)
            continue
        if tuple(override.shape) != (spec.tangent_dim,):
            raise ValueError(
                f"Phase {phase.name!r} mask for {spec.name!r} must have tangent "
                f"shape ({spec.tangent_dim},), got {tuple(override.shape)}"
            )
        enabled = override.to(dtype=torch.bool)
        if spec.mask is not None:
            enabled = enabled.to(device=spec.mask.device) & spec.mask.to(dtype=torch.bool)
        phase_vars.append(replace(spec, mask=enabled))

    residuals = tuple(
        replace(item, weight=phase.weight_overrides.get(item.name, item.weight)) for item in problem.residuals
    )
    objectives = tuple(
        replace(item, weight=phase.weight_overrides.get(item.name, item.weight)) for item in problem.objectives
    )
    return Problem(
        vars=tuple(phase_vars),
        residuals=residuals,
        objectives=objectives,
        providers=problem.providers,
        parameters=problem.parameters,
        differentiable_parameters=tuple(problem.parameter_gradients),
    )


def run_phases(
    problem: Problem,
    values: Mapping[str, torch.Tensor],
    phases: Sequence[Phase],
) -> PhaseResult:
    """Run fresh solver state in each phase and carry only accepted values.

    Phase-local problems are functional rebuilds, so the input ``problem`` is
    never modified and exception restoration is automatic.
    """

    if not phases:
        problem._validate_values(values)
        return PhaseResult(dict(values), ())
    current = dict(values)
    states: list[Any | None] = []
    for phase in phases:
        if not isinstance(phase, Phase):
            raise TypeError("run_phases expects a sequence of Phase values")
        # Validate every override even for hook-only phases; a zero iteration
        # budget must not turn misspelled item/block names into silent no-ops.
        local_problem = _phase_problem(problem, phase)
        if phase.on_start is not None:
            phase.on_start()
        if phase.iters == 0:
            states.append(None)
            continue
        optimizer = replace(phase.optimizer, max_iter=phase.iters)
        current, state = optimizer.run(current, local_problem)
        states.append(state)
    return PhaseResult(current, tuple(states))


__all__ = ["Phase", "PhaseResult", "run_phases"]
