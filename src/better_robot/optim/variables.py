"""Named variable blocks and their retraction geometry."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from functools import reduce
from itertools import count
from numbers import Real
from operator import mul
from types import MappingProxyType

import torch

from ..exceptions import DeviceMismatchError, DtypeMismatchError
from ..data_model.model import Model
from ..lie import se3, so3


_VARIABLE_COUNTERS: dict[type, count] = {}


@dataclass(frozen=True)
class Bounds:
    """Elementwise lower and upper state-coordinate bounds."""

    lower: torch.Tensor
    upper: torch.Tensor

    def __post_init__(self) -> None:
        if not isinstance(self.lower, torch.Tensor) or not isinstance(self.upper, torch.Tensor):
            raise TypeError("Bounds lower and upper must be torch.Tensor values")
        if self.lower.shape != self.upper.shape:
            raise ValueError(
                f"Bounds lower/upper shapes must match, got {tuple(self.lower.shape)} and {tuple(self.upper.shape)}"
            )
        if self.lower.dtype != self.upper.dtype or self.lower.device != self.upper.device:
            raise ValueError("Bounds lower and upper must have the same dtype and device")
        if not self.lower.is_floating_point():
            raise TypeError("Bounds lower and upper must use a floating dtype")
        if torch.isnan(self.lower).any() or torch.isnan(self.upper).any():
            raise ValueError("Bounds lower and upper must not contain NaN")
        if bool((self.lower > self.upper).any()):
            raise ValueError("Bounds lower must be <= upper at every state coordinate")


def _auto_name(cls: type) -> str:
    counter = _VARIABLE_COUNTERS.setdefault(cls, count())
    return f"{cls.__name__.lower()}_{next(counter)}"


def _numel(shape: tuple[int, ...]) -> int:
    return reduce(mul, shape, 1)


class Variable:
    """A Euclidean tensor variable with an owned value and tangent layout.

    Bool and integer tensors are supported only as non-trainable static inputs.
    """

    _feature_width: int | None = None

    def __init__(  # noqa: PLR0912, PLR0915 - validates one structural public boundary
        self,
        tensor: torch.Tensor,
        *,
        name: str | None = None,
        trainable: bool = True,
        bounds: Bounds | None = None,
        batch_ndim: int = 0,
        time_axis: int | None = None,
    ) -> None:
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"tensor must be a torch.Tensor, got {type(tensor).__name__}")
        if not isinstance(trainable, bool):
            raise TypeError(f"trainable must be a bool, got {type(trainable).__name__}")
        if tensor.is_complex() or (tensor.is_floating_point() and tensor.dtype not in (torch.float32, torch.float64)):
            raise DtypeMismatchError(f"tensor must use torch.float32 or torch.float64, got dtype={tensor.dtype}")
        if trainable and not tensor.is_floating_point():
            raise DtypeMismatchError(
                f"trainable tensor must use torch.float32 or torch.float64, got dtype={tensor.dtype}"
            )
        if name is None:
            name = _auto_name(type(self))
        if not isinstance(name, str) or not name:
            raise ValueError(f"name must be a non-empty string, got {name!r}")
        if isinstance(batch_ndim, bool) or not isinstance(batch_ndim, int) or batch_ndim < 0:
            raise ValueError(f"batch_ndim must be a non-negative int, got {batch_ndim!r}")
        if batch_ndim > tensor.ndim:
            raise ValueError(f"batch_ndim must not exceed tensor.ndim={tensor.ndim}, got {batch_ndim}")
        if time_axis is not None and (isinstance(time_axis, bool) or not isinstance(time_axis, int)):
            raise TypeError(f"time_axis must be an int or None, got {time_axis!r}")
        if time_axis not in (None, 0):
            raise ValueError(f"time_axis must identify the leading event axis 0, got {time_axis}")

        width = self._feature_width
        if width is None:
            event_shape = tuple(tensor.shape[batch_ndim:])
            inferred_batch_ndim = batch_ndim
            if time_axis is not None and not event_shape:
                raise ValueError("time_axis=0 requires a non-scalar event shape")
        else:
            event_ndim = 2 if time_axis is not None else 1
            if tensor.ndim < event_ndim:
                expected = f"(..., T, {width})" if time_axis is not None else f"(..., {width})"
                raise ValueError(f"tensor must have shape {expected}, got {tuple(tensor.shape)}")
            event_shape = tuple(tensor.shape[-event_ndim:])
            inferred_batch_ndim = tensor.ndim - event_ndim
            if event_shape[-1] != width:
                raise ValueError(f"tensor must end in feature width {width}, got {tuple(tensor.shape)}")
            if time_axis is not None and event_shape[0] <= 0:
                raise ValueError("time_axis=0 requires at least one timestep")
            if batch_ndim not in (0, inferred_batch_ndim):
                raise ValueError(
                    f"batch_ndim must match the {inferred_batch_ndim} inferred leading batch axes, got {batch_ndim}"
                )

        self.name = name
        self.trainable = trainable
        self.tensor = tensor
        self.shape = event_shape
        self.batch_ndim = inferred_batch_ndim
        self.time_axis = time_axis
        self.bounds = self._normalize_bounds(bounds)
        self._validate_bounds(self.bounds)

        self.validate_value(tensor)
        self._free_indices = (
            torch.arange(self.tangent_dim(), dtype=torch.long, device=tensor.device)
            if tensor.is_floating_point()
            else torch.empty(0, dtype=torch.long, device=tensor.device)
        )

    def _normalize_bounds(self, bounds: Bounds | None) -> Bounds | None:
        if bounds is not None and not isinstance(bounds, Bounds):
            raise TypeError(f"bounds must be Bounds or None, got {type(bounds).__name__}")
        return bounds

    def _validate_bounds(self, bounds: Bounds | None) -> None:
        if bounds is not None and tuple(bounds.lower.shape) != self.shape:
            raise ValueError(f"bounds must have event shape {self.shape}, got {tuple(bounds.lower.shape)}")

    def tangent_dim(self) -> int:
        """Return the full tangent width of one event."""
        assert self.tensor.is_floating_point(), "non-floating static Variables have no tangent"
        return _numel(self.shape)

    @property
    def free_dim(self) -> int:
        assert self.tensor.is_floating_point(), "non-floating static Variables have no tangent"
        return self._free_indices.numel()

    @property
    def free_indices(self) -> torch.Tensor:
        """Return the immutable construction-time free tangent indices."""
        assert self.tensor.is_floating_point(), "non-floating static Variables have no tangent"
        return self._free_indices.clone()

    @property
    def batch_shape(self) -> tuple[int, ...]:
        return self.batch_shape_of(self.tensor)

    def batch_shape_of(self, value: torch.Tensor) -> tuple[int, ...]:
        self.validate_value(value)
        return tuple(value.shape[: value.ndim - len(self.shape)]) if self.shape else tuple(value.shape)

    @property
    def time_length(self) -> int:
        if self.time_axis is None:
            raise ValueError(f"Variable {self.name!r} has no time_axis")
        return self.shape[0]

    @property
    def temporal_tangent_width(self) -> int:
        return self.free_dim // self.time_length

    @property
    def temporal_free_indices(self) -> torch.Tensor:
        """Return per-knot full-tangent indices retained by a trajectory."""
        full_width = self.tangent_dim() // self.time_length
        return self._free_indices[self._free_indices < full_width].clone()

    def validate_value(self, value: torch.Tensor) -> None:
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"Value for Variable {self.name!r} must be a torch.Tensor")
        if value.is_complex() or (value.is_floating_point() and value.dtype not in (torch.float32, torch.float64)):
            raise DtypeMismatchError(
                f"Value for Variable {self.name!r} must use torch.float32 or torch.float64, got {value.dtype}"
            )
        if self.trainable and not value.is_floating_point():
            raise DtypeMismatchError(
                f"Value for trainable Variable {self.name!r} must use torch.float32 or torch.float64, got {value.dtype}"
            )
        trailing = tuple(value.shape[-len(self.shape) :]) if self.shape else ()
        if value.ndim < len(self.shape) or trailing != self.shape:
            raise ValueError(
                f"Value for Variable {self.name!r} must end in event shape {self.shape}, got {tuple(value.shape)}"
            )
        if self.bounds is not None and (
            value.dtype != self.bounds.lower.dtype or value.device != self.bounds.lower.device
        ):
            raise ValueError(
                f"Value and bounds for Variable {self.name!r} must share dtype/device, "
                f"got {value.dtype}/{value.device} and "
                f"{self.bounds.lower.dtype}/{self.bounds.lower.device}"
            )

    def _tangent_event_shape(self) -> tuple[int, ...]:
        return self.shape

    def gather_tangent(self, full: torch.Tensor) -> torch.Tensor:
        """Gather a full flattened tangent into free coordinates."""
        if not isinstance(full, torch.Tensor):
            raise TypeError(f"full must be a torch.Tensor, got {type(full).__name__}")
        if full.ndim == 0 or full.shape[-1] != self.tangent_dim():
            raise ValueError(f"full tangent must end in {self.tangent_dim()}, got {tuple(full.shape)}")
        if self.free_dim == self.tangent_dim():
            return full
        return full.index_select(-1, self._free_indices.to(full.device))

    def expand_tangent(self, reduced: torch.Tensor) -> torch.Tensor:
        """Scatter free coordinates into a zero-filled full tangent."""
        if not isinstance(reduced, torch.Tensor):
            raise TypeError(f"reduced must be a torch.Tensor, got {type(reduced).__name__}")
        if reduced.ndim == 0 or reduced.shape[-1] != self.free_dim:
            raise ValueError(f"reduced tangent must end in {self.free_dim}, got {tuple(reduced.shape)}")
        if self.free_dim == self.tangent_dim():
            return reduced
        full = reduced.new_zeros(*reduced.shape[:-1], self.tangent_dim())
        if self.free_dim:
            full.index_copy_(-1, self._free_indices.to(reduced.device), reduced)
        return full

    def _retract_full(self, value: torch.Tensor, delta: torch.Tensor) -> torch.Tensor:
        return value + delta

    def _difference_full(self, x0: torch.Tensor, x1: torch.Tensor) -> torch.Tensor:
        return x1 - x0

    def _project_full(self, value: torch.Tensor) -> torch.Tensor:
        if self.bounds is None:
            return value
        return torch.maximum(torch.minimum(value, self.bounds.upper), self.bounds.lower)

    def project(self, value: torch.Tensor) -> torch.Tensor:
        """Project a value onto this variable's valid representation."""
        return self._project_full(value)

    def retract(self, delta: torch.Tensor) -> torch.Tensor:
        """Retract ``delta`` from the current value without assigning it."""
        value = self.tensor
        assert value.is_floating_point(), "non-floating static Variables cannot retract"
        batch = self.batch_shape_of(value)
        if not isinstance(delta, torch.Tensor):
            raise TypeError(f"delta must be a torch.Tensor, got {type(delta).__name__}")
        if delta.dtype != value.dtype:
            raise DtypeMismatchError(f"delta and value must share dtype, got {delta.dtype} and {value.dtype}")
        if delta.device != value.device:
            raise DeviceMismatchError(f"delta and value must share device, got {delta.device} and {value.device}")
        expected = (*batch, self.free_dim)
        if tuple(delta.shape) != expected:
            raise ValueError(f"delta must have shape {expected}, got {tuple(delta.shape)}")
        full = self.expand_tangent(delta).reshape((*batch, *self._tangent_event_shape()))
        return self.project(self._retract_full(value, full))

    def _retract_from(self, value: torch.Tensor, delta: torch.Tensor) -> torch.Tensor:
        original = self.tensor
        self.tensor = value
        try:
            return self.retract(delta)
        finally:
            self.tensor = original

    def difference(self, other: torch.Tensor) -> torch.Tensor:
        """Return the tangent displacement from ``other`` to the current value."""
        value = self.tensor
        assert value.is_floating_point(), "non-floating static Variables have no tangent difference"
        batch = self.batch_shape_of(other)
        self.validate_value(value)
        if tuple(other.shape) != tuple(value.shape):
            raise ValueError(
                f"difference inputs must have equal shape, got {tuple(other.shape)} and {tuple(value.shape)}"
            )
        if other.dtype != value.dtype:
            raise DtypeMismatchError(f"difference inputs must share dtype, got {other.dtype} and {value.dtype}")
        if other.device != value.device:
            raise DeviceMismatchError(f"difference inputs must share device, got {other.device} and {value.device}")
        return self._difference_full(other, value).reshape(*batch, self.tangent_dim())

    def _difference_from(self, x0: torch.Tensor, x1: torch.Tensor) -> torch.Tensor:
        original = self.tensor
        self.tensor = x1
        try:
            return self.difference(x0)
        finally:
            self.tensor = original


_GROUP_BOUNDS_ERROR = (
    "{kind} variables have no meaningful global box bound. Express rotation "
    "limits as residuals or use RobotVariable with joint limits; got "
    "bounds={bounds!r} on Variable {name!r}."
)


class SO3Variable(Variable):
    """A unit-quaternion variable with a three-dimensional tangent."""

    _feature_width = 4

    def _validate_bounds(self, bounds: Bounds | None) -> None:
        if bounds is not None:
            raise ValueError(_GROUP_BOUNDS_ERROR.format(kind="SO3", bounds=bounds, name=self.name))

    def tangent_dim(self) -> int:
        assert self.tensor.is_floating_point(), "non-floating static Variables have no tangent"
        return _numel(self.shape[:-1]) * 3

    def _tangent_event_shape(self) -> tuple[int, ...]:
        return (*self.shape[:-1], 3)

    def _retract_full(self, value: torch.Tensor, delta: torch.Tensor) -> torch.Tensor:
        return so3.normalize(so3.compose(value, so3.exp(delta)))

    def _difference_full(self, x0: torch.Tensor, x1: torch.Tensor) -> torch.Tensor:
        return so3.log(so3.normalize(so3.compose(so3.inverse(x0), x1)))

    def _project_full(self, value: torch.Tensor) -> torch.Tensor:
        return so3.normalize(value)


class SE3Variable(Variable):
    """An SE(3) pose variable with a six-dimensional tangent."""

    _feature_width = 7

    def _validate_bounds(self, bounds: Bounds | None) -> None:
        if bounds is not None:
            raise ValueError(_GROUP_BOUNDS_ERROR.format(kind="SE3", bounds=bounds, name=self.name))

    def tangent_dim(self) -> int:
        assert self.tensor.is_floating_point(), "non-floating static Variables have no tangent"
        return _numel(self.shape[:-1]) * 6

    def _tangent_event_shape(self) -> tuple[int, ...]:
        return (*self.shape[:-1], 6)

    def _retract_full(self, value: torch.Tensor, delta: torch.Tensor) -> torch.Tensor:
        return se3.normalize(se3.compose(value, se3.exp(delta)))

    def _difference_full(self, x0: torch.Tensor, x1: torch.Tensor) -> torch.Tensor:
        return se3.log(se3.compose(se3.inverse(x0), x1))

    def _project_full(self, value: torch.Tensor) -> torch.Tensor:
        return se3.normalize(value)


@dataclass(frozen=True)
class _JointCoordinateLayout:
    box_mask: tuple[bool, ...]
    unit_ranges: tuple[tuple[int, int], ...]
    q_for_v: tuple[int, ...]
    unsafe_q: tuple[int, ...]


def _joint_layout(joint) -> _JointCoordinateLayout:
    if joint.kind == "composite":
        box_mask: list[bool] = []
        unit_ranges: list[tuple[int, int]] = []
        q_for_v: list[int] = []
        unsafe_q: list[int] = []
        q_offset = 0
        for child in joint.sub_joints:
            layout = _joint_layout(child)
            box_mask.extend(layout.box_mask)
            unit_ranges.extend((q_offset + start, q_offset + stop) for start, stop in layout.unit_ranges)
            q_for_v.extend(q_offset + index if index >= 0 else -1 for index in layout.q_for_v)
            unsafe_q.extend(q_offset + index for index in layout.unsafe_q)
            q_offset += child.nq
        return _JointCoordinateLayout(tuple(box_mask), tuple(unit_ranges), tuple(q_for_v), tuple(unsafe_q))

    box_mask = (False,) * joint.nq
    unit_ranges: tuple[tuple[int, int], ...] = ()
    q_for_v = (-1,) * joint.nv
    unsafe_q: tuple[int, ...] = ()
    if joint.kind == "free_flyer":
        box_mask, unit_ranges, unsafe_q = (True, True, True, False, False, False, False), ((3, 7),), (0, 1, 2)
    elif joint.kind == "spherical":
        unit_ranges = ((0, 4),)
    elif joint.kind == "planar":
        box_mask, unit_ranges, q_for_v = (True, True, False, False), ((2, 4),), (0, 1, -1)
    elif joint.kind == "revolute_unbounded":
        unit_ranges = ((0, 2),)
    elif joint.kind.startswith(("revolute", "prismatic")) or joint.kind in {"helical", "translation"}:
        box_mask, q_for_v = (True,) * joint.nq, tuple(range(joint.nv))
    return _JointCoordinateLayout(box_mask, unit_ranges, q_for_v, unsafe_q)


class RobotVariable(Variable):
    """A robot configuration variable with optional construction-time group freezing."""

    def __init__(
        self,
        model: Model,
        tensor: torch.Tensor | None = None,
        *,
        name: str | None = None,
        trainable: bool = True,
        bounds: Bounds | bool | None = None,
        batch_ndim: int = 0,
        time_axis: int | None = None,
        frozen_groups: Sequence[str] = (),
    ) -> None:
        if not isinstance(model, Model):
            raise TypeError(f"model must be a Model, got {type(model).__name__}")
        self.model = model
        self._feature_width = model.nq
        if tensor is None:
            tensor = model.q_neutral.clone()
        if bounds is True:
            bounds = self.joint_bounds()
        elif bounds is False:
            bounds = None
        super().__init__(
            tensor,
            name=name,
            trainable=trainable,
            bounds=bounds,
            batch_ndim=batch_ndim,
            time_axis=time_axis,
        )
        if isinstance(frozen_groups, str) or not isinstance(frozen_groups, Sequence):
            raise TypeError("frozen_groups must be a sequence of tangent-group names")
        if any(not isinstance(group, str) for group in frozen_groups):
            raise TypeError("frozen_groups must contain only tangent-group names")
        self._frozen_groups = tuple(frozen_groups)
        groups = self._checked_tangent_groups(self._frozen_groups) if self._frozen_groups else {}
        free = torch.ones(model.nv, dtype=torch.bool, device=self.tensor.device)
        for group in self._frozen_groups:
            free[groups[group]] = False
        per_knot = torch.nonzero(free, as_tuple=False).flatten()
        knots = self.time_length if self.time_axis is not None else 1
        offsets = torch.arange(knots, dtype=torch.long, device=self.tensor.device).unsqueeze(-1) * model.nv
        self._free_indices = (offsets + per_knot).flatten()
        frozen = ~free
        self._frozen_q_mask = torch.zeros(model.nq, dtype=torch.bool, device=self.tensor.device)
        for joint, nq, nv, iq, iv in zip(
            model.joint_models, model.nqs, model.nvs, model.idx_qs, model.idx_vs, strict=True
        ):
            local = frozen[iv : iv + nv]
            if not local.any():
                continue
            if local.all():
                self._frozen_q_mask[iq : iq + nq] = True
            else:
                assert joint.kind == "free_flyer", "only root_lin/root_ang may split a joint tangent"
                self._frozen_q_mask[iq : iq + 3] = local[:3]
                self._frozen_q_mask[iq + 3 : iq + 7] = local[3:].any()

    @property
    def frozen_groups(self) -> tuple[str, ...]:
        """Return the immutable names frozen when this variable was constructed."""
        return self._frozen_groups

    def tangent_groups(self) -> Mapping[str, torch.Tensor]:
        """Return ordered joint and derived groups of per-knot tangent indices."""
        root_id = next(
            (
                joint_id
                for joint_id in self.model.structure.manifold_free_flyer_joint_ids
                if self.model.parents[joint_id] == 0
            ),
            None,
        )
        if root_id is None:
            ambiguous = sorted({"root", "root_lin", "root_ang"}.intersection(self.model.joint_names))
            if ambiguous:
                raise ValueError(
                    f"Robot model {self.model.name!r} joint names {ambiguous} conflict with reserved "
                    "floating-root tangent groups"
                )
        groups = {
            name: torch.arange(start, start + width, dtype=torch.long, device=self.tensor.device)
            for name, start, width in zip(self.model.joint_names, self.model.idx_vs, self.model.nvs, strict=True)
        }

        def add_derived(name: str, indices: torch.Tensor) -> None:
            existing = groups.get(name)
            if existing is not None and not torch.equal(existing, indices):
                raise ValueError(
                    f"Robot model {self.model.name!r} joint name {name!r} conflicts with the derived "
                    "tangent group of the same name"
                )
            groups[name] = indices

        joints = torch.ones(self.model.nv, dtype=torch.bool, device=self.tensor.device)
        if root_id is not None:
            start = self.model.idx_vs[root_id]
            root = torch.arange(start, start + self.model.nvs[root_id], dtype=torch.long, device=self.tensor.device)
            add_derived("root", root)
            add_derived("root_lin", root[:3])
            add_derived("root_ang", root[3:])
            joints[root] = False
        add_derived("joints", torch.nonzero(joints, as_tuple=False).flatten())
        return MappingProxyType(groups)

    def _checked_tangent_groups(self, names: Sequence[str]) -> Mapping[str, torch.Tensor]:
        groups = self.tangent_groups()
        unknown = [name for name in names if name not in groups]
        if unknown:
            raise ValueError(f"Robot model {self.model.name!r} has no tangent groups {unknown}")
        return groups

    def tangent_weight(self, weights: Mapping[str, Real], default: Real = 1.0) -> torch.Tensor:
        """Build per-coordinate square-root-information row multipliers.

        Values multiply residual rows directly; the least-squares objective
        therefore sees their square. Later overlapping entries take precedence.
        """
        if not isinstance(weights, Mapping):
            raise TypeError(f"weights must be a mapping, got {type(weights).__name__}")
        if any(not isinstance(name, str) for name in weights):
            raise TypeError("weights must use tangent-group names as keys")
        if isinstance(default, bool) or not isinstance(default, Real):
            raise TypeError(f"default must be a real number, got {type(default).__name__}")
        groups = self._checked_tangent_groups(tuple(weights))
        result = self.tensor.new_full((self.model.nv,), float(default))
        for name, value in weights.items():
            if isinstance(value, bool) or not isinstance(value, Real):
                raise TypeError(f"weight for group {name!r} must be a real number")
            result[groups[name].to(result.device)] = float(value)
        return result

    @property
    def box_mask(self) -> torch.Tensor:
        mask = torch.zeros(self.model.nq, dtype=torch.bool)
        for joint, nq_joint, iq in zip(self.model.joint_models, self.model.nqs, self.model.idx_qs, strict=True):
            if nq_joint:
                mask[iq : iq + nq_joint] = torch.tensor(_joint_layout(joint).box_mask)
        return mask

    def joint_bounds(self) -> Bounds:
        lower, upper = self.model.lower_pos_limit, self.model.upper_pos_limit
        box = self.box_mask.to(lower.device)
        return Bounds(
            torch.where(box, lower, torch.full_like(lower, -torch.inf)),
            torch.where(box, upper, torch.full_like(upper, torch.inf)),
        )

    @property
    def unit_coordinate_slices(self) -> tuple[slice, ...]:
        ranges: list[tuple[int, int]] = []
        for joint, nq_joint, iq in zip(self.model.joint_models, self.model.nqs, self.model.idx_qs, strict=True):
            if nq_joint:
                ranges.extend((iq + start, iq + stop) for start, stop in _joint_layout(joint).unit_ranges)
        return tuple(slice(start, stop) for start, stop in ranges)

    def _validate_bounds(self, bounds: Bounds | None) -> None:
        if bounds is None:
            return
        if tuple(bounds.lower.shape) != (self.model.nq,):
            raise ValueError(f"bounds must have state shape ({self.model.nq},), got {tuple(bounds.lower.shape)}")
        protected = ~self.box_mask.to(bounds.lower.device)
        valid = torch.isneginf(bounds.lower[protected]).all() & torch.isposinf(bounds.upper[protected]).all()
        if not bool(valid):
            indices = torch.nonzero(protected, as_tuple=False).flatten().tolist()
            raise ValueError(
                "RobotVariable bounds must be (-inf, +inf) on non-box manifold coordinates "
                f"{indices}; quaternion/unit-circle coordinates are never clamped"
            )

    def tangent_dim(self) -> int:
        assert self.tensor.is_floating_point(), "non-floating static Variables have no tangent"
        return _numel(self.shape[:-1]) * self.model.nv

    def retract(self, delta: torch.Tensor) -> torch.Tensor:
        """Retract a free delta while restoring frozen ambient coordinates."""
        projected = super().retract(delta)
        if not self._frozen_groups:
            return projected
        return torch.where(self._frozen_q_mask.to(projected.device), self.tensor, projected)

    def _tangent_event_shape(self) -> tuple[int, ...]:
        return (*self.shape[:-1], self.model.nv)

    def _retract_full(self, value: torch.Tensor, delta: torch.Tensor) -> torch.Tensor:
        return self.model.integrate(value, delta)

    def _difference_full(self, x0: torch.Tensor, x1: torch.Tensor) -> torch.Tensor:
        return self.model.difference(x0, x1)


__all__ = ["Bounds", "RobotVariable", "SE3Variable", "SO3Variable", "Variable"]
