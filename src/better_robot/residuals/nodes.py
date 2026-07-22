"""Evaluation-scoped computation nodes shared by object-referenced residuals."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, cast

from ..kinematics.forward import forward_kinematics
from .utils import RobotLike, RobotValueLike, RobotVariableLike, ValueLike


class Node(ABC):
    """A lazy value whose graph-bearing memo is scoped to one evaluation.

    Inputs may be Variable-like leaves or child nodes. ``variables`` contains
    the transitive leaves, while ``nodes`` contains only direct children.
    """

    def __init__(self, *inputs: ValueLike | Node) -> None:
        variables: list[ValueLike] = []
        nodes: list[Node] = []
        variable_ids: set[int] = set()
        node_ids: set[int] = set()
        for value in inputs:
            if isinstance(value, Node):
                if id(value) not in node_ids:
                    nodes.append(value)
                    node_ids.add(id(value))
                leaves = value.variables
            elif isinstance(value, ValueLike):
                leaves = (value,)
            else:
                raise TypeError(f"node inputs must be Variable-like or Node, got {type(value).__name__}")
            for variable in leaves:
                if id(variable) not in variable_ids:
                    variables.append(variable)
                    variable_ids.add(id(variable))
        self.variables = tuple(variables)
        self.nodes = tuple(nodes)
        self._memo: Any = None
        self._has_memo = False
        self._evaluation_depth = False
        self._memo_owner: Node = self

    @property
    def merge_key(self) -> tuple[object, ...] | None:
        """Return an identity-only merge key, or ``None`` to disable merging."""
        return None

    def _share_with(self, canonical: "Node") -> None:
        self._memo_owner = canonical._memo_owner

    def _invalidate(self) -> None:
        owner = self._memo_owner
        owner._memo = None
        owner._has_memo = False

    def _begin_evaluation(self) -> bool:
        owner = self._memo_owner
        nested = owner._evaluation_depth
        self._invalidate()
        owner._evaluation_depth = True
        return nested

    def _end_evaluation(self, nested: bool) -> None:
        owner = self._memo_owner
        if not owner._evaluation_depth:
            raise RuntimeError("Node evaluation scope underflow")
        self._invalidate()
        owner._evaluation_depth = nested

    def value(self):
        """Return a fresh standalone value or the active evaluation's memo."""
        owner = self._memo_owner
        if not owner._evaluation_depth:
            return owner.compute()
        if not owner._has_memo:
            owner._memo = owner.compute()
            owner._has_memo = True
        return owner._memo

    def _checked_value(self, expected: type) -> Any:
        value = self.value()
        if not isinstance(value, expected):
            raise TypeError(f"{type(self).__name__} must produce {expected.__name__}, got {type(value).__name__}")
        return value

    @abstractmethod
    def compute(self):
        """Compute this node from its current variable tensors."""


class RobotState(Node):
    """Lazy forward-kinematics bundle for one robot configuration variable."""

    def __init__(self, q: RobotValueLike) -> None:
        if not isinstance(q, RobotValueLike):
            raise TypeError(f"q must expose robot variable state, got {type(q).__name__}")
        self.q = q
        self.model = q.model
        super().__init__(q)

    @property
    def merge_key(self) -> tuple[object, ...]:
        return (RobotState, id(self.q))

    def compute(self):
        return forward_kinematics(self.model, self.q.tensor, compute_frames=True)


def robot_state(value: RobotVariableLike | RobotState) -> tuple[RobotVariableLike, RobotState]:
    state = value if isinstance(value, RobotState) else RobotState(value)
    if not isinstance(state.q, RobotLike):
        raise TypeError(f"q must be a RobotVariable or RobotState, got {type(value).__name__}")
    return cast(RobotVariableLike, state.q), state


__all__ = ["Node", "RobotState"]
