"""In-tree composed posed-mesh residual integration."""

from __future__ import annotations

import torch

from better_robot.optim import Problem, Variable
from better_robot.residuals import (
    MaskedChamferResidual,
    PointProjectionResidual,
    SceneAttractionResidual,
    ScenePenetrationResidual,
    SceneSDFState,
)
from better_robot.residuals.nodes import Node


class _PosedBody(Node):
    def __init__(self, q: Variable) -> None:
        self.q = q
        self.calls = 0
        self.base = torch.tensor([[[0.0, -0.2, 2.0], [1.0, 0.2, 2.0]]])
        super().__init__(q)

    def compute(self) -> torch.Tensor:
        self.calls += 1
        offset = torch.tensor([0.0, 1.0, 0.0]) * self.q.tensor[0]
        return self.base + offset


def test_one_posed_body_feeds_all_shipped_mesh_residuals() -> None:
    q = Variable(torch.tensor([0.05]), name="q")
    posed = _PosedBody(q)
    scene = torch.tensor([[[0.0, 0.0, 2.0], [1.0, 0.0, 2.0]]])
    normals = torch.tensor([[[0.0, 1.0, 0.0], [0.0, 1.0, 0.0]]])
    validity = torch.ones(1, 2, dtype=torch.bool)
    contact_mask = Variable(torch.tensor([[True, False]]), name="contact_mask", trainable=False)
    state = SceneSDFState(posed, validity, scene, normals, validity, distance="plane")

    residuals = [
        ScenePenetrationResidual(
            state,
            mask=contact_mask,
            max_distance=0.3,
            max_penetration=0.3,
            margin=0.05,
            reduce="mean_active",
        ),
        SceneAttractionResidual(state, band=0.3, reduce="mean_active"),
        MaskedChamferResidual(
            posed,
            scene,
            validity,
            validity,
            vertex_weights=torch.ones(1, 2),
            bidirectional=False,
        ),
        PointProjectionResidual(
            posed,
            torch.tensor([[100.0, 0.0, 0.0], [0.0, 100.0, 0.0], [0.0, 0.0, 1.0]]),
            torch.eye(4),
            torch.zeros(1, 2, 2),
            confidence=torch.tensor([[0.5, 1.0]]),
            visibility=contact_mask,
            time_axis=0,
            reduce="mean_active",
        ),
    ]
    problem = Problem(residuals)

    posed.calls = 0
    rows = problem.error()
    assert rows.shape == (10,)
    assert posed.calls == 1
    assert torch.isfinite(rows).all()
    assert rows[1].item() == 0.0
    assert rows[-2:].tolist() == [0.0, 0.0]

    posed.calls = 0
    jacobian = problem.dense_jacobian(strategy="jacrev")
    assert jacobian.shape == (10, 1)
    assert posed.calls > 0
    assert torch.isfinite(jacobian).all()
