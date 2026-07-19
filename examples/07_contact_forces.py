"""Fit standing contact forces for a floating body.

``solve_contact_forces`` applies every point force at its joint origin, so
the reported local wrench has zero moment. Contacts at an offset need the
additional ``r × force`` moment and are outside this example's model.
"""

import torch

import better_robot as br
from better_robot.tasks import ContactForceWeights


def build_standing_body() -> br.Model:
    """Build a minimal floating body whose root joint is the contact."""
    builder = br.ModelBuilder("standing_body")
    builder.add_body(
        "base",
        mass=2.0,
        inertia=torch.diag(torch.tensor([0.2, 0.25, 0.3])),
    )
    builder.add_free_flyer_root(
        "ground_contact",
        child="base",
        origin=torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]),
    )
    return br.io.build_model(builder.finalize(), dtype=torch.float64)


def run(*, time_steps: int = 3, max_iter: int = 30):
    """Solve and print the contact wrench at each trajectory sample."""
    model = build_standing_body()
    q = model.q_neutral.expand(time_steps, -1).clone()
    result = br.solve_contact_forces(
        model,
        q,
        contact_joint_ids=[1],
        active_mask=torch.ones(time_steps, 1, dtype=torch.bool),
        dt=0.05,
        weights=ContactForceWeights(
            base_wrench=1.0,
            force_magnitude=1e-6,
            force_smooth=1e-3,
        ),
        max_iter=max_iter,
        tolerance=1e-9,
    )
    wrenches = result.fext_local[:, 1].detach()
    print("local contact wrenches [force, moment] at the joint origin:")
    print(wrenches.round(decimals=4))
    return result


if __name__ == "__main__":
    run()
