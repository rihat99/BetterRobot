"""Use RNEA for gravity compensation and verify it with ABA."""

import torch

import better_robot as br


def build_pendulum() -> br.Model:
    """Build a one-joint pendulum with a gravity-sensitive center of mass."""
    builder = br.ModelBuilder("pendulum")
    builder.add_body("base", mass=0.0)
    builder.add_body(
        "link",
        mass=2.0,
        com=torch.tensor([0.3, 0.0, 0.0]),
        inertia=torch.diag(torch.tensor([0.2, 0.3, 0.4])),
    )
    builder.add_revolute_y(
        "shoulder",
        parent="base",
        child="link",
        origin=torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]),
        lower=-2.0,
        upper=2.0,
    )
    return br.io.build_model(builder.finalize(), dtype=torch.float64)


def run() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute gravity torque and recover the requested zero acceleration."""
    model = build_pendulum()
    q = torch.tensor([0.4], dtype=torch.float64)
    velocity = torch.zeros(model.nv, dtype=torch.float64)
    desired_acceleration = torch.zeros_like(velocity)

    gravity_torque = br.rnea(model, q, velocity, desired_acceleration)
    recovered_acceleration = br.aba(model, q, velocity, gravity_torque)
    round_trip_error = (recovered_acceleration - desired_acceleration).abs().max()

    print(f"gravity-compensation torque: {gravity_torque.tolist()}")
    print(f"ABA round-trip max error: {round_trip_error.item():.3e}")
    return gravity_torque, recovered_acceleration, round_trip_error


if __name__ == "__main__":
    run()
