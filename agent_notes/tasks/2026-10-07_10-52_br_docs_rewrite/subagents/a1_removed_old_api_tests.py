# ---- from tests/kinematics/test_jacobians.py
def test_pose_residual_resolves_frame_name(arm):
    q = arm.q_neutral
    data = forward_kinematics(arm, q, compute_frames=True)
    frame_name = "body_link1"
    target = data.frame_pose_world[arm.frame_id(frame_name)].clone()
    q_variable = RobotVariable(arm, q, name="q")
    residual = PoseResidual(q_variable, frame=frame_name, target=target)

    assert residual.frame_id == arm.frame_id(frame_name)
    torch.testing.assert_close(residual.error(), torch.zeros(6))



# ---- from tests/io/test_branching_tree.py
def test_branching_rest_residual_backward_is_nan_free(branching_model):
    """RestResidual traverses one free-flyer and 23 identity SO3 logs."""
    q_rest = branching_model.q_neutral.to(dtype=torch.float64)
    q = q_rest.detach().clone().requires_grad_(True)
    q_variable = RobotVariable(branching_model, q, name="q")
    residual = RestResidual(q_variable, q_rest).error()
    residual.square().sum().backward()

    assert q.grad is not None
    assert torch.isfinite(q.grad).all()
    torch.testing.assert_close(q.grad, torch.zeros_like(q))

# ---- from tests/autograd/test_public_differentiability.py
def test_pose_residual_reaches_q_and_model_values(panda) -> None:
    q = _configuration(panda).requires_grad_()
    model, placements = _placement_model(panda)
    target_q = model.integrate(_configuration(model), torch.full((model.nv,), 0.02, dtype=q.dtype))
    target = br.forward_kinematics(model, target_q, compute_frames=True).frame_pose_world[_hand_frame(model)].detach()
    q_variable = RobotVariable(model, q, name="q")
    residual = PoseResidual(q_variable, frame_id=_hand_frame(model), target=target)
    _assert_gradients(residual.error(), q, placements)

# ---- from tests/data_model/test_mimic_reduced_map.py
def test_sign_aware_limits_and_generalized_capacities_are_reduced() -> None:
    model = build_model(_mimic_ir(), dtype=torch.float64)

    torch.testing.assert_close(model.lower_pos_limit, torch.tensor([-0.25], dtype=torch.float64))
    torch.testing.assert_close(model.upper_pos_limit, torch.tensor([0.375], dtype=torch.float64))
    torch.testing.assert_close(model.velocity_limit, torch.tensor([2.0], dtype=torch.float64))
    torch.testing.assert_close(model.effort_limit, torch.tensor([19.0], dtype=torch.float64))

    q = RobotVariable(model, torch.tensor([0.5], dtype=torch.float64), name="q")
    result = JointPositionLimit(q).error()
    assert result.shape == (2,)
    torch.testing.assert_close(result, torch.tensor([0.0, 0.125], dtype=torch.float64))



