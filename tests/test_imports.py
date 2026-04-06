"""Smoke test: verify all public modules and symbols are importable."""


def test_top_level_imports() -> None:
    import better_robot as br

    assert hasattr(br, "RobotModel")
    assert hasattr(br, "load_urdf")
    assert hasattr(br, "solve_ik")
    assert hasattr(br, "solve_trajopt")
    assert hasattr(br, "retarget")
    assert hasattr(br, "forward_kinematics")
    assert hasattr(br, "compute_jacobian")
    assert hasattr(br, "solvers")
    assert hasattr(br, "costs")
    assert hasattr(br, "viewer")
    assert hasattr(br, "models")
    assert hasattr(br, "algorithms")
    from better_robot import IKConfig
    cfg = IKConfig()
    assert cfg.pos_weight == 1.0
    assert cfg.ori_weight == 0.1
    assert cfg.pose_weight == 1.0
    assert cfg.limit_weight == 0.1
    assert cfg.rest_weight == 0.01


def test_geometry_imports() -> None:
    from better_robot.algorithms.geometry import primitives
    assert hasattr(primitives, "Sphere")
    assert hasattr(primitives, "Capsule")
    assert hasattr(primitives, "Box")
    assert hasattr(primitives, "HalfSpace")
    assert hasattr(primitives, "Heightmap")


def test_solver_imports() -> None:
    from better_robot import solvers

    assert hasattr(solvers, "LevenbergMarquardt")
    assert hasattr(solvers, "GaussNewton")
    assert hasattr(solvers, "AdamSolver")
    assert hasattr(solvers, "LBFGSSolver")
    assert hasattr(solvers, "SOLVERS")
    assert set(solvers.SOLVERS.list()) == {"lm", "lm_pypose", "gn", "adam", "lbfgs"}


def test_costs_imports() -> None:
    from better_robot import costs

    for name in [
        "pose_residual", "limit_residual", "velocity_residual",
        "acceleration_residual", "jerk_residual", "rest_residual",
        "smoothness_residual", "self_collision_residual",
        "world_collision_residual", "manipulability_residual",
    ]:
        assert hasattr(costs, name), f"Missing: {name}"


def test_viewer_helpers() -> None:
    import torch
    from better_robot.viewer import wxyz_pos_to_se3, qxyzw_to_wxyz, build_joint_dict, build_cfg_dict
    from better_robot import load_urdf
    from robot_descriptions.loaders.yourdfpy import load_robot_description

    # wxyz_pos_to_se3
    se3 = wxyz_pos_to_se3((1.0, 0.0, 0.0, 0.0), (0.1, 0.2, 0.3))
    assert se3.shape == (7,)
    assert abs(se3[0].item() - 0.1) < 1e-6  # tx
    assert abs(se3[6].item() - 1.0) < 1e-6  # qw

    # qxyzw_to_wxyz
    quat = torch.tensor([0.0, 0.0, 0.0, 1.0])  # identity [qx,qy,qz,qw]
    wxyz = qxyzw_to_wxyz(quat)
    assert wxyz == (1.0, 0.0, 0.0, 0.0)

    # build_joint_dict (primary function)
    urdf = load_robot_description("panda_description")
    model = load_urdf(urdf)
    q = model._q_default.clone()
    d = build_joint_dict(model, q)
    assert isinstance(d, dict)
    assert len(d) == model.joints.num_actuated_joints
    assert all(isinstance(v, float) for v in d.values())

    # build_cfg_dict (deprecated alias — must still work)
    d2 = build_cfg_dict(model, q)
    assert d2 == d
