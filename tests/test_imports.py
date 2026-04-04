"""Smoke test: verify all public modules and symbols are importable."""


def test_top_level_imports() -> None:
    import better_robot as br

    assert hasattr(br, "Robot")
    assert hasattr(br, "solve_ik")
    assert hasattr(br, "solve_trajopt")
    assert hasattr(br, "retarget")
    assert hasattr(br, "collision")
    assert hasattr(br, "solvers")
    assert hasattr(br, "costs")
    assert hasattr(br, "viewer")


def test_collision_imports() -> None:
    from better_robot import collision

    assert hasattr(collision, "RobotCollision")
    assert hasattr(collision, "Sphere")
    assert hasattr(collision, "Capsule")
    assert hasattr(collision, "Box")
    assert hasattr(collision, "HalfSpace")
    assert hasattr(collision, "Heightmap")


def test_solver_imports() -> None:
    from better_robot import solvers

    assert hasattr(solvers, "LevenbergMarquardt")
    assert hasattr(solvers, "GaussNewton")
    assert hasattr(solvers, "AdamSolver")
    assert hasattr(solvers, "LBFGSSolver")
    assert hasattr(solvers, "SOLVER_REGISTRY")
    assert set(solvers.SOLVER_REGISTRY.keys()) == {"lm", "gn", "adam", "lbfgs"}


def test_costs_imports() -> None:
    from better_robot import costs

    for name in [
        "pose_residual", "limit_residual", "velocity_residual",
        "acceleration_residual", "jerk_residual", "rest_residual",
        "smoothness_residual", "self_collision_residual",
        "world_collision_residual", "manipulability_residual",
    ]:
        assert hasattr(costs, name), f"Missing: {name}"
