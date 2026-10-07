# A1 state (subagent ran out of context)
Done: trash + renames + optim2->optim sed (BR src/tests/examples, BH src/tests/examples), root __init__ (tasks exports dropped, DAG), layer ranks,
CLAUDE.md (root, optim, viewer), contract tests ported (submodule_public_imports, protocols, public_api, docstrings, hot_path_lint), skeleton_signatures.
Removed old-API test functions saved in a1_removed_old_api_tests.py.
Open: tests/kinematics/test_jacobians.py helper ~L222 uses Problem/RobotVariable/PoseResidual (port finite-diff helper to terms/optim);
tests/docs/* and tests/examples/test_{ik,contact_dynamics}_examples.py target trashed examples/old docs; full suite not finished counting;
README.md still old API; docs/reference/api/*optim2* (auto-generated Sphinx) untouched; kinematics/CLAUDE.md:58 old Problem wording;
BH tests + examples not run; BR examples not run; ruff: 3 F821 in test_jacobians.py, many pre-existing errors elsewhere.
