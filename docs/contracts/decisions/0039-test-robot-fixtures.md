# 0039 The test robots are tracked fixtures

**Status:** Accepted (owner). **Date:** 2026-11-01

## Context
34 `importorskip("robot_descriptions")` calls in 19 BetterRobot test files supply the Panda and the G1 for forward
kinematics, Jacobian, dynamics, IK, collision, viewer and Pinocchio-parity tests. The package sits in the `demos` extra
and clones the robot repositories from GitHub at first use. Without it the suite skips and stays green, and the README's
install command removes the extra.

## Decision
We will track the Panda and G1 URDFs (with meshes only where a test needs them) under `tests/fixtures/robots/`
with their licence files, load them by path, keep `robot_descriptions` in the README and the examples, and, for a robot whose licence
forbids redistribution, put `robot_descriptions` in the test dependency group and import it plainly.

## Example
```python { .no-run }
# tests/support/robots.py
FIXTURES = Path(__file__).parents[1] / "fixtures" / "robots"

def panda_urdf() -> Path:
    return FIXTURES / "panda" / "panda.urdf"            # tracked, beside fixtures/robots/panda/LICENSE

def test_fk_matches_pinocchio() -> None:              # in tests/parity/pinocchio/
    model = br.load(panda_urdf(), dtype=torch.float64)    # offline, the same on every machine; no importorskip
```

## Consequences
Tests run offline and identically everywhere; a missing robot is a failing test, not a skip. The repository grows by the
URDFs and the needed meshes (each file within the 1 MiB rule, 9.3). A licence check is needed per robot before it is
tracked.

## Alternatives
`robot_descriptions` in the test dependency group with a plain import: the first run clones from GitHub and depends on
the network. Keep the silent skips: against principle 6.5.

## Compatibility
Test-only; no public change. Reversible: yes.

## Enforcement
`tests/contract/test_no_skips.py`: a shrink-only ratchet to zero on `importorskip("robot_descriptions")` in `tests/`
(0033); new `tests/contract/test_fixtures.py` (every directory under `tests/fixtures/robots/` has a licence file);
`tests/contract/test_repository_layout.py::test_no_symlinks_and_size` (the 1 MiB rule).

## Open questions
None.
