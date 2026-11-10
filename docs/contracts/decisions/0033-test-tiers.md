# 0033 Three test tiers, each one local command; a missing oracle fails

**Status:** Accepted (owner); amended by 0046. **Date:** 2026-11-01

## Context
The Pinocchio parity suite uses `importorskip` while `pin` sits in the `dev` extra; 34 `importorskip("robot_descriptions")`
sit in 19 BetterRobot test files; BetterHuman has 33 licensed-asset skip sites; 8 BetterRobot contract tests skip and
report healthy; pytest runs without strict settings; the BetterRobot full suite takes 210 s, mostly `tests/docs`; the
four BetterHuman examples run in no test. There is no CI in this major: every check is a command on this machine.

## Decision
We will run three tiers per repository, each one local command with a recorded time: fast (the contract tests
without `slow`, seconds, from the pre-commit hook), full (CPU, all but `cuda` and `licensed`, from the pre-push hook and
the release checklist), and release (`cuda`, `licensed` and the counters, by hand through the free-GPU script), with
oracles in a test dependency group, `licensed` deselected in fast and full, the release command failing when the models directory or an
asset is missing, and strict pytest settings.

## Example
```toml
[tool.pytest.ini_options]
addopts = "--strict-markers --strict-config"
xfail_strict = true
filterwarnings = ["error"]
empty_parameter_set_mark = "fail_at_collect"
markers = ["cuda", "licensed", "slow"]
```
```text
tier     selects                              run by                          fails when
fast     tests/contract, not slow             pre-commit hook                 any test fails
full     all but cuda and licensed            pre-push hook, release list     any test fails; a missing oracle fails
release  everything, plus the counters        by hand, free-GPU script        the models directory or an asset is missing
```

## Consequences
A green run means the gates ran. Parity runs in the full tier on oracles installed from the test group (and on random
weights, BetterHuman 0016). A CI service is a later decision with its own record. Cost: tiers run only when someone runs them, so
the hook and the checklist are the enforcement.

## Alternatives
Two tiers with licensed tests skipping as today: against principle 6.5, a skip reads as a pass. CI would run fast and
full on every push; it is deferred.

## Compatibility
Changes test behaviour only: previously skipped tests fail until their oracle or fixture exists (0039 for robots).
Reversible: yes.

## Enforcement
`tests/contract/test_pytest_config.py` (the strict settings above); `tests/contract/test_no_skips.py` (no `skip`,
`skipif`, `importorskip` or `xfail` in a test the fast or full tier selects, an oracle or `robot_descriptions` import
included; a shrink-only ratchet to zero).

## Open questions
None.
