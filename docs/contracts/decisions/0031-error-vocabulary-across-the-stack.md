# 0031 BetterHuman raises BetterRobot's errors and roots its own under them

**Status:** Accepted (owner). **Date:** 2026-11-01

## Context
BetterRobot builds one error tree (0020). BetterHuman adds `io.FormatError` with no stated base and raises plain
`ValueError` elsewhere (for example SOMA's "does not support contact frames"). It silently accepts a misspelled stream
group: `identity.forward(q, {"expresion": ...})` returns with no offset applied, where BetterRobot would raise
`UnknownNameError`.

## Decision
We will have BetterHuman raise BetterRobot's classes for the shared failure kinds (`ShapeError`,
`DtypeMismatchError`, `DeviceMismatchError`, `UnknownNameError`, `ArgumentError`) and define only its own under
BetterRobot's root: `FormatError`, and an `AssetNotFoundError` whose message names `BETTERHUMAN_MODELS_DIR`.

## Example
```python { .no-run }
class FormatError(BetterRobotError): ...          # io.check_format raises it
class AssetNotFoundError(BetterRobotError): ...   # "SMPLX_NEUTRAL.npz not found under $BETTERHUMAN_MODELS_DIR=/data/models"

try:
    state = identity.forward(q, streams={"expresion": expression})
except br.exceptions.UnknownNameError as error:    # valid names: ['expression', 'jaw', ...]
    ...
```

## Consequences
One `except BetterRobotError` catches the whole stack. Misspelled groups fail at the call. A reader of a BetterHuman
traceback sees the classes it already knows from BetterRobot.

## Alternatives
A separate tree rooted at `BetterHumanError(Exception)`: callers catch two roots and two trees drift. Pinocchio and
Drake raise standard exceptions with no library root, so there is no precedent for a split tree to follow.

## Compatibility
Changes the class of errors BetterHuman raises (still `ValueError` where BetterRobot's class is also one); callers that
caught a bare `ValueError` keep working. Partly reversible: adding classes is compatible.

## Enforcement
New `BetterHuman/tests/contract/test_error_tree.py`: every exception class defined in `better_human` derives from
`BetterRobotError`, `src/better_human` raises no bare `ValueError` or `KeyError` (AST), and the misspelled-group case
above raises `UnknownNameError`.

## Open questions
None.
