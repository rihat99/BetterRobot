# 0030 The formats that cross repositories, listed in the standard

**Status:** Accepted (owner). **Date:** 2026-11-01

## Context
Formats cross repositories today: URDF and MJCF read by BetterRobot, `.viser` written by its viewer, SMPL-family `.npz`
(upstream), the converted MHR and SOMA archives (`format_version=1`, no `format`), and BetterHuman's frame-set JSON
(no version), read by two apps. BetterRobot's serialization ADR 0006 is Proposed (see 0032).

## Decision
We will list exactly these formats in the coupling standard: URDF (read), MJCF (read, if 0036 keeps it), `.viser`
(write), `better_human.mhr` and `better_human.soma` (BetterHuman 0012) and `better_human.frame_set` (BetterHuman 0013), with no BetterRobot
model file, and give every listed BetterHuman format a spec page, a reader that checks `format` and `format_version`,
and one fixture per past version.

## Example
```text
format                  direction  owner          horizon
URDF                    read       BetterRobot    upstream format
MJCF                    read       BetterRobot    compiled by MuJoCo (0036), or absent
.viser                  write      BetterRobot    viewer recording
better_human.mhr        read       BetterHuman    format_version 2; v1 migrated in memory
better_human.soma       read       BetterHuman    format_version 2; v1 migrated in memory
better_human.frame_set  read/write BetterHuman    format_version 1; read by BetterVideoReconstruction and BetterRetarget
```

## Consequences
A file written by one release stays readable by every later one (8.7), and a reader refuses a later version with an
error naming both versions. Formats the standard does not list carry no promise.

## Alternatives
List none and document formats ad hoc: the apps read the frame-set JSON without any promise today. SQLite's file header
carries a format version that older readers refuse to guess at, which is the behaviour required here.

## Compatibility
Adds a promise; the listed BetterHuman formats get a version where they had none (BetterHuman 0012, BetterHuman 0013). Removing a format later
needs a record.

## Enforcement
Per format, in `BetterHuman/tests/contract/test_formats.py`: a round-trip test, one fixture per past `format_version`
under `BetterHuman/tests/support/formats/`, a test that a later version is refused, and a check that no `format_version`
comparison exists outside `check_format` and the migration module (8.7, 8.8).

## Open questions
None.
