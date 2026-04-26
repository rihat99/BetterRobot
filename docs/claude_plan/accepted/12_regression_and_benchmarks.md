# 12 · Regression oracles and benchmark baselines

★★ **Structural.** Lands the regression oracle and benchmark
baseline files specified in
[16 §4.5](../../conventions/16_TESTING.md) and
[14 §4](../../conventions/14_PERFORMANCE.md), and noted as missing in
[18 §3, §5](../../status/18_ROADMAP.md).

## Problem

Two missing artefacts cause silent drift:

### 12.1 No frozen FK reference

[16 §4.5](../../conventions/16_TESTING.md) says:

> `tests/kinematics/fk_reference.npz` holds FK outputs for Panda and
> G1 at a pinned set of `q`s, saved with a git-tracked version tag.
> A regression test fails if the current FK diverges from the stored
> reference beyond tolerance.

[18 §5](../../status/18_ROADMAP.md) lists this as **missing**. Until it
exists, a subtle Lie-algebra change (frame convention drift, log
branch flip near `π`, quaternion sign convention) can pass every
unit test and still produce numerically different FK outputs. We
would notice only when a downstream user reports it.

### 12.2 No benchmark baseline

[14 §1](../../conventions/14_PERFORMANCE.md) sets concrete latency
targets (Panda FK ≤ 150 µs, `solve_ik` ≤ 8 ms, etc.). The CI gate
is

> `pytest tests/bench/ --benchmark-compare --benchmark-fail=mean:20%`

against `tests/bench/baseline.json`. That file does not exist; the
benchmarks themselves are not present.

## Goal

Two artefacts land in the repo, with the procedure for regenerating
them documented and reviewable. Once landed, every PR is silently
guarded against numerical drift and perf regression.

## The proposal

### 12.A `tests/kinematics/fk_reference.npz`

A `.npz` (numpy zipped) file with:

```
arrays:
  panda_q                : (50, 7)        random q's, fixed seed=0
  panda_joint_pose_world : (50, 9, 7)     reference FK output (8 joints + universe)
  panda_frame_pose_world : (50, K, 7)     reference frame placements

  g1_q                   : (50, 36)       random q's incl. free-flyer head
  g1_joint_pose_world    : (50, 31, 7)
  g1_frame_pose_world    : (50, K, 7)

metadata:
  oracle_version : "1.0"
  generation_seed: 0
  generated_with : "torch=2.5.0, pypose=0.6.7"
  fk_dtype       : "float64"          # always fp64 for the oracle
  generated_at   : "2026-04-25T12:34:56Z"
```

Reproducibility hinges on `oracle_version` + `generation_seed` +
`fk_dtype` + the pinned dependency versions, **not** on a git SHA.
The gpt-plan review correctly noted: a SHA in the metadata is
useless once that commit is squashed/rebased and worse than useless
when contributors who don't have it locally try to regenerate. The
combination above is enough to reproduce the oracle byte-for-byte
on a clean check-out at any future commit, provided the generator
script and FK semantics are unchanged. When they *do* change,
`oracle_version` bumps and the new file is committed — the version
field carries the meaning the SHA pretended to.

Generation script:

```python
# tests/kinematics/_generate_fk_reference.py
"""Regenerate fk_reference.npz. Run only when the FK convention or
the test fixture intentionally changes; commit the new file as part
of the same PR. Discussed in proposal 12 of docs/claude_plan/."""
import numpy as np
import torch
import better_robot as br

def main():
    torch.manual_seed(0)

    panda = br.load("panda.urdf").to(dtype=torch.float64)
    panda_q = sample_q(panda, n=50)
    panda_data = br.forward_kinematics(panda, panda_q, compute_frames=True)

    g1 = br.load("g1.urdf", free_flyer=True).to(dtype=torch.float64)
    g1_q = sample_q(g1, n=50)
    g1_data = br.forward_kinematics(g1, g1_q, compute_frames=True)

    np.savez(
        "tests/kinematics/fk_reference.npz",
        panda_q=panda_q.numpy(),
        panda_joint_pose_world=panda_data.joint_pose_world.numpy(),
        panda_frame_pose_world=panda_data.frame_pose_world.numpy(),
        g1_q=g1_q.numpy(),
        g1_joint_pose_world=g1_data.joint_pose_world.numpy(),
        g1_frame_pose_world=g1_data.frame_pose_world.numpy(),
        oracle_version="1.0",
        generation_seed=0,
        generated_with=f"torch={torch.__version__}",
        fk_dtype="float64",
    )

if __name__ == "__main__":
    main()
```

The regression test:

```python
# tests/kinematics/test_fk_regression.py
import numpy as np
import torch
import better_robot as br
from pathlib import Path
from utils.testing import assert_close_manifold

REF = np.load(Path(__file__).parent / "fk_reference.npz")

def test_panda_fk_matches_oracle():
    panda = br.load("panda.urdf").to(dtype=torch.float64)
    q = torch.from_numpy(REF["panda_q"])
    data = br.forward_kinematics(panda, q, compute_frames=True)

    assert_close_manifold(
        data.joint_pose_world,
        torch.from_numpy(REF["panda_joint_pose_world"]),
        atol=1e-10,
    )
    assert_close_manifold(
        data.frame_pose_world,
        torch.from_numpy(REF["panda_frame_pose_world"]),
        atol=1e-10,
    )
```

Same shape for G1.

### 12.B Pinocchio cross-check (optional but cheap)

[18 §5](../../status/18_ROADMAP.md) also lists the Pinocchio cross-check
as missing. The way to do it: a sibling file `_pinocchio_oracle.npz`
generated *from Pinocchio* (not from BetterRobot), with the same `q`
samples. The cross-check test compares both oracles to within
`1e-8` (fp64). The cross-check makes the FK convention provably
identical to Pinocchio's, which is a strong correctness statement
for new users coming from C++ Pinocchio.

This is opt-in: the test skips if `pinocchio` is not importable.

### 12.C Benchmark baselines

`tests/bench/` ships:

```
tests/bench/
├── conftest.py                # fixtures: panda, g1, panda_data
├── bench_lie.py               # se3_compose / log / exp / inverse  micro-bench
├── bench_forward_kinematics.py # FK at B=1, 8, 1024 on CPU and CUDA
├── bench_jacobian.py
├── bench_solve_ik.py
├── baseline_cpu.json          # committed; 100MB max; one row per benchmark
├── baseline_cuda_l40.json     # committed; per-GPU-generation
└── README.md                  # how to update baselines
```

The benchmark format follows `pytest-benchmark`'s native JSON.

**Gate-promotion ladder.** Both CPU and CUDA benchmarks start
*advisory* — CI records numbers and posts a delta comment on the
PR but does not fail the build:

1. **CPU bench** — runs on every PR, advisory. After two release
   cycles in which the runner-noise floor sits below 5%, promote
   to blocking with a 20% threshold.
2. **CUDA bench** — runs nightly on the self-hosted GPU runner,
   advisory. Promote to PR-blocking once the same 5% noise-floor
   criterion holds and pinned GPU clocks are confirmed.
3. **Memory watermark** — CUDA-only, nightly, advisory throughout
   pre-1.0; promotion is part of the v1 release gate.

Hard CUDA gates *before* runner stability is measured produce
flaky, false-positive failures that get muted; the ladder above is
the gpt-plan-driven correction.

### 12.D Update flow for baselines

The README documents the procedure:

1. **Why update?** Either you intentionally improved a number (PR
   description states the win) or hardware changed.
2. **How?**
   ```bash
   uv run pytest tests/bench/ --benchmark-only --benchmark-autosave
   uv run pytest tests/bench/ --benchmark-compare=baseline_cpu \
                              --benchmark-compare-fail=mean:5%
   # If the comparison shows wins everywhere ≥ 5%:
   cp .benchmarks/Linux-CPython-3.11-64bit/0001_xxx.json \
      tests/bench/baseline_cpu.json
   ```
3. **PR review** — reviewer sees the JSON delta (text-diff is small;
   `pytest-benchmark` JSON is structured) and approves the bump.

The baseline file is **not** auto-bumped by CI; CI only fails if a
threshold is missed. The intentional-update is a human decision.

### 12.E Memory watermark

[14 §4.4](../../conventions/14_PERFORMANCE.md) defines a `mem_watermark`
test. The implementation is short:

```python
# tests/bench/test_mem_watermark.py
import torch, pytest, better_robot as br

@pytest.mark.gpu
def test_ik_b1024_under_200_mb(g1):
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    q = g1.q_neutral.unsqueeze(0).expand(1024, -1).cuda()
    targets = {"left_hand": ..., "right_hand": ...}
    br.solve_ik(g1, targets, initial_q=q)
    peak_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
    assert peak_mb < 200, f"peak {peak_mb:.1f} MB exceeds 200 MB budget"
```

Not run on every PR (CUDA-only); runs nightly.

### 12.F What a baseline-bump PR looks like

Concretely:

```
PR title: bench: bump CPU baseline after FK fast-path optimisation

Files changed:
  src/better_robot/kinematics/forward.py    # actual code change
  tests/bench/baseline_cpu.json             # JSON delta

PR description should include the relevant pytest-benchmark output:

  Performance: bench_forward_kinematics.test_panda_fk_cpu_b1
    OLD: 1.2 ms
    NEW: 0.94 ms       (-22%)
```

Reviewer reads the description, verifies the JSON delta corresponds
to the prose, approves. No CI bypass needed; the new baseline is
just a committed file.

## Tradeoffs

| For | Against |
|-----|---------|
| Numerical drift becomes impossible to merge silently. | A 50-sample fp64 npz is ~50 KB per robot. Fine. |
| Performance regressions show up as red CI on the same day they're introduced. | Benchmarks need a stable runner; flaky GPU CI is a maintenance cost. Mitigation: pin GPU clocks; run on a self-hosted runner. |
| Pinocchio cross-check makes "we're consistent with the rigid-body literature" testable, not just claimed. | Optional dependency; not every CI runner has Pinocchio installed. Mitigation: test `skipif`. |

## Acceptance criteria

- `tests/kinematics/fk_reference.npz` exists, < 200 KB total.
- `tests/kinematics/test_fk_regression.py` passes.
- `tests/kinematics/_generate_fk_reference.py` runs cleanly and
  reproduces the same file given the same seed and pinned versions.
- `tests/bench/baseline_cpu.json` and at least one CUDA baseline
  are committed.
- CI runs `pytest tests/bench/ --benchmark-compare` against the
  committed baseline on the main branch.
- PR template includes a "did you bump the baseline?" check.

## Cross-references

- [14 §4](../../conventions/14_PERFORMANCE.md) — performance budgets
  these tests enforce.
- [16 §4.5](../../conventions/16_TESTING.md) — regression-oracle spec.
- [18 §3, §5](../../status/18_ROADMAP.md) — closes the listed missing
  items.
- [Proposal 03](03_replace_pypose.md) — the FK reference is what
  catches PyPose-vs-pure-torch numerical disagreement during the
  swap.
- [Proposal 11](11_quality_gates_ci.md) — wires the benchmark gate
  into CI.
