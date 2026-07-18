# External comparison definition

This definition is committed before measurement. It is intentionally not a
performance claim. Results may be added only with exact package revisions,
raw JSON, and the caveats below; bare ratios are not acceptable.

## Environment and provenance

Use the RTX 6000 Ada host and software header in `../definitions.md`. Run every
competitor in an isolated environment without modifying BetterRobot's lock.
Record the CUDA driver/toolkit, framework and compiler versions, git revision
or wheel hash, cache directories, allocator settings, and whether the first
call compiled code. The in-tree cuRobo candidate is the research cuRoboV2
checkout; its exact commit must be recorded and it must not be described as a
public v0.7 result.

## Batched Panda IK versus cuRobo

- Robot: the same fixed-base Franka Panda URDF from `robot_descriptions`, with
  identical joint limits and one terminal hand frame. Collision is off because
  this repository has no implemented capsule-collision oracle.
- Dataset: seed `20260718`; `B ∈ {1, 16, 256, 4096}`. Generate targets by
  sampling bounded Panda configurations with a local generator, then applying
  the same FK oracle once and persisting the configurations and SE(3) targets.
  Both solvers start at neutral. No solver may tune on the measured targets.
- Work budget: 32 iterations with early exit disabled is the primary throughput
  comparison. A secondary time-to-success study may be reported, but it must
  not replace the fixed-work result.
- Success: translation error at most 1 mm and orientation geodesic error at
  most 0.01 rad, recomputed afterward with BetterRobot's fp32 FK from the
  returned configurations. Report success count and both error distributions;
  throughput without success is invalid.
- BetterRobot configurations: public eager Torch is the required baseline.
  A compiled/Warp/graph "best approved" result may be added only for features
  already accepted on that device/dtype; list every enabled lane. Tuned and
  default configurations are separate rows.

## JAX-class FK and inverse-dynamics sweeps

Use the same fixed-base Panda model, fp32 inputs, gravity, spatial-force order,
and output contract. Sweep FK world joint poses and RNEA generalized forces at
the four batch sizes above using seed `20260718`. Persist q/v/a tensors and
compare output values before timing. A library case is dropped—not adapted
into a different workload—if it cannot express the same joints, frame
convention, gravity, outputs, or precision. SMPL-scale may be added only after
the exact spherical/free-flyer model is representable on both sides.

For JAX, record tracing/JIT compilation as cold start just as Torch compilation
and Warp module compilation are recorded. Synchronize device work on every
timed sample.

## Timing, memory, and result schema

Use 20 warmups and 100 samples. Report median, inclusive Q1/Q3, min and max for
host-observed synchronized latency; also report successful solves/s and the
per-iteration denominator. Cold import, model construction, compilation,
first kernel launch, and graph recording are separate fields. Measure peak
live allocation after resetting framework counters, plus allocator reserved
bytes and Warp/JAX pool information where exposed.

Each JSON result contains: schema version; benchmark/case id; seed; robot and
target artifact hash; batch; precision; fixed iteration count; success
thresholds/count; collision setting; software and hardware header; cold costs;
steady timing distribution; memory counters; output parity/error metrics; and
a free-text caveat list. A smoke test must validate this schema before any
result is committed.

## Required caveats

Report solver formulation, specialization, graph capture, collision support,
and success-rate differences next to every comparison. BetterRobot knowingly
trades per-robot generated CUDA specialization for a generic two-lane design;
an expected cuRobo gap is still published. B=1 is mandatory. No result may be
extrapolated to an unmeasured robot, batch, collision mode, or precision.
