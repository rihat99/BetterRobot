# scan_backend — torch/warp duality, kinematics + dynamics structure

Scope: `src/better_robot/{kinematics,dynamics}/`, warp bridges/kernels, benches, plan results, CHANGELOG.
Paths below are relative to `src/better_robot/` unless they start with `docs/`, `plan/`, `benchmarks/`, `tests/`.
Budget-limited audit: I grepped most files and read only key regions. Items marked UNVERIFIED were not read or run.

## 1. Algorithm map

| Algorithm | Torch impl (lines) | Warp impl | Loop structure (torch) |
|---|---|---|---|
| FK (joint poses) | `kinematics/forward.py:120` `forward_kinematics_raw` (file 321) -> `kinematics/_fk_matrix.py` (355) | `kinematics/_warp_kernels.py` `fk_frames_f32/f64` (472) + `_warp_bridge.py` (403) | Local transforms batched per joint kind (`_fk_matrix.py:114-146` plan, `:192` group loop over kinds, `:215` fallback loop). World chain is a **per-joint Python loop**, one 4x4 matmul each: `_fk_matrix.py:262-265`, then `torch.stack`. |
| Frame placements | `kinematics/forward.py:305` `frame_placements_raw` | fused into the warp FK kernel (`forward.py:273-274`) | batched gather |
| Joint / frame Jacobians | `kinematics/jacobian.py:53,80` (file 504) + `_jacobian_columns.py` (140) | none | Batched column pass, constant 52 launches for any njoints (`plan/for_agents/02_results.md:32-35`). Column index maps built once with Python loops at structure time (`_jacobian_columns.py:67-76`). |
| Jacobian time variation (J-dot) | `kinematics/jacobian.py:132,193,253` | none | Reuses batched column pass, +8 constant launches (`docs/CHANGELOG.md:19`). |
| RNEA | `dynamics/rnea.py:106` `rnea_raw` (file 375) | `dynamics/_warp_kernels.py:202` `rnea_f32`, `:568` `rnea_f64` (749) + `_warp_bridge.py` (582) | Per-joint Python loops: forward `rnea.py:186`, backward `:234`, plus per-DOF inner loop `:241`. Inverse adjoints built in one batched call (`:170`). |
| CRBA | `dynamics/crba.py:43` (147) | none | Per-joint loops `crba.py:75` (reversed), `:87`, `:99`, with ancestor walk inside (`:121`). Lists of per-joint slices (`:68,72`). |
| ABA | `dynamics/aba.py:51` (313) | none | Three per-joint loops `aba.py:163, 196, 238`; per-DOF inner loop `:255`; per-joint Python caches `:155-158`. |
| Centroidal (CCRBA, CoM, map) | `dynamics/centroidal.py:75,147,186,216` (262) | none | Per-joint list of composite inertias `centroidal.py:104-107`. |
| Dynamics derivatives (dRNEA/dq, dABA) | none found (no exports in `dynamics/__init__.py`) | none | Only through autograd on the passes above. |
| State manifold | `dynamics/state_manifold.py` (72) | none | n/a |

No topology "levels" exist anywhere: `grep level data_model/topology.py data_model/model_structure.py` finds nothing. Every recursive pass iterates `structure.topo_order` joint by joint. The only thing batched across joints is the non-recursive work (local transforms, adjoints, inertia tables).

Line totals: warp = 472 + 403 + 749 + 582 = **2206**. Torch kin/dyn = 321 + 355 + 504 + 140 + 375 + 147 + 313 + 262 = 2417, plus joint models `data_model/joint_dispatch.py` + `data_model/joint_models/*.py` = 1180.

## 2. How the warp lane is wired

- **Opt-in:** a per-call flag on only two functions: `forward_kinematics(..., use_warp=False, use_compile=False)` (`kinematics/forward.py:173-174`) and `rnea(..., use_warp=False)` (`dynamics/rnea.py:275`). CRBA, ABA, Jacobians, J-dot and centroidal have neither flag.
- **Fallback:** if warp is missing or the input is unsupported, the call warns once per reason and runs the torch lane (`forward.py:40-52, 240-257`; `rnea.py:52-58, 306-323`). During CUDA graph capture it raises instead (`kinematics/_warp_bridge.py:264-285`).
- **Tensor crossing:** zero-copy `wp.from_torch(...)` views, about 25 arrays per launch (`kinematics/_warp_bridge.py:127-151`; `dynamics/_warp_bridge.py:174-214`). It launches on torch's current stream (`kinematics/_warp_bridge.py:125`). Outputs are fresh torch tensors. Every call re-runs `.to(device=...)` on about 11 structure tensors (`kinematics/_warp_bridge.py:219-229`). That is a no-op when they are already on the device, but it still adds Python overhead.
- **torch.compile:** the forward is a `torch.library.custom_op("better_robot::warp_fk_forward")` (`kinematics/_warp_bridge.py:93`) with a `register_fake` (`:165`). RNEA has the same pattern (`dynamics/_warp_bridge.py:229` `_warp_rnea_forward_fake`). The flag's docstring says `use_warp` takes precedence over `use_compile` (`forward.py:201`).
- **Autograd:** `_WarpFKFunction(torch.autograd.Function)` (`kinematics/_warp_bridge.py:193`) and `_WarpRNEAFunction` (`dynamics/_warp_bridge.py:273`). Their `backward` (`kinematics/_warp_bridge.py:233`) calls `_vjp` (`:38`; dynamics `:27`). `_vjp` **re-runs the torch implementation** under autograd (`forward_kinematics_raw` at `kinematics/_warp_bridge.py:72`; `rnea_raw` imported at `dynamics/_warp_bridge.py:15`) and backpropagates through it. The module docstring explains why: "Warp's generated adjoint is not correct for a dynamic topological loop" (`kinematics/_warp_bridge.py:5-8`). So there are **no hand-written adjoint kernels and no warp Tape**. Warp speeds up the forward only; the backward costs torch-forward plus torch-backward, which is more than the pure-torch lane's backward.
- **Supported:**
  - dtypes: f32 and f64 (`kinematics/_warp_bridge.py:119-122`).
  - joint kinds: FK covers codes 0..helical (`kinematics/_warp_bridge.py:35`); RNEA covers 0-15, and composite and mimic-sentinel joints are declined (`dynamics/_warp_bridge.py:20-24`).
  - devices: FK tolerates CPU (`stream ... if q.is_cuda else None`, `:125`). RNEA calls `torch.cuda.current_stream` unconditionally (`dynamics/_warp_bridge.py:171`), so it is CUDA only. The docs call FK "CUDA-validated" (`forward.py:193`).
  - double backward: works only because the backward is a torch recompute (claimed in `kinematics/_warp_bridge.py:6-7`; I did not test it).
  - shape-batched placements (`docs/CHANGELOG.md:28`).
  - RNEA backward skips outputs that receive no gradient (`docs/CHANGELOG.md:32`).
- **Not supported / UNVERIFIED:** `torch.func` / vmap. The autograd.Function has no `setup_context`/`vmap` staticmethod that I saw. Joint types outside the code ranges above, and graph capture with non-contiguous input, are also unsupported.
- **Import side effects:** `wp.init()` and a global `wp.config.kernel_cache_dir` defaulting to `/tmp/betterrobot-warp-cache` run at import (`dynamics/_warp_bridge.py:18-19`).
- **Duplication:**
  1. The same math is written in torch and again in warp: FK, and RNEA, whose kernel also recomputes FK and writes `local`/`world` (`dynamics/_warp_bridge.py:210-211`). That makes 3 copies of FK.
  2. Within warp, every kernel and helper is copied for f32 and f64: `dynamics/_warp_kernels.py:15-380` (f32) mirrors `:383-749` (f64), about 2 x 183 lines for `rnea_*` alone; `kinematics/_warp_kernels.py:14/41` `_compose_f32/_f64`, and so on.
  3. Joint-type dispatch is written twice. Torch uses class dispatch in `data_model/joint_models` (1180 lines). The warp FK kernel has its own 22 `kind ==` branches (`grep -c` on `kinematics/_warp_kernels.py`). Warp RNEA uses tables instead (`motion_subspaces`, `_expanded_tangent_f32` at `dynamics/_warp_kernels.py:67`).
  4. The bridges duplicate the execution-batch flattening and index maps (q_map, value_map, frame_map, ...) for each lane.

## 3. Stored performance numbers

| Algorithm | Robot | Batch | Device | Lane: time | Source |
|---|---|---|---|---|---|
| FK fwd | SMPL-like (nj 25, nq 99) | 1 | RTX 6000 Ada, f32? | warp 0.542 ms; compile-fullgraph 0.712 ms (warp 1.32x) | `tests/bench/baselines/warp_fk_cuda_rtx6000_ada_b1.json` (commit c0560e3, torch 2.13+cu126, warp 1.15) |
| FK fwd | same | 16 | same | warp 0.582; compile 1.424 (2.45x) | `..._b16.json` |
| FK fwd | same | 256 | same | warp 0.630; compile 1.253 (1.99x) | `..._b256.json` |
| FK fwd | same | 4096 | same | warp 0.625; compile 1.246 (1.99x) | `..._b4096.json` |
| FK fwd / fwd+bwd, eager, matrix rewrite | 203-joint model | 256 | GPU | fwd 152 -> 9.5 ms; fwd+bwd 551 -> 44 ms | `docs/CHANGELOG.md:22-25` |
| Joint Jacobians fwd / fwd+bwd, eager | smpl25 | 256 | GPU f32 | 14.20 -> 0.614 ms; 64.53 -> 16.93 ms | `plan/for_agents/02_results.md:32` |
| same | smplx56 | 256 | GPU f32 | 39.34 -> 0.778; 164.4 -> 28.59 | `02_results.md:33` |
| same | mhr203 (nj 68, nv 204) | 256 | GPU f32 | 47.35 -> 0.842; 192.9 -> 31.28 | `02_results.md:34` |
| same | panda | 256 | GPU f32 | 7.39 -> 0.480; 44.79 -> 13.23 | `02_results.md:35` |
| same | panda | 1 | CPU | 2.291 -> 0.131; 12.50 -> 2.656 | `02_results.md:44` |
| Inverse dynamics fwd+bwd (batched adjoints) | — | — | — | "roughly halving", no absolute numbers | `docs/CHANGELOG.md:29-31` |
| Warp RNEA vs torch RNEA | — | — | — | **no stored numbers found** | — |
| ABA / CRBA / centroidal | — | — | — | no numbers found | — |

Caveats:
- The warp-vs-compile baseline compiles `forward_kinematics_raw` (`benchmarks/m1_warp_fk_vs_compiled_torch.py:183, 494-496`) at commit c0560e3. That commit may predate the matrix-FK rewrite (`CHANGELOG.md:22-27`), so the compile column may be stale.
- Both lanes stay **flat from batch 1 to 4096** (warp about 0.55-0.63 ms, compile about 0.7-1.4 ms). Both are latency-bound (launches plus Python and guard overhead), not compute-bound, at 25 joints. The 2x warp advantage is a constant overhead gap, not a throughput gap.
- Every warp number is forward only. No stored number covers warp forward+backward. Since the backward recomputes torch, warp fwd+bwd must be at least the torch fwd+bwd.
- `benchmarks/baseline.py` and `m1_eager_cpu.py` write JSON results; I found no stored warp/compile numbers from them except `tests/bench/baselines/torch_filtered_smpl_b1_rtx6000_ada.json`, which holds definitions only in the fields I extracted.

Project conclusions:
- `docs/concepts/design_decisions.md:94-97`: warp over raw CUDA to avoid a second source language; "kernels are whole passes, never per-op. PyTorch is the truth; Warp is the speed."
- `plan/for_agents/02_results.md:46-47`: batched torch beats the Python loop even on CPU for a tiny model.

No note gives a rule for when warp wins. The only evidence is the FK table above: about 2x on the forward pass, at a single model size.

## 4. How torch speed is achieved today

- **Launch-volume reduction** is the main lever:
  - FK went from about 78 launches per joint to about 2 per joint. Local transforms are grouped by joint kind, there is one matmul per joint, and one final matrix-to-quaternion pass (`docs/CHANGELOG.md:22-25`; `_fk_matrix.py:1-11`).
  - The Jacobian pass went from about 50 launches per joint to a constant 52 (`02_results.md:22-38`).
  - RNEA, ABA, CRBA and centroidal build inverse adjoints and inertia tables in one batched call (`CHANGELOG.md:29-31`; `rnea.py:170`).
- **Precomputed plans and index tensors:**
  - `_fk_matrix.get_plan` is memoised per structure (`_fk_matrix.py:104`); it holds kind groups, q_index gathers and axes (`:72-97`).
  - Jacobian column maps are built once per structure (`_jacobian_columns.py:56-76`).
  - Structure tensors (`parents_tensor`, `topo_order_tensor`, `joint_kind_tensor`, `idx_*_tensor`) live on `ModelStructure`.
- **torch.compile:** used in exactly one place. `_fk_matrix.matrix_forward_compiled` wraps `matrix_core` in `torch.compile(run, dynamic=True)` and caches it **on the structure object** via `object.__setattr__(structure, "_fk_matrix_compiled_core", ...)` (`_fk_matrix.py:322-334`). Nothing else is compiled. No CUDA graphs are used, except that warp tolerates graph capture.
- **Joint dispatch:** class per joint kind in `data_model/joint_models/` plus `data_model/joint_dispatch.py`. FK groups joints by kind so dispatch happens per kind, not per joint (`_fk_matrix.py:192`). Joints without a family go through a per-joint fallback loop (`:215`).
- **Remaining per-joint Python loops on the hot path** (each iteration launches several kernels; launches scale with njoints):
  - FK world chain: `_fk_matrix.py:262`.
  - RNEA: `rnea.py:186, 234`, plus per-DOF `:241`.
  - CRBA: `crba.py:75, 87, 99`, plus the ancestor walk.
  - ABA: `aba.py:163, 196, 238`, plus per-DOF `:255`.
  - CCRBA: `centroidal.py:104-107`.
  - For the 203-joint model, FK eager is still 9.5 ms fwd and 44 ms fwd+bwd at b256 (`CHANGELOG.md:24`), against 0.84 ms for the fully batched Jacobian pass (`02_results.md:34`). That gap is the cost of a per-joint loop.

## 5. Structural problems

1. **Public/raw split is consistent in shape but heavy.** Each algorithm has a `*_raw(structure, values, ...)` pure function returning a frozen `*Result`, plus a public wrapper that validates, routes lanes and writes into `Data`:
   - `forward.py:120` vs `:167`; `crba.py:43` vs `:130`; `aba.py:51` vs `:275`; `rnea.py:106` vs `:267`.
   - There are seven Result dataclasses: FKResult, FramePlacementsResult, JointJacobiansResult, JointJacobiansTimeVariationResult, RNEAResult, CRBAResult, ABAResult, plus CentroidalResult and CCRBAResult.
   - Jacobians also carry a Pinocchio-style `compute_*` / `get_*` pair (`jacobian.py:299-504`). That gives three API layers for one quantity.
2. **Data mutation:**
   - `Data` is a frozen dataclass that is mutated through `object.__setattr__`. There are 52 assignment or `object.__setattr__` sites in kin/dyn (grep count), including the `_kinematics_level` state machine (`crba.py:146`, `aba.py:312`).
   - Hidden mutable caches sit on "immutable" `ModelStructure`: the compiled FK core (`_fk_matrix.py:333`) and the memoised plan.
3. **Lane flags are inconsistent:** `use_warp` exists on 2 of about 8 algorithms and `use_compile` on 1, with precedence rules (`forward.py:201`). The flags leak backend choice into the user API.
4. **Fallback behavior differs by context:** it warns and continues normally, but raises under graph capture (`kinematics/_warp_bridge.py:278-285`). The user cannot tell from the call which lane ran.
5. **Joint dispatch exists in three forms:**
   - torch classes (`data_model/joint_models`),
   - an int8 `kind` code switch in the warp FK kernel (22 branches),
   - tables in the warp RNEA kernel (motion subspaces, expansions).
   Adding a joint kind means touching all three.
6. **Frames:** frame placements are a separate pass in torch (`forward.py:285, 305`) but fused in warp (`forward.py:273`). Frame Jacobians are derived from joint Jacobians (`jacobian.py:80`).
7. **Large files:** `dynamics/_warp_kernels.py` (749 lines, half of it f32/f64 copy), `dynamics/_warp_bridge.py` (582), `jacobian.py` (504, 12 public and raw functions), `kinematics/_warp_kernels.py` (472).
8. **Import side effects:** `wp.init()` and a global cache dir in `/tmp` (`dynamics/_warp_bridge.py:18-19`).

## 6. Options for ending the duality

Facts from item 3 that decide this:
- (a) Warp wins only the **forward** pass, about 2x (0.63 vs 1.25 ms), at one model size (25 joints). Both lanes are flat in batch, so the gap is overhead, not throughput.
- (b) The warp backward **is** torch (recompute), so warp fwd+bwd is never faster than torch fwd+bwd.
- (c) Warp covers 2 of about 8 algorithms. RNEA warp has no stored timing.
- (d) The biggest measured wins in the project came from torch launch-volume work (Jacobians 23-56x fwd; FK 16x fwd and 12.5x fwd+bwd at 203 joints).
- (e) The remaining torch cost is the per-joint loops listed in item 4.

**(A) Pure torch, delete warp.**
- Deleted: 2206 lines of warp, the `use_warp` flags and fallback paths in `forward.py` and `rnea.py` (about 60 lines), and the warp tests and bench (not counted). Added: nothing.
- Lost: about 0.6 ms per FK forward call at batch 1-4096 for a 25-joint model; the fused frames in the same kernel; and an unmeasured RNEA forward gain.
- Kept: gradients, double backward, model-value gradients, CPU, f64, vmap (UNVERIFIED today).
- Risk: low. The backward is unchanged, because warp's backward is already torch.
- Weakness: it leaves the per-joint loops as the throughput limit for 50-200-joint human models (FK 9.5 ms fwd at 203 joints, b256).

**(B) Warp kernels as the single implementation behind a torch autograd boundary.**
- Required: hand-written adjoint kernels for FK, frames, Jacobians, J-dot, RNEA, CRBA, ABA and centroidal, including gradients with respect to model values (placements, inertias; human shape fitting needs these). None exist today. The project itself says warp's generated adjoint is wrong for the dynamic topological loop (`kinematics/_warp_bridge.py:5-8`).
- Double backward would need adjoint-of-adjoint kernels or would be lost. That matters for use case 4 (the solver as a differentiable layer in NN training), for Gauss-Newton through gradients, and for `torch.func.jacrev`/vmap, which would be lost.
- Every joint kind must be re-coded in the kernel switch; the torch joint_models would become dead for compute.
- Size: rough estimate +4-8k lines (forward and adjoint for about 8 passes, times f32/f64 unless templated with `wp.func` overloads); delete about 2.4k torch lines plus 1.2k joint_models compute.
- Risk: high. Correctness of hand adjoints, a CUDA-centric CPU story, and an extra dependency. The evidence (a 2x forward gain on one model) does not justify that rewrite cost.

**(C) A single level-batched tensor program in torch, optionally compiled.**
- The idea: precompute per-depth index tensors (joints grouped by tree depth). Replace every `for i in topo_order` with `for level in levels` and do gather-and-scatter batched ops. FK chain, RNEA forward and backward, CRBA composite accumulation, ABA's three passes and CCRBA all fit this pattern. CRBA's ancestor fill becomes a precomputed support-mask product, like the Jacobian columns already are.
- Launches then scale with tree depth instead of njoints. For SMPL-like trees the depth is about 8-10 against 24 joints (UNVERIFIED; depth not measured). `torch.compile(dynamic=True)` can then fuse each level.
- Kept: all autograd features, double backward, vmap, CPU and f64, one joint dispatch (torch classes grouped by kind, as `_fk_matrix` already does).
- Deleted: 2206 lines of warp. Changed: about 1.4k lines of torch passes rewritten at similar size; plus a level-plan builder of about 100-150 lines, reusing `_fk_matrix.get_plan`.
- Risk: medium. It is unproven that compiled level-batched torch closes the 2x FK forward gap: the only compile number (1.25 ms) is from an older, per-joint FK. Other risks are compile and recompile time, and guard overhead at batch 1.
- Precedent: the Jacobian rewrite cut launches to a constant 52 and reached 0.84 ms forward on 68 joints (`02_results.md:34`).

**Judgment:**
- (C) is the only option that removes the duality without losing a feature the four use cases need: double backward, model-value gradients, vmap, CPU.
- (A) is the zero-risk first step, since warp's backward already is torch.
- (B) inverts the "PyTorch is the truth" rule (`design_decisions.md:97`) and costs the most for a gain that has only been measured as about 2x on the forward pass.

Before committing to C, measure on the same RTX 6000 Ada, batches 1/256/4096, fwd and fwd+bwd, for SMPL-25 and MHR-203:
- current eager matrix FK,
- compiled matrix FK,
- warp FK,
- a quick level-batched FK prototype.

## Open questions

- What are the tree depths of SMPL, SMPLX and MHR-203? This sets the launch count for option C.
- Has the warp-vs-compile bench been re-run since the matrix-FK rewrite? Commit c0560e3 compared to HEAD is unknown.
- Does warp RNEA have any timing at all? None is stored.
- Do vmap and `torch.func` work through the current `use_warp` lane? Not tested.
