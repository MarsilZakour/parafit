# parafit — design document

**parafit** is a pluggable, pip-installable analytical solver for fitting
parametric models (hands, bodies, 6-DoF objects) to multi-view evidence. It is
the shared solver core extracted from the UA-Fit / POEM-v2 line of work.

## 1. Motivation

The same batched Levenberg-Marquardt fitter has been hand-ported **four times**
across the thesis-hope tree, sharing ideas but no code:

| # | Solver | Location | Method |
|---|---|---|---|
| 1 | UmeTrack IK | `show3d_challenge/scripts/umetrack_ik.py` | Adam over sigmoid-reparam flexion |
| 2 | UA-Fit LM (canonical) | `mvhpe/POEM-v2/lib/models/dovf/analytic_fitter_unc.py` | analytic block-weighted GN/LM |
| 3 | Real-time HOI | `action_segmentation/hoitcn/hoi_lm.py::solve_gn_multi` | self-contained multi-body port |
| 4 | SHOW3D refiner | `show3d_challenge/scripts/refine.py` | UA-Fit re-implementation, no import |

Each re-derivation is a maintenance tax and a source of drift. parafit makes the
solver a versioned dependency that UA-Fit, FuseFit, SHOW3D, and future work all
import.

A second problem parafit fixes: in the canonical solver, optional energies
(`WILOR_2D`, `TEMPORAL_ANCHOR`, `ANATOMY_W`, `OBJECT_SDF`, …) are toggled via
**module-global switches** that callers set and restore in `finally`. parafit
replaces that global state with explicit, composable `Energy` objects — same
math, no hidden coupling.

## 2. Architecture — three decoupled axes

The core solver imports neither MANO nor cameras. Everything specific is a plugin.

```
parafit/
  core/     solver.py (LMSolver), types.py (ParamSpec/State/GNBlock/SolveResult),
            lie.py (SO(3) exp/log/right-Jac)
  io/       camera.py (pinhole project + closed-form Jacobian)
  models/   base.py (Model ABC), example.py (RigidKeypointModel, tested),
            mano.py [mano], umetrack.py [umetrack]
  energies/ base.py (Energy ABC + gauss_newton_block), reprojection.py, anchor3d.py,
            pose_prior.py, contact_sdf.py [contact]
  registry.py  (lazy plugin registry with actionable ImportError per extra)
```

### 2.1 Model

A `Model` owns its parameter layout (`ParamSpec`), a differentiable `forward`
producing landmarks/verts, and an optional analytic `landmark_jacobian`
(`(B, J, 3, P)`). Returning `None` falls back to autograd (`torch.func`), so a
new model works immediately and can be accelerated later. An optional
reparameterization (`to_raw`/`from_raw`) supports constructions like UmeTrack's
normalized flexion `[0,1]` mapped into per-subject `joint_limits` via a sigmoid.

Reference: `RigidKeypointModel` (6-DoF, analytic Jacobian, fully tested) —
this is the "wrist 6-DoF" sub-problem and needs no external assets.

### 2.2 Energy

Every energy contributes one Gauss-Newton block `(A (B,P,P), g (B,P), cost (B,))`
via `linearize(model, params, state)`. The solver populates `state.landmark_jac`
once so landmark-space energies chain through it. `gauss_newton_block(r, J, Ω)`
assembles `A = ΣJᵀΩJ`, `g = ΣJᵀΩr` for scalar, diagonal, or anisotropic block
precision `Ω`.

Implemented (core, torch-only): `ReprojectionEnergy` (subsumes both the DOVF
data term and the external-2D term), `Anchor3DEnergy` (temporal / learned-3D
prior), `PosePriorEnergy` (isotropic or Mahalanobis). Stub: `ContactSDFEnergy`
([contact], `push_fn(verts)->push`).

### 2.3 Solver

`LMSolver` generalizes `_gn_loop_unc`: assemble the summed normal equations,
solve the damped system, per-sample accept/reject trust-region step, adapt
damping (×0.3 on accept, ×3 on reject). Modes:

- `lm` — adaptive (best accuracy; torch/cuda backends).
- `fixed` — fixed-iteration, no data-dependent control flow → **the
  TensorRT-exportable path** (see §4).
- `pure` — undamped Gauss-Newton (diagnostics/ablation).

Backward (roadmap): `unroll` (autograd through the loop, default) and
`implicit` (O(1)-memory fixed-point differentiation, from `_ImplicitGNUnc`).

## 3. Installation and extras

Core is torch-only. Extras pull only what a given model/energy needs; a missing
extra yields an actionable `ImportError` rather than a cryptic traceback.

| Command | Adds | Enables |
|---|---|---|
| `pip install parafit` | torch | solver + reprojection/anchor/pose-prior |
| `parafit[mano]` | smplx (+ manotorch via git) | `ManoModel`, anatomy energy |
| `parafit[umetrack]` | pure torch, no deps (hand-model json not vendored -- see README) | `UmeTrackModel` |
| `parafit[contact]` | mesh2sdf, trimesh | `ContactSDFEnergy` |
| `parafit[all]` / `[dev]` | everything / tests | — |

**uv.** Clean PEP 621 packaging means `uv pip install "parafit[mano]"` and
`uv add parafit --extra mano` work directly; dev deps use PEP 735 dependency
groups. manotorch has no PyPI release, so it is declared under
`[tool.uv.sources]` as a git dependency (works for git/uv installs; a PyPI
publish documents manotorch as a manual step).

**Two frictions handled deliberately:** (1) torch+CUDA is depended on loosely
(`torch>=2.1`) and installed by the user for their platform first; (2) MANO
weights are not shipped (license) — the loader points at the user's local
`mano_v1_2` assets.

## 4. Backend roadmap: torch → CUDA → TensorRT

Assembly (residuals → Jacobians → normal equations) is separated from the
**execution backend** (loop + linear solve):

- **torch** (now) — eager, autograd-differentiable, adaptive LM.
- **cuda** (planned) — fused kernels for assembly and the batched SPD solve.
- **tensorrt** (planned) — export a *static* graph. TensorRT cannot represent
  data-dependent accept/reject, so only `mode="fixed"` (fixed iteration count,
  fixed damping schedule) is exportable. Keeping both modes from day one means
  we can train/eval in `lm` and deploy the exact same energies/models in `fixed`
  through a TensorRT engine. This constraint is already reflected in the solver.

## 5. Migration map — the four solvers become configs

| Consumer | Today | With parafit |
|---|---|---|
| **UA-Fit** (`DOVFManoMVUnc`) | `analytic_fit_unc` + module globals | `ManoModel` + `[Reprojection(dovf,Ω), Anchor3D, PosePrior]` |
| **FuseFit** | globals `WILOR_2D`, `OBJECT_SDF` | `+ Reprojection(external_uv,Ω) + ContactSDF` |
| **SHOW3D** | `refine.py` re-impl + Adam IK | `UmeTrackModel` + `[Reprojection, Anatomy]` |
| **Real-time HOI** | `solve_gn_multi` port | multi-body `Scene` (§6, v0.2) |

During development every consumer does `pip install -e /code/parafit`, so edits
are live everywhere — one source of truth, no vendoring — before publishing.

## 6. Roadmap

- **v0.1** — DONE: `ManoModel` (analytic `mano_kinematic_jac`, FD-validated),
  `UmeTrackModel` (analytic revolute-chain Jacobian, FD-validated to <1e-10 rel-err)
  + tensordict batchable containers. Remaining: the anatomy-barrier energy, the
  implicit backward, and full UA-Fit / SHOW3D parity through parafit (same-harness
  parity is the acceptance gate).
- **v0.2** — a `Scene` of multiple `Model`s with cross-body coupling energies
  (hand+object contact), covering the real-time `solve_gn_multi` case; SMPL-X and
  6-DoF object models.
- **v0.3** — CUDA assembly/solve backend; TensorRT export of the `fixed` path;
  publish to PyPI.

## 7. Status (v0.1.0)

**Batchable containers (tensordict).** `State`, `GNBlock`, and the new
`Observations` (multi-view evidence bundle: K, w2c, target_uv, precision) are
`tensordict` tensorclasses with a leading batch dim `B`. A whole solve moves
(`state.to("cuda")`), indexes (`state[mask]`), stacks (`torch.stack`), and
accumulates (`block_a + block_b`, elementwise) as one object -- the "batchable
and fast" foundation. `tensordict` is a core dependency.

**Validated end to end** (`tests/`):
- `test_core_overfit.py` -- analytic Jacobian vs finite differences max error
  5e-11; multi-view rigid overfit to 0.0000px / 0.0000mm (cost 1.4e3 → 4.6e-27).
- `test_tensordict_batchable.py` -- State/GNBlock/Observations move, index,
  stack, add; an `Observations`-driven solve reaches 0.0000mm.
- `test_mano_model.py` -- `ManoModel` forward (21 OpenPose joints); the analytic
  kinematic Jacobian matches finite differences at **cos 1.0000 / rel-err 0.004**
  (the pose-blendshape term is omitted by design, a near-exact GN direction); a
  multi-view MANO overfit converges to ~1.5mm.

`ManoModel` is now a real, validated model (forward + analytic Jacobian). Note:
MANO uses boolean-mask ops that `vmap`/`jacrev` cannot batch, so such models must
provide an analytic Jacobian (as `ManoModel` does) rather than rely on the
autograd fallback.

- `test_umetrack_model.py` -- `UmeTrackModel` forward/verts match an independent
  numpy re-derivation of UmeTrack's FK+LBS to <1e-6m (both hands); the analytic
  revolute-chain Jacobian matches finite differences at rel-err ~2e-11; a
  3D-anchor overfit recovers a random pose to <1e-6mm; batch independence and
  left/right mirror consistency (including a nontrivial root pose) hold exactly.

`ContactSDFEnergy` remains a grounded port stub that names its thesis-hope source.
