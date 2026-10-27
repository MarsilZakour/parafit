# parafit

**A pluggable analytical solver for parametric model fitting.**

`parafit` fits parametric models -- hands (MANO, UmeTrack), bodies (SMPL-X), and
6-DoF objects -- to multi-view evidence with a batched, differentiable
Levenberg-Marquardt solver. Its **energies**, **models**, and **execution
backend** are all pluggable. It is the reusable solver core behind
[UA-Fit](https://marsilzakour.github.io/ua-fit/) (ECCV 2026 HANDS Workshop;
[code](https://github.com/MarsilZakour/ua-fit-code), [weights](https://huggingface.co/Marsil123/ua-fit)), released
standalone so new projects depend on one implementation instead of re-porting the same math.

<p align="center">
  <img src="docs/media/lm_fit.gif" alt="Batched LM solver fitting a MANO hand: mesh converging alongside the decreasing energy" width="640">
  <br>
  <em>The LM solver fitting a MANO hand to multi-view evidence, with the energy trajectory live.</em>
</p>

## Why

Three ideas, cleanly separated:

- **Models** (`parafit.models`) -- a parametric model owns its parameter layout,
  a differentiable forward (FK+LBS -> landmarks/verts), and an optional
  *analytic* Jacobian (autograd fallback otherwise).
- **Energies** (`parafit.energies`) -- each residual term contributes a
  Gauss-Newton block `(A, g, cost)`. Compose them explicitly; no global state.
- **Solver** (`parafit.LMSolver`) -- adaptive LM with per-sample accept/reject,
  swappable backend (torch now; CUDA / TensorRT planned via the fixed-iteration
  exportable path).

## Install

Core is torch-only; install from GitHub (PyPI release planned):

```bash
pip install "git+https://github.com/MarsilZakour/parafit.git"                # core solver + reprojection/anchor/pose-prior energies
pip install "parafit[mano] @ git+https://github.com/MarsilZakour/parafit.git"  # + MANO hand model
pip install "parafit[all] @ git+https://github.com/MarsilZakour/parafit.git"   # everything
```

Or for development:

```bash
git clone https://github.com/MarsilZakour/parafit.git
cd parafit && pip install -e ".[mano]"
```

With **uv** (recommended; resolves the git-only manotorch dep):

```bash
uv pip install "parafit[mano] @ git+https://github.com/MarsilZakour/parafit.git"
```

Install torch for your CUDA build first. MANO weights are not shipped (license):
point the loader at your local `mano_v1_2` assets.

## Quick start (core, no MANO needed)

```python
import torch
from parafit import LMSolver, ReprojectionEnergy, PinholeCameras
from parafit.models.example import RigidKeypointModel

model = RigidKeypointModel(canonical_points)          # (J,3)
cams = PinholeCameras(K, w2c)                          # (B,V,3,3), (B,V,4,4)
energy = ReprojectionEnergy(cams, target_uv)          # (B,V,J,2)
result = LMSolver(max_iters=15).solve(model, init_params, [energy])
print(result.params, result.diagnostics["final_cost"])
```

Run the end-to-end test (multi-view rigid fit + finite-difference Jacobian check):

```bash
python tests/test_core_overfit.py
```

## UmeTrack hand model

`UmeTrackModel` (`parafit.models.umetrack`, extra `[umetrack]`, pure torch -- no extra
dependencies) is a from-scratch FK+LBS port of
[facebookresearch/UmeTrack](https://github.com/facebookresearch/UmeTrack)'s hand model
(`lib/common/hand_skinning.py`) under the parafit `Model` interface: a 22-joint, 17-bone,
788-vertex hand skeleton with an exact analytic landmark Jacobian.

The hand-model json itself is **not vendored** (UmeTrack is released under
**CC-BY-NC 4.0**, non-commercial, incompatible with bundling into an MIT package). Get one
yourself and point `load_umetrack_hand` at it:

```bash
git clone https://github.com/facebookresearch/UmeTrack.git /tmp/UmeTrack
# generic (non-subject-specific) hand model, used by the upstream repo's own demos:
#   /tmp/UmeTrack/dataset/generic_hand_model.json
```

```python
from parafit.models.umetrack import load_umetrack_hand, UmeTrackModel

hand = load_umetrack_hand("/tmp/UmeTrack/dataset/generic_hand_model.json")
model = UmeTrackModel(hand, side="right")   # side="left" mirrors the native model
```

Per-subject profiles in the same json layout (e.g. a dataset's own
`profile_umetrack.json`, with the fields nested one level under a `"hand_model"` key)
load the same way -- `load_umetrack_hand` accepts both layouts. See
`parafit/models/umetrack.py`'s module docstring for the parameter layout, units
(metres internally; UmeTrack json files are millimetres), and the exact native-UmeTrack
-> OpenPose-21 landmark mapping.

## MANO-convention landmarks on other hands (`parafit.mapping`)

Hand networks trained on MANO predict MANO's 21 joints (16 mesh-regressed joints
+ 5 fingertip vertices, OpenPose order). To fit MHR or UmeTrack to such evidence,
`ManoConventionHand` attaches every MANO joint to one bone of the target hand with
a fixed offset (a `ManoCorrespondence`), applies a uniform hand scale (shape
transfer, `hand_scale_between`), and exposes the result as a regular parafit
`Model` with an exact Jacobian. The correspondence also maps every MANO vertex to
a barycentric point on the target mesh, so a fitted MHR / UmeTrack hand can be
scored against MANO ground truth vertex by vertex.

```python
from parafit.mapping import ManoConventionHand, ManoCorrespondence
from parafit.mapping.calibration import calibrate_mano_correspondence
from parafit.models.mano import ManoModel
from parafit.models.mhr import MhrModel
from parafit.models.mhr_hand import MhrHandModel

hand = MhrHandModel(MhrModel.from_cache("mhr_lod1_cache.pt"), side="right", full_mesh=True)
mano = ManoModel("assets/mano_v1_2", side="right")
corr = calibrate_mano_correspondence(hand, mano, "assets/mano_v1_2", "mhr_right")   # once; corr.save(...)
model = ManoConventionHand(hand, corr)          # landmarks = MANO joints on the MHR hand
```

Any hand that implements the small `ArticulatedHand` protocol (wrist-local bone
frames, native landmarks, mesh) can be wrapped; `MhrHandModel` and `UmeTrackModel`
do. Calibration fits the target to synthetic MANO hands (PCA pose prior, random
betas), re-solves each joint's bone and offset in closed form, and repeats; it
reports the held-out residual (MHR 1.6 mm, UmeTrack see `calibration`) as the
representation gap between the two models.

## Status

v0.1.0 -- core solver (tensordict-batchable) + reprojection / 3D-anchor /
pose-prior energies, the `ManoModel` (analytic-Jacobian MANO hand), and the
`UmeTrackModel` (analytic-Jacobian UmeTrack hand), the MHR body / `MhrHandModel`
and the MANO correspondence (`parafit.mapping`) are implemented and tested.
`ContactSDFEnergy` is an interface stub slated for the next release.
Roadmap: multi-body scene (hand+object) -> CUDA/TensorRT backend. See
`docs/parafit-library-design.md`.

## License

MIT.
