"""UmeTrackModel: FK/LBS correctness, exact Jacobian, mirroring, overfit, batching.

No real UmeTrack hand-model json is vendored (CC-BY-NC 4.0, non-commercial; see the README), so most
tests build a structurally-valid *synthetic* profile (`_synthetic_hand`) in-process: same topology (22
joints / 17 bone frames / 21 native landmarks, see `parafit.models.umetrack`'s module docstring) as a
real UmeTrack hand, but arbitrary rest geometry -- enough to exercise the FK/LBS math without the asset.

`test_forward_matches_independent_numpy_reference` re-derives UmeTrack's FK+LBS
(facebookresearch/UmeTrack ``lib/common/hand_skinning.py``) independently in plain numpy, sharing no
code with `UmeTrackModel._fk`, and checks both sides. An optional extra check against a REAL profile
runs if `UMETRACK_PROFILE_JSON` points at one (skips cleanly otherwise).
"""
from __future__ import annotations

import os

import numpy as np
import pytest
import torch

from parafit import LMSolver
from parafit.energies.anchor3d import Anchor3DEnergy
from parafit.models.umetrack import (
    DOF_PER_FINGER,
    NUM_BONE_FRAMES,
    NUM_DIGITS,
    NUM_JOINTS,
    UmeTrackHand,
    UmeTrackModel,
    load_umetrack_hand,
)

DT = torch.float64

# Fixed UmeTrack native-landmark -> bone-frame format (see umetrack.py's `_NATIVE_NAMES` /
# module docstring): tips(0-4) bind to each finger's depth-3 bone, wrist(5) and every
# "proximal"/MCP point (8,11,14,17) bind to the wrist bone(1), and the rest bind to the
# depth-1/2 bones. Used only to build a *structurally* valid synthetic profile below.
_NATIVE_BONE = [4, 7, 10, 13, 16, 1, 2, 3, 1, 5, 6, 1, 8, 9, 1, 11, 12, 1, 14, 15, 1]


def _synthetic_hand(seed: int = 0, n_verts: int = 40) -> UmeTrackHand:
    g = np.random.RandomState(seed)
    joint_rest = np.zeros((NUM_JOINTS, 3))
    axes = np.zeros((NUM_JOINTS, 3))
    for f in range(NUM_DIGITS):
        origin = np.array([0.015 * f, 0.0, 0.0])
        for k in range(DOF_PER_FINGER):
            j = f * DOF_PER_FINGER + k
            joint_rest[j] = origin + np.array([0.0, 0.02 * (k + 1), 0.0]) + 0.002 * g.randn(3)
            a = np.array([1.0, 0.0, 0.0]) if k % 2 == 0 else np.array([0.3, 0.0, 1.0])
            a = a + 0.05 * g.randn(3)
            axes[j] = a / np.linalg.norm(a)
    axes[20] = [1.0, 0.0, 0.0]; axes[21] = [0.0, 1.0, 0.0]        # inert joints; direction irrelevant

    limits = np.stack([-np.full(NUM_JOINTS, np.pi / 2), np.full(NUM_JOINTS, np.pi / 2)], axis=1)
    parent = np.full(NUM_JOINTS, -1, dtype=np.int64)
    for f in range(NUM_DIGITS):
        for k in range(1, DOF_PER_FINGER):
            parent[f * DOF_PER_FINGER + k] = f * DOF_PER_FINGER + k - 1

    lm_w = np.zeros((21, NUM_BONE_FRAMES)); lm_rest = np.zeros((21, 3))
    for i, b in enumerate(_NATIVE_BONE):
        lm_w[i, b] = 1.0
        lm_rest[i] = joint_rest[min(i, NUM_JOINTS - 1)] + 0.01 * g.randn(3)

    mesh_w = g.rand(n_verts, NUM_BONE_FRAMES)
    mesh_w /= mesh_w.sum(1, keepdims=True)            # dense, non-trivial multi-bone blend
    mesh_rest = 0.05 * g.randn(n_verts, 3)
    faces = np.zeros((1, 3), dtype=np.int64)          # unused by these tests

    t = lambda a, dt=torch.float32: torch.as_tensor(np.asarray(a), dtype=dt)
    return UmeTrackHand(
        joint_rotation_axes=t(axes), joint_rest_positions=t(joint_rest),
        joint_parent=t(parent, torch.long), joint_limits=t(limits),
        mesh_vertices=t(mesh_rest), faces=t(faces, torch.long),
        dense_bone_weights=t(mesh_w), landmark_rest_positions=t(lm_rest),
        landmark_bone_weights=t(lm_w),
    )


# --- independent numpy FK+LBS reference (no code shared with UmeTrackModel._fk) --------------------

def _np_rodrigues(av: np.ndarray) -> np.ndarray:
    th = np.linalg.norm(av)
    if th < 1e-12:
        return np.eye(3)
    k = av / th
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * (K @ K)


def _np_local_xf(axis: np.ndarray, rest: np.ndarray, angle: float) -> np.ndarray:
    R = _np_rodrigues(axis * angle)
    T = np.eye(4); T[:3, :3] = R; T[:3, 3] = rest - R @ rest
    return T


def _np_skinning_frames(axes: np.ndarray, rest: np.ndarray, angles: np.ndarray, wrist: np.ndarray) -> np.ndarray:
    """-> (17,4,4); independent re-derivation of UmeTrack's hand_skinning.py FK."""
    frames = [wrist, wrist]
    for f in range(NUM_DIGITS):
        chain = [wrist]
        for k in range(DOF_PER_FINGER):
            j = f * DOF_PER_FINGER + k
            chain.append(chain[-1] @ _np_local_xf(axes[j], rest[j], angles[j]))
        frames += chain[2:]           # keep "after joints {0,1}/{0,1,2}/{0,1,2,3}"; drop chain[0],chain[1]
    return np.stack(frames)


def _np_lbs(frames: np.ndarray, points: np.ndarray, weights: np.ndarray) -> np.ndarray:
    homo = np.concatenate([points, np.ones((len(points), 1))], 1)
    posed = np.einsum("kij,vj->vki", frames, homo)
    return np.einsum("vk,vki->vi", weights, posed)[:, :3]


def _np_forward(axes, rest, pt_rest, pt_w, angles, wristR, wristT):
    wrist = np.eye(4); wrist[:3, :3] = wristR; wrist[:3, 3] = wristT
    frames = _np_skinning_frames(axes, rest, angles, wrist)
    return _np_lbs(frames, pt_rest, pt_w)


# --- (1) FK/LBS correctness, both sides -------------------------------------------------------------

@pytest.mark.parametrize("side", ["right", "left"])
def test_forward_matches_independent_numpy_reference(side):
    hand = _synthetic_hand()
    m = UmeTrackModel(hand, side=side, dtype=DT, with_mesh=True)
    axes_np, rest_np = m.axes.numpy(), m.rest.numpy()
    op_rest_np, op_w_np = m.op_rest.numpy(), m.op_bone_weights.numpy()
    mesh_rest_np, mesh_w_np = m.mesh_vertices.numpy(), m.dense_bone_weights.numpy()

    g = np.random.RandomState(3)
    for _ in range(5):
        angles = np.zeros(NUM_JOINTS); angles[:20] = 0.4 * g.randn(20)
        root_rot = 0.3 * g.randn(3)
        root_trans = 0.1 * g.randn(3)
        wristR = _np_rodrigues(root_rot)

        lm_ref = _np_forward(axes_np, rest_np, op_rest_np, op_w_np, angles, wristR, root_trans)
        mesh_ref = _np_forward(axes_np, rest_np, mesh_rest_np, mesh_w_np, angles, wristR, root_trans)

        theta = torch.zeros(1, m.n_params, dtype=DT)
        theta[0, 0:3] = torch.as_tensor(root_rot)
        theta[0, 3:23] = torch.as_tensor(angles[:20])
        theta[0, 25:28] = torch.as_tensor(root_trans)
        st = m.forward(theta)

        assert np.abs(st.landmarks[0].numpy() - lm_ref).max() < 1e-6
        assert np.abs(st.verts[0].numpy() - mesh_ref).max() < 1e-6


# --- (2) exact Jacobian vs finite differences -------------------------------------------------------

def test_landmark_jacobian_matches_finite_differences():
    hand = _synthetic_hand(seed=1)
    m = UmeTrackModel(hand, side="right", dtype=DT, with_mesh=False)
    torch.manual_seed(0)
    B = 3
    theta = torch.zeros(B, m.n_params, dtype=DT)
    theta[:, 0:3] = 0.2 * torch.randn(B, 3, dtype=DT)
    theta[:, 3:23] = 0.3 * torch.randn(B, 20, dtype=DT)
    theta[:, 25:28] = 0.1 * torch.randn(B, 3, dtype=DT)

    state = m.forward(theta)
    Jan = m.landmark_jacobian(theta, state)
    assert Jan.shape == (B, m.num_landmarks, 3, m.n_params)

    eps = 1e-6
    Jfd = torch.zeros_like(Jan)
    for k in range(m.n_params):
        d = torch.zeros_like(theta); d[:, k] = eps
        Jfd[..., k] = (m.forward(theta + d).landmarks - m.forward(theta - d).landmarks) / (2 * eps)

    rel = (Jan - Jfd).norm() / Jfd.norm()
    assert rel < 1e-6, rel


def test_jacobian_is_zero_for_joints_outside_depth_and_for_inert_slots():
    """Pinned landmarks (wrist/MCPs/synthesized thumb CMC) must have exactly zero gradient w.r.t. any
    finger joint angle, and the two inert joint slots (20,21) must have zero gradient everywhere."""
    hand = _synthetic_hand(seed=2)
    m = UmeTrackModel(hand, side="right", dtype=DT, with_mesh=False)
    theta = 0.3 * torch.randn(2, m.n_params, dtype=DT)
    theta[:, 0:3] = 0.0  # isolate the finger-joint columns from the root-rotation block
    J = m.landmark_jacobian(theta, m.forward(theta))

    pinned = [i for i in range(m.num_landmarks) if m._op_depth[i] == 0]
    assert torch.count_nonzero(J[:, pinned, :, 3:23]) == 0
    assert torch.count_nonzero(J[:, :, :, 23:25]) == 0  # inert joint slots 20,21


# --- (3) LM overfit -----------------------------------------------------------------------------------

def test_overfits_to_a_random_pose_under_1mm():
    hand = _synthetic_hand(seed=3)
    m = UmeTrackModel(hand, side="right", dtype=DT, with_mesh=False)
    torch.manual_seed(4)
    B = 1
    gt = torch.zeros(B, m.n_params, dtype=DT)
    gt[:, 0:3] = 0.2 * torch.randn(B, 3, dtype=DT)
    gt[:, 3:23] = 0.3 * torch.randn(B, 20, dtype=DT)
    gt[:, 25:28] = 0.05 * torch.randn(B, 3, dtype=DT)
    target = m.forward(gt).landmarks

    init = gt.clone()
    init[:, 0:3] += 0.1
    init[:, 3:23] += 0.1
    init[:, 25:28] += 0.02

    res = LMSolver(max_iters=60, damping=1e-3).solve(m, init, [Anchor3DEnergy(target=target)])
    err_mm = (m.forward(res.params).landmarks - target).norm(dim=-1).max().item() * 1000
    assert err_mm < 1.0, err_mm


# --- (4) batch independence ---------------------------------------------------------------------------

def test_batch_independence():
    hand = _synthetic_hand(seed=4)
    m = UmeTrackModel(hand, side="right", dtype=DT, with_mesh=True)
    torch.manual_seed(5)
    B = 4
    theta = torch.zeros(B, m.n_params, dtype=DT)
    theta[:, 0:3] = 0.3 * torch.randn(B, 3, dtype=DT)
    theta[:, 3:23] = 0.3 * torch.randn(B, 20, dtype=DT)
    theta[:, 25:28] = 0.1 * torch.randn(B, 3, dtype=DT)

    st_batch = m.forward(theta)
    for b in range(B):
        st_single = m.forward(theta[b: b + 1])
        torch.testing.assert_close(st_batch.landmarks[b: b + 1], st_single.landmarks)
        torch.testing.assert_close(st_batch.verts[b: b + 1], st_single.verts)


# --- (5) mirror consistency ----------------------------------------------------------------------------

def test_mirror_consistency():
    """left(theta) == mirror_x(right(mirror_theta(theta))), where `mirror_theta` negates the root
    rotation's y/z axis-angle components and the root translation's x component (the induced action of
    an x-reflection on an axis-angle rotation generator and on a translation vector, respectively);
    finger joint angles are untouched since UmeTrack bakes handedness entirely into the geometry
    (`_mirror_hand`), not into the angle convention."""
    hand = _synthetic_hand(seed=5)
    mr = UmeTrackModel(hand, side="right", dtype=DT)
    ml = UmeTrackModel(hand, side="left", dtype=DT)

    torch.manual_seed(6)
    B = 3
    theta = torch.zeros(B, mr.n_params, dtype=DT)
    theta[:, 0:3] = 0.3 * torch.randn(B, 3, dtype=DT)
    theta[:, 3:23] = 0.3 * torch.randn(B, 20, dtype=DT)
    theta[:, 25:28] = 0.1 * torch.randn(B, 3, dtype=DT)

    mirror_theta = theta.clone()
    mirror_theta[:, 1] *= -1; mirror_theta[:, 2] *= -1
    mirror_theta[:, 25] *= -1

    left = ml.forward(theta).landmarks
    right_mirrored = mr.forward(mirror_theta).landmarks.clone()
    right_mirrored[..., 0] *= -1

    torch.testing.assert_close(left, right_mirrored, rtol=1e-10, atol=1e-10)


# --- optional: real SHOW3D profile (skips cleanly if not available) ------------------------------------

def test_real_profile_if_available():
    path = os.environ.get("UMETRACK_PROFILE_JSON", "")
    if not path or not os.path.exists(path):
        pytest.skip("set UMETRACK_PROFILE_JSON to a real profile_umetrack.json for this extra check")
    hand = load_umetrack_hand(path)
    m = UmeTrackModel(hand, side="right", dtype=DT)
    theta = torch.zeros(1, m.n_params, dtype=DT)
    st = m.forward(theta)
    span = (st.landmarks.max(1).values - st.landmarks.min(1).values).norm(dim=-1)
    assert (span > 0.05).all() and (span < 0.4).all(), span
