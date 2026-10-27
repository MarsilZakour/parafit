"""Calibrate a :class:`ManoCorrespondence` for an articulated hand model.

Alternating least squares on synthetic MANO hands (PCA pose prior, random betas):

1. start from the hand's *native* landmarks (each of its own 21 OpenPose joints
   rides exactly on one of its bones),
2. fit the hand (root, articulation, uniform scale) to the MANO landmarks of every
   sample with the current correspondence,
3. re-solve, for every MANO landmark, the bone and in-bone offset that best explain
   it across all fitted samples (closed form per bone, best bone wins),
4. repeat 2-3; finally fit held-out MANO hands from scratch and report the
   residual (the representation gap between the two models), and build the dense
   MANO-vertex -> target-triangle map at the flat hand.
"""
from __future__ import annotations

import time
from typing import Optional

import torch
from torch import Tensor

from parafit.core.lie import rotation_to_axis_angle, so3_exp
from parafit.core.solver import LMSolver
from parafit.energies.anchor3d import Anchor3DEnergy
from parafit.energies.pose_prior import PosePriorEnergy
from parafit.mapping.mano_correspondence import ManoConventionHand, ManoCorrespondence, hand_scale_between


# ------------------------------------------------------------------------------------------------ geometry helpers
def rigid_align(source: Tensor, target: Tensor) -> tuple[Tensor, Tensor]:
    """Kabsch: rotation (B,3,3) and translation (B,3) with target ~= R @ source + t. Inputs (B,N,3)."""
    mu_s, mu_t = source.mean(1, keepdim=True), target.mean(1, keepdim=True)
    H = (source - mu_s).transpose(1, 2) @ (target - mu_t)
    U, _, Vh = torch.linalg.svd(H)
    V = Vh.transpose(1, 2)
    D = torch.diag_embed(torch.ones_like(U[:, 0]))
    D[:, 2, 2] = torch.sign(torch.det(V @ U.transpose(1, 2)))
    R = V @ D @ U.transpose(1, 2)
    return R, (mu_t - mu_s @ R.transpose(1, 2)).squeeze(1)


def closest_points_on_triangles(points: Tensor, triangles: Tensor, chunk: int = 256) -> tuple[Tensor, Tensor, Tensor]:
    """Closest point on a triangle soup for every query point.

    points (P,3), triangles (F,3,3) -> face index (P,), barycentric coordinates (P,3), distance (P,).
    Region-based closest point on a triangle (Ericson, Real-Time Collision Detection 5.1.5), vectorised.
    """
    a, b, c = triangles[:, 0], triangles[:, 1], triangles[:, 2]          # (F,3)
    ab, ac = b - a, c - a
    faces, barys, dists = [], [], []
    for p in points.split(chunk):
        ap = p[:, None] - a[None]; bp = p[:, None] - b[None]; cp = p[:, None] - c[None]     # (C,F,3)
        d1 = (ab * ap).sum(-1); d2 = (ac * ap).sum(-1)
        d3 = (ab * bp).sum(-1); d4 = (ac * bp).sum(-1)
        d5 = (ab * cp).sum(-1); d6 = (ac * cp).sum(-1)
        va = d3 * d6 - d5 * d4; vb = d5 * d2 - d1 * d6; vc = d1 * d4 - d3 * d2
        denom = (va + vb + vc).clamp_min(1e-20)
        v = vb / denom; w = vc / denom                                       # interior
        bary = torch.stack([1 - v - w, v, w], -1)
        def put(mask, value):
            nonlocal bary
            bary = torch.where(mask[..., None], value, bary)
        one, zero = torch.ones_like(d1), torch.zeros_like(d1)
        # edges
        t_ab = (d1 / (d1 - d3).clamp_min(1e-20)); put((vc <= 0) & (d1 >= 0) & (d3 <= 0), torch.stack([1 - t_ab, t_ab, zero], -1))
        t_ac = (d2 / (d2 - d6).clamp_min(1e-20)); put((vb <= 0) & (d2 >= 0) & (d6 <= 0), torch.stack([1 - t_ac, zero, t_ac], -1))
        t_bc = ((d4 - d3) / ((d4 - d3) + (d5 - d6)).clamp_min(1e-20))
        put((va <= 0) & ((d4 - d3) >= 0) & ((d5 - d6) >= 0), torch.stack([zero, 1 - t_bc, t_bc], -1))
        # vertices
        put((d1 <= 0) & (d2 <= 0), torch.stack([one, zero, zero], -1))
        put((d3 >= 0) & (d4 <= d3), torch.stack([zero, one, zero], -1))
        put((d6 >= 0) & (d5 <= d6), torch.stack([zero, zero, one], -1))
        closest = bary[..., 0:1] * a + bary[..., 1:2] * b + bary[..., 2:3] * c                # (C,F,3)
        dist = (closest - p[:, None]).norm(dim=-1)
        best = dist.argmin(1)
        idx = torch.arange(p.shape[0], device=p.device)
        faces.append(best); barys.append(bary[idx, best]); dists.append(dist[idx, best])
    return torch.cat(faces), torch.cat(barys), torch.cat(dists)


# ------------------------------------------------------------------------------------------------ MANO samples
def sample_mano_hands(mano, mano_assets_root: str, n: int, generator: torch.Generator, pose_std: float = 1.0,
                      beta_std: float = 1.0):
    """Random right MANO hands at identity root: landmarks (n,21,3), vertices (n,778,3), betas (n,10).

    Finger poses come from MANO's PCA hand-pose distribution (mean + z @ components, z ~ N(0, pose_std^2)),
    betas ~ N(0, beta_std^2). ``mano`` is a parafit ManoModel (flat_hand_mean=True, center_idx=0).
    """
    from manotorch.manolayer import ManoLayer
    pca = ManoLayer(joint_rot_mode="axisang", use_pca=True, ncomps=45, side="right",
                    mano_assets_root=mano_assets_root, flat_hand_mean=True)
    components = pca.th_selected_comps.float()                               # (45,45)
    mean = pca.th_hands_mean.float().reshape(45)
    z = torch.randn(n, 45, generator=generator) * pose_std
    fingers = mean + z @ components
    betas = torch.randn(n, 10, generator=generator) * beta_std
    params = torch.cat([torch.zeros(n, 3), fingers, torch.zeros(n, 3)], 1)
    mano.betas = betas
    with torch.no_grad():
        st = mano.forward(params)
    return st.landmarks, st.verts, betas


# ------------------------------------------------------------------------------------------------ fitting
def initial_params(model: ManoConventionHand, targets: Tensor, articulation: Optional[Tensor] = None) -> Tensor:
    """Root rotation/translation by rigid alignment of the (scaled) hand at ``articulation`` (default 0) to targets."""
    B = targets.shape[0]
    k = model.hand.articulation_slice.stop - model.hand.articulation_slice.start
    q = targets.new_zeros(B, k) if articulation is None else articulation
    with torch.no_grad():
        local = model.broadcast_scale(targets) * model.local_landmarks(q)
        R, t = rigid_align(local, targets)
    return model.join(rotation_to_axis_angle(R), q, t)


def fit_to_landmarks(model: ManoConventionHand, targets: Tensor, init: Optional[Tensor] = None, iterations: int = 40,
                     articulation_prior: float = 1e-6) -> Tensor:
    """Least-squares fit of the hand's MANO landmarks to ``targets`` (B,21,3) m; weak L2 pull of the articulation to 0."""
    init = initial_params(model, targets) if init is None else init
    P = model.n_params
    precision = targets.new_zeros(P)
    precision[model.hand.articulation_slice] = articulation_prior
    energies = [Anchor3DEnergy(targets), PosePriorEnergy(torch.zeros_like(init), precision=precision)]
    with torch.no_grad():
        return LMSolver(max_iters=iterations, damping=1e-3).solve(model, init, energies).params


def solve_bone_attachment(rotation: Tensor, origin: Tensor, scale: Tensor, targets: Tensor):
    """Best bone and in-bone offset for every landmark, across samples, in the hand-local frame.

    rotation (n,N,3,3), origin (n,N,3), scale (n,N): bone frames of n fitted samples; targets (n,21,3) the
    landmarks in the same frame. For a bone b the offset minimising sum_n |o_nb + s_nb R_nb p - y_n|^2 is
    p = sum_n s_nb R_nb^T (y_n - o_nb) / sum_n s_nb^2. Returns bone index (21,), offset (21,3), rms (21,).
    """
    d = targets[:, None] - origin[:, :, None]                                   # (n,N,21,3)
    local = torch.einsum("nbki,nbjk->nbji", rotation, d)                        # R^T d
    offset = (scale[..., None, None] * local).sum(0) / (scale ** 2).sum(0)[:, None, None]       # (N,21,3)
    pred = origin[:, :, None] + scale[..., None, None] * torch.einsum("nbik,bjk->nbji", rotation, offset)
    rms = ((pred - targets[:, None]) ** 2).sum(-1).mean(0).sqrt()               # (N,21)
    best = rms.argmin(0)                                                        # (21,)
    j = torch.arange(21, device=targets.device)
    return best, offset[best, j], rms[best, j]


def native_correspondence(hand, target: str, n_samples: int = 64, articulation_std: float = 0.3,
                          generator: Optional[torch.Generator] = None) -> ManoCorrespondence:
    """The hand's own 21 landmarks as a correspondence (each rides exactly on one bone)."""
    k = hand.articulation_slice.stop - hand.articulation_slice.start
    q = torch.randn(n_samples, k, generator=generator) * articulation_std
    q[0] = 0
    q = q.to(_device(hand))
    with torch.no_grad():
        rot, origin, scale = hand.local_bone_frames(q)
        native = hand.local_native_landmarks(q)
    bone, offset, _ = solve_bone_attachment(rot, origin, scale, native)
    empty = torch.zeros(0)
    return ManoCorrespondence(target, bone, offset, empty.long(), empty.reshape(0, 3))


def _device(hand) -> torch.device:
    for name in ("device", "_dev"):
        d = getattr(hand, name, None)
        if d is not None:
            return torch.device(d)
    return torch.device("cpu")


def calibrate_mano_correspondence(hand, mano, mano_assets_root: str, target: str, n_fit: int = 512, n_val: int = 256, rounds: int = 4,
                                  fit_iterations: int = 40, articulation_prior: float = 1e-6, seed: int = 0,
                                  log=print) -> ManoCorrespondence:
    """Calibrate the MANO correspondence of ``hand`` (an ArticulatedHand) against ``mano`` (parafit ManoModel)."""
    t0 = time.time()
    dev = _device(hand)
    gen = torch.Generator().manual_seed(seed)
    y_fit, _, _ = sample_mano_hands(mano, mano_assets_root, n_fit, gen)
    y_val, v_val, _ = sample_mano_hands(mano, mano_assets_root, n_val, gen)
    y_fit, y_val, v_val = y_fit.to(dev), y_val.to(dev), v_val.to(dev)

    corr = native_correspondence(hand, target, generator=gen)
    model = ManoConventionHand(hand, corr)
    k = hand.articulation_slice.stop - hand.articulation_slice.start
    params = None
    history = []
    for r in range(rounds + 1):
        with torch.no_grad():
            rest = model.local_landmarks(y_fit.new_zeros(1, k)).expand(n_fit, -1, -1)
        model.hand_scale = hand_scale_between(rest, y_fit)
        params = fit_to_landmarks(model, y_fit, init=None if params is None else params,
                                  iterations=fit_iterations, articulation_prior=articulation_prior)
        with torch.no_grad():
            err = (model.forward(params).landmarks - y_fit).norm(dim=-1).mean().item() * 1000
        history.append(round(err, 3))
        log(f"[calib {target}] round {r}: fit residual {err:.2f} mm")
        if r == rounds:
            break
        w, q, t = model.split(params)
        sigma = model.broadcast_scale(params)
        y_local = torch.einsum("bki,bjk->bji", so3_exp(w), y_fit - t[:, None]) / sigma     # R^T (y - t) / sigma
        with torch.no_grad():
            rot, origin, scale = hand.local_bone_frames(q)
        bone, offset, _ = solve_bone_attachment(rot, origin, scale, y_local)
        corr = ManoCorrespondence(target, bone, offset, corr.vertex_face, corr.vertex_barycentric)
        model.correspondence = corr

    # held-out: fit from scratch with the final correspondence
    with torch.no_grad():
        rest = model.local_landmarks(y_val.new_zeros(1, k)).expand(n_val, -1, -1)
    model.hand_scale = hand_scale_between(rest, y_val)
    p_val = fit_to_landmarks(model, y_val, iterations=fit_iterations, articulation_prior=articulation_prior)
    with torch.no_grad():
        val_joint = (model.forward(p_val).landmarks - y_val).norm(dim=-1).mean().item() * 1000

    # dense map at the flat mean hand
    mano.betas = torch.zeros(1, 10)
    flat = torch.zeros(1, mano.n_params)
    with torch.no_grad():
        flat_state = mano.forward(flat)
        y0, v0 = flat_state.landmarks.to(dev), flat_state.verts.to(dev)
        rest = model.local_landmarks(y0.new_zeros(1, k))
        model.hand_scale = hand_scale_between(rest, y0)
        p0 = fit_to_landmarks(model, y0, iterations=fit_iterations, articulation_prior=articulation_prior)
        mesh0 = model.mesh(p0)[0]
        face, bary, dist = closest_points_on_triangles(v0[0], mesh0[hand.faces.to(dev)])
    corr = ManoCorrespondence(target, corr.bone_index, corr.bone_offset, face, bary)
    model.correspondence = corr
    # held-out surface gap through the dense map
    model.hand_scale = hand_scale_between(model.local_landmarks(y_val.new_zeros(1, k)).expand(n_val, -1, -1), y_val)
    with torch.no_grad():
        val_vertex = (model.mano_vertices(p_val) - v_val).norm(dim=-1).mean().item() * 1000

    corr.calibration = {
        "seed": seed, "n_fit": n_fit, "n_val": n_val, "rounds": rounds, "fit_iterations": fit_iterations,
        "articulation_prior": articulation_prior, "fit_residual_mm_per_round": history,
        "heldout_joint_residual_mm": round(val_joint, 3), "heldout_vertex_residual_mm": round(val_vertex, 3),
        "flat_hand_vertex_to_surface_mm": round(dist.mean().item() * 1000, 3), "seconds": round(time.time() - t0, 1),
    }
    log(f"[calib {target}] held-out joint residual {val_joint:.2f} mm, vertex residual {val_vertex:.2f} mm, "
        f"flat-hand vertex->surface {dist.mean().item() * 1000:.2f} mm")
    return corr
