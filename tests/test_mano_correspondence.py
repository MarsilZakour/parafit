"""ManoConventionHand: landmarks attached to bones of an articulated hand (UmeTrack, synthetic profile)."""
from __future__ import annotations

import torch

from parafit import LMSolver
from parafit.energies.anchor3d import Anchor3DEnergy
from parafit.mapping import ManoConventionHand, ManoCorrespondence, hand_scale_between
from parafit.mapping.calibration import closest_points_on_triangles, rigid_align, solve_bone_attachment
from parafit.models.umetrack import UmeTrackModel

from test_umetrack_model import _synthetic_hand


def _wrapped(seed: int = 0) -> ManoConventionHand:
    hand = UmeTrackModel(_synthetic_hand(seed), dtype=torch.float64)
    g = torch.Generator().manual_seed(seed)
    corr = ManoCorrespondence("synthetic", torch.randint(0, 17, (21,), generator=g),
                              0.01 * torch.randn(21, 3, generator=g, dtype=torch.float64),
                              torch.zeros(778, dtype=torch.long), torch.full((778, 3), 1.0 / 3, dtype=torch.float64))
    return ManoConventionHand(hand, corr)


def _random_params(model, B, g):
    p = 0.3 * torch.randn(B, model.n_params, generator=g, dtype=torch.float64)
    p[:, -3:] *= 0.1
    return p


def test_jacobian_matches_finite_differences():
    model = _wrapped()
    model.hand_scale = torch.tensor([0.9, 1.1], dtype=torch.float64)
    g = torch.Generator().manual_seed(1)
    params = _random_params(model, 2, g)
    J = model.landmark_jacobian(params, model.forward(params))
    eps, fd = 1e-6, torch.zeros_like(J)
    for k in range(model.n_params):
        d = torch.zeros_like(params); d[:, k] = eps
        fd[..., k] = (model.forward(params + d).landmarks - model.forward(params - d).landmarks) / (2 * eps)
    assert ((J - fd).norm() / fd.norm()) < 1e-6


def test_solve_bone_attachment_recovers_exact_attachment():
    model = _wrapped()
    g = torch.Generator().manual_seed(2)
    q = 0.4 * torch.randn(64, 22, generator=g, dtype=torch.float64)
    with torch.no_grad():
        rot, origin, scale = model.hand.local_bone_frames(q)
        targets = model.local_landmarks(q)
    bone, offset, rms = solve_bone_attachment(rot, origin, scale, targets)
    assert rms.max() < 1e-9
    recon = origin[:, bone] + scale[:, bone, None] * torch.einsum("njik,jk->nji", rot[:, bone], offset)
    assert (recon - targets).abs().max() < 1e-9


def test_overfit_and_scale_transfer():
    model = _wrapped()
    g = torch.Generator().manual_seed(3)
    truth = _random_params(model, 4, g)
    model.hand_scale = torch.full((4,), 1.07, dtype=torch.float64)
    with torch.no_grad():
        target = model.forward(truth).landmarks
    rest = model.local_landmarks(torch.zeros(1, 22, dtype=torch.float64)).expand(4, -1, -1)
    assert torch.allclose(hand_scale_between(rest, 1.07 * rest), torch.full((4,), 1.07, dtype=torch.float64))
    R, t = rigid_align(model.broadcast_scale(target) * rest, target)
    from parafit.core.lie import rotation_to_axis_angle
    init = model.join(rotation_to_axis_angle(R), torch.zeros(4, 22, dtype=torch.float64), t)
    fit = LMSolver(max_iters=60, damping=1e-3).solve(model, init, [Anchor3DEnergy(target)]).params
    assert (model.forward(fit).landmarks - target).norm(dim=-1).max() < 1e-4


def test_closest_point_on_triangle_regions():
    tri = torch.tensor([[[0.0, 0, 0], [1, 0, 0], [0, 1, 0]]])
    pts = torch.tensor([[0.2, 0.2, 1.0], [-1.0, -1.0, 0], [2.0, -0.5, 0], [0.8, 0.8, 0]])
    _, bary, dist = closest_points_on_triangles(pts, tri)
    closest = torch.einsum("pc,cd->pd", bary, tri[0])
    expected = torch.tensor([[0.2, 0.2, 0], [0, 0, 0], [1, 0, 0], [0.5, 0.5, 0]])
    assert torch.allclose(closest, expected, atol=1e-6)
    assert torch.allclose(dist, (pts - expected).norm(dim=-1), atol=1e-6)
