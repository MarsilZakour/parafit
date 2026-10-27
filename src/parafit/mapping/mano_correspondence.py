"""MANO correspondence: give any articulated hand model MANO's 21 landmarks.

Networks trained on MANO (UA-Fit, HaMeR, WiLoR, ...) predict MANO's joints: the
16 joints regressed from the MANO mesh plus 5 fingertip vertices, in OpenPose
order. Those points are *not* the skeleton joints of another hand model (MHR,
UmeTrack): MANO's MCP "joint", for example, sits at a different place inside the
palm than MHR's ``index1`` joint. Fitting another model to MANO-convention
evidence therefore needs the MANO landmarks expressed on that model.

A :class:`ManoCorrespondence` stores, for each MANO landmark ``j``, the bone
``b_j`` of the target hand it rides on and a fixed offset ``p_j`` in that
bone's frame. :class:`ManoConventionHand` wraps any :class:`ArticulatedHand`
into a parafit :class:`~parafit.models.base.Model` whose landmarks are

    X_j = t + sigma * R(w) * (o_b + s_b * R_b @ p_j),    b = b_j,

with root rotation ``w`` (axis-angle), root translation ``t``, a uniform hand
scale ``sigma`` (shape transfer from MANO betas) and the bone frames
``(R_b, o_b, s_b)`` of the articulated hand. The correspondence also carries a
dense map (MANO vertex -> barycentric point on the target mesh) so a fitted
target mesh can be compared to MANO ground truth vertex by vertex.

The correspondence is calibrated once per target model by
:func:`parafit.mapping.calibration.calibrate_mano_correspondence`.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Optional, Protocol

import numpy as np
import torch
from torch import Tensor

from parafit.core.lie import point_rotation_jacobian, so3_exp
from parafit.core.types import ParamSpec, State
from parafit.models.base import Model

#: OpenPose parent of each of the 21 hand landmarks (wrist = root).
OPENPOSE_PARENTS = [-1, 0, 1, 2, 3, 0, 5, 6, 7, 0, 9, 10, 11, 0, 13, 14, 15, 0, 17, 18, 19]


class ArticulatedHand(Protocol):
    """What :class:`ManoConventionHand` needs from a hand model.

    Parameters are ``[root rotation (3) | articulation (k) | root translation (3)]``
    (``param_spec`` blocks ``root_rot``, the articulation block, ``root_trans``).
    All ``local_*`` outputs are wrist-local, in metres, at zero root rotation and
    translation and unit hand scale.
    """

    param_spec: ParamSpec
    faces: Optional[Tensor]

    @property
    def articulation_slice(self) -> slice: ...

    def local_bone_frames(self, articulation: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """(B,k) -> rotation (B,N,3,3), origin (B,N,3), scale (B,N) of the N bones."""

    def local_native_landmarks(self, articulation: Tensor) -> Tensor:
        """(B,k) -> the model's own 21 OpenPose-ordered landmarks (B,21,3)."""

    def local_mesh(self, articulation: Tensor) -> Tensor:
        """(B,k) -> mesh vertices (B,V,3)."""


@dataclass
class ManoCorrespondence:
    """MANO landmarks and vertices expressed on a target hand model."""

    target: str                     # e.g. "mhr_right", "umetrack_right"
    bone_index: Tensor              # (21,) long, index into the hand's bone list
    bone_offset: Tensor             # (21,3) metres, in the bone frame, at unit hand scale
    vertex_face: Tensor             # (778,) long, target-mesh triangle carrying each MANO vertex
    vertex_barycentric: Tensor      # (778,3) barycentric coordinates in that triangle
    calibration: dict = field(default_factory=dict)   # residuals / settings of the calibration run

    def to(self, device) -> "ManoCorrespondence":
        return ManoCorrespondence(self.target, self.bone_index.to(device), self.bone_offset.to(device),
                                  self.vertex_face.to(device), self.vertex_barycentric.to(device), self.calibration)

    def save(self, path: str) -> None:
        np.savez(path, target=self.target, bone_index=self.bone_index.cpu().numpy(),
                 bone_offset=self.bone_offset.cpu().numpy(), vertex_face=self.vertex_face.cpu().numpy(),
                 vertex_barycentric=self.vertex_barycentric.cpu().numpy(),
                 calibration=np.array(repr(self.calibration)))

    @classmethod
    def load(cls, path: str, device: str = "cpu", dtype: torch.dtype = torch.float32) -> "ManoCorrespondence":
        d = np.load(path, allow_pickle=False)
        import ast
        return cls(target=str(d["target"]),
                   bone_index=torch.as_tensor(d["bone_index"], dtype=torch.long, device=device),
                   bone_offset=torch.as_tensor(d["bone_offset"], dtype=dtype, device=device),
                   vertex_face=torch.as_tensor(d["vertex_face"], dtype=torch.long, device=device),
                   vertex_barycentric=torch.as_tensor(d["vertex_barycentric"], dtype=dtype, device=device),
                   calibration=ast.literal_eval(str(d["calibration"])))


def bone_lengths(landmarks: Tensor) -> Tensor:
    """(B,21,3) OpenPose landmarks -> (B,20) lengths of the 20 parent-child segments."""
    child = torch.arange(1, 21, device=landmarks.device)
    parent = torch.tensor(OPENPOSE_PARENTS[1:], device=landmarks.device)
    return (landmarks[:, child] - landmarks[:, parent]).norm(dim=-1)


def hand_scale_between(source: Tensor, reference: Tensor) -> Tensor:
    """Uniform scale (B,) that matches the total segment length of ``source`` (B,21,3) to ``reference`` (B,21,3).
    Pose-invariant, so the two hands need not be in the same pose."""
    return bone_lengths(reference).sum(-1) / bone_lengths(source).sum(-1).clamp_min(1e-9)


class ManoConventionHand(Model):
    """An :class:`ArticulatedHand` whose landmarks are MANO's 21 mesh-regressed joints (OpenPose order).

    ``hand_scale`` (float or (B,) tensor) is the uniform shape scale about the wrist; set it per batch
    (e.g. with :func:`hand_scale_between` from the MANO betas a network predicted).
    """

    def __init__(self, hand: ArticulatedHand, correspondence: ManoCorrespondence):
        self.hand = hand
        self.correspondence = correspondence
        self.param_spec = hand.param_spec
        self.hand_scale: float | Tensor = 1.0
        self.num_joints = 21

    # -------------------------------------------------------------- parameter layout
    def split(self, params: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """params (B,P) -> root rotation (B,3), articulation (B,k), root translation (B,3)."""
        return params[:, :3], params[:, self.hand.articulation_slice], params[:, -3:]

    def join(self, root_rotation: Tensor, articulation: Tensor, root_translation: Tensor) -> Tensor:
        return torch.cat([root_rotation, articulation, root_translation], dim=-1)

    def zero_params(self, batch_size: int, device=None, dtype=torch.float32) -> Tensor:
        return torch.zeros(batch_size, self.n_params, device=device, dtype=dtype)

    def broadcast_scale(self, ref: Tensor) -> Tensor:
        """hand_scale shaped to multiply (B,N,3) points: (B,1,1) per-sample or a 0-dim tensor."""
        s = self.hand_scale
        s = torch.as_tensor(s, dtype=ref.dtype, device=ref.device)
        return s.reshape(-1, 1, 1) if s.ndim else s

    # -------------------------------------------------------------- landmarks
    def local_landmarks(self, articulation: Tensor) -> Tensor:
        """MANO landmarks (B,21,3) on the hand, wrist-local, unit scale."""
        rot, origin, scale = self.hand.local_bone_frames(articulation)
        b = self.correspondence.bone_index
        p = self.correspondence.bone_offset.to(origin)
        if hasattr(self.hand, "deform_bone_points"):      # hands with a rest-shape model move attached points with their bone
            p = self.hand.deform_bone_points(p, b, articulation.shape[0])
            return origin[:, b] + scale[:, b, None] * torch.einsum("bjik,bjk->bji", rot[:, b], p)
        return origin[:, b] + scale[:, b, None] * torch.einsum("bjik,jk->bji", rot[:, b], p)

    def forward(self, params: Tensor) -> State:
        w, q, t = self.split(params)
        local = self.broadcast_scale(params) * self.local_landmarks(q)
        landmarks = torch.einsum("bik,bjk->bji", so3_exp(w), local) + t[:, None]
        return State(landmarks=landmarks, batch_size=[params.shape[0]])

    def _articulation_jacobian(self, articulation: Tensor) -> Tensor:
        """d local_landmarks / d articulation, (B,21,3,k), forward-mode autograd (exact)."""
        from torch.func import jacfwd, vmap

        def single(q: Tensor) -> Tensor:
            return self.local_landmarks(q[None])[0]
        try:
            return vmap(jacfwd(single))(articulation)
        except RuntimeError:          # forward not vmappable: per-sample
            return torch.stack([jacfwd(single)(q) for q in articulation])

    def landmark_jacobian(self, params: Tensor, state: State) -> Tensor:
        w, q, _ = self.split(params)
        B, P = params.shape
        sigma = self.broadcast_scale(params)
        local = sigma * self.local_landmarks(q)
        R = so3_exp(w)
        J = params.new_zeros(B, 21, 3, P)
        J[..., :3] = point_rotation_jacobian(w, local)
        J[..., self.hand.articulation_slice] = torch.einsum("bik,bjkp->bjip", R, sigma[..., None] * self._articulation_jacobian(q))
        J[..., -3:] = torch.eye(3, dtype=params.dtype, device=params.device)
        return J

    # -------------------------------------------------------------- surfaces
    def mesh(self, params: Tensor) -> Tensor:
        """The target hand's own mesh (B,V,3) at ``params`` (world, metres)."""
        w, q, t = self.split(params)
        local = self.broadcast_scale(params) * self.hand.local_mesh(q)
        return torch.einsum("bik,bvk->bvi", so3_exp(w), local) + t[:, None]

    def mano_vertices(self, params: Tensor, mesh: Optional[Tensor] = None) -> Tensor:
        """MANO-topology vertices (B,778,3) sampled on the target mesh through the dense correspondence."""
        mesh = self.mesh(params) if mesh is None else mesh
        tri = self.hand.faces.to(mesh.device)[self.correspondence.vertex_face]       # (778,3)
        corners = mesh[:, tri]                                                         # (B,778,3,3)
        return torch.einsum("bvcd,vc->bvd", corners, self.correspondence.vertex_barycentric.to(mesh))
